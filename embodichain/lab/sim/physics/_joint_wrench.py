# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""DexSim runtime control exposing completed Newton parent reactions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.runtime_control import (
    RuntimeControl,
    RuntimeControlContext,
)

from embodichain.utils.device_utils import current_warp_stream

if TYPE_CHECKING:
    from embodichain.lab.sim.sensors.base_sensor import SensorCfg
    from dexsim.scene import Scene, SpawnedArticulation
    from newton import Model


@wp.kernel(enable_backward=False)
def _sample_joint_reaction(
    ids: wp.array(dtype=wp.int32),
    report: wp.array(dtype=wp.spatial_vector),
    body_pose: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    sensor_from_body: wp.quat,
    sensor_position: wp.vec3,
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
    validity: wp.array(dtype=wp.bool),
):
    row = wp.tid()
    body = ids[row]
    wrench = report[body]
    inv_rotation = wp.quat_inverse(wp.transform_get_rotation(body_pose[body]))
    force_body = wp.quat_rotate(inv_rotation, wp.spatial_top(wrench))
    torque_body = wp.quat_rotate(inv_rotation, wp.spatial_bottom(wrench))
    torque_body += wp.cross(body_com[body] - sensor_position, force_body)
    force = wp.quat_rotate(sensor_from_body, force_body)
    torque = wp.quat_rotate(sensor_from_body, torque_body)
    valid = wp.isfinite(force) and wp.isfinite(torque)
    forces[row] = wp.where(valid, force, wp.vec3(0.0))
    torques[row] = wp.where(valid, torque, wp.vec3(0.0))
    validity[row] = valid


class _JointWrenchSource(RuntimeControl):
    def __init__(self) -> None:
        self._completed: RuntimeControlContext | None = None
        self._stream: torch.cuda.Stream | None = None

    def initialize_model(self, model: Model, result: Scene) -> None:
        # DexSim creates the solver after this hook. Newton 1.7's state
        # attribute request is the path available before solver creation.
        model.request_state_attributes("body_parent_f")

    def substep_callbacks(self) -> tuple[()]:
        return ()

    def after_step(self, context: RuntimeControlContext, step_dt: float) -> None:
        self._completed = context
        stream = wp.get_stream(context.device)
        if self._stream is None or self._stream.cuda_stream != stream.cuda_stream:
            self._stream = (
                wp.stream_to_torch(stream)
                if stream.cuda_stream
                else torch.cuda.default_stream(str(context.device))
            )

    def prepare_episode_reset(
        self, arena_indices: tuple[int, ...]
    ) -> Callable[[], None]:
        # Only sensor-owned rows have history. Reset never republishes native state.
        return lambda: None

    def close(self) -> None:
        self._completed = None
        self._stream = None

    def create_reader(
        self,
        handles: Sequence[SpawnedArticulation],
        joint_name: str,
        offset: SensorCfg.OffsetCfg,
        force: torch.Tensor,
        torque: torch.Tensor,
        valid: torch.Tensor,
    ) -> Callable[[], None]:
        ids: list[int] = []
        for handle in handles:
            child = handle.get_child_link_name(joint_name)
            if not child or not handle.get_parent_link_name(joint_name):
                raise ValueError(
                    f"Joint {joint_name!r} must connect two links on {handle.path!r}."
                )
            # The Spawn batch API has no native per-link ID. Use the same
            # articulation binding as DexSim's Newton runtime controls.
            binding = handle._physics_binding
            if not any(
                joint.name == joint_name and joint.child_link_name == child
                for joint in getattr(binding, "joints", ())
            ):
                raise NotImplementedError(
                    f"Joint {joint_name!r} on {handle.path!r} has no native Newton joint."
                )
            body_id = next(
                (link.body_id for link in binding.links if link.name == child), -1
            )
            if body_id < 0:
                raise NotImplementedError(
                    f"Link {child!r} on {handle.path!r} has no native Newton body."
                )
            ids.append(body_id)

        device = force.device
        indices = wp.array(ids, dtype=wp.int32, device=str(device))
        position = wp.vec3(*offset.pos)
        x, y, z, w = offset.quat
        sensor_from_body = wp.normalize(wp.quat(-x, -y, -z, w))
        outputs = (
            wp.from_torch(force, dtype=wp.vec3),
            wp.from_torch(torque, dtype=wp.vec3),
            wp.from_torch(valid),
        )

        def read() -> None:
            context = self._completed
            producer = self._stream
            if context is None or producer is None:
                raise RuntimeError("Joint wrench requires a completed Newton step.")
            state = context.current_state
            if state.body_parent_f is None:
                raise RuntimeError(
                    "Newton did not allocate its parent reaction report."
                )
            consumer = torch.cuda.current_stream(device)
            if consumer != producer:
                consumer.wait_stream(producer)
            wp.launch(
                _sample_joint_reaction,
                dim=len(ids),
                inputs=[
                    indices,
                    state.body_parent_f,
                    state.body_q,
                    context.model.body_com,
                    sensor_from_body,
                    position,
                ],
                outputs=outputs,
                stream=current_warp_stream(device),
            )
            if consumer != producer:
                producer.wait_stream(consumer)

        return read
