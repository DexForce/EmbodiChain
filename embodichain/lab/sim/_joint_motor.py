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
"""Manager-owned explicit joint motor using the Newton control lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from dexsim.engine.newton_physics.runtime_control import (
    GraphRuntimeControl,
    RuntimeControlContext,
)

__all__: list[str] = []

wp.set_module_options({"enable_fp_fusion": False, "enable_backward": False})


@wp.kernel(enable_backward=False)
def _dc_motor(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_indices: wp.array2d(dtype=wp.int32),
    qd_indices: wp.array2d(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.float32),
    speeds: wp.array(dtype=wp.float32),
    model_force: wp.array(dtype=wp.float32),
    control_force: wp.array(dtype=wp.float32),
    effort: wp.array2d(dtype=wp.float32),
    previous_velocity: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    dof = qd_indices[row, column]
    velocity = joint_qd[dof]
    desired = (
        kp[column] * (targets[row, column] - joint_q[q_indices[row, column]])
        - kd[column] * velocity
    )
    speed = wp.clamp(velocity, -2.0 * speeds[column], 2.0 * speeds[column])
    maximum = wp.min(limits[column] * (1.0 - speed / speeds[column]), limits[column])
    minimum = wp.max(limits[column] * (-1.0 - speed / speeds[column]), -limits[column])
    value = wp.clamp(desired, minimum, maximum)
    if wp.isnan(desired) or wp.isnan(velocity):
        value = desired + velocity
    model_force[dof] = value
    control_force[dof] = value
    effort[row, column] = value
    previous_velocity[row, column] = velocity


class _BatchedJointDCMotorControl(GraphRuntimeControl):
    """Hold DC motor effort over each existing outer physics step."""

    def __init__(self, robot: dict[str, Any], target_paths: tuple[str, ...]) -> None:
        self.robot = robot
        self.target_paths = target_paths
        self.num_envs = len(target_paths)
        self.targets = self.effort = self.velocity = None
        self._claims = ()

    def initialize(self, context: RuntimeControlContext) -> None:
        """Resolve scalar joints once in the manager-provided arena order."""
        names = self.robot["joint_names"]
        q, qd, claims = [], [], []
        for path in self.target_paths:
            articulation = context.result.get_articulation(path)._physics_binding
            joints = {joint.name: joint for joint in articulation.public_joints}
            selected = [joints[name] for name in names]
            if any(joint.q_size != 1 or joint.qd_size != 1 for joint in selected):
                raise ValueError("DC motor control requires scalar joints")
            q.append([joint.q_start for joint in selected])
            qd.append([joint.qd_start for joint in selected])
            claims.extend(("newton_joint", int(joint.joint_id)) for joint in selected)
        if len(q) != self.num_envs:
            raise ValueError("Runtime control rows differ from the policy batch")
        self.q_indices = wp.array(q, dtype=wp.int32, device=context.device)
        self.qd_indices = wp.array(qd, dtype=wp.int32, device=context.device)
        self.kp, self.kd, self.limits, self.speeds = (
            wp.array(self.robot[name], dtype=wp.float32, device=context.device)
            for name in ("stiffness", "damping", "effort_limit", "velocity_limit")
        )
        self.targets = wp.array(
            np.tile(self.robot["default_joint_position"], (self.num_envs, 1)).astype(
                np.float32
            ),
            device=context.device,
        )
        self.effort = wp.zeros_like(self.targets)
        self.velocity = wp.zeros_like(self.targets)
        self._claims = tuple(claims)

    def exclusive_resource_claims(self) -> tuple[object, ...]:
        """Reserve the driven joints through the existing control registry."""
        return self._claims

    def __call__(
        self,
        context: RuntimeControlContext,
        substep_index: int,
        substep_count: int,
        substep_dt: float,
    ) -> None:
        """Update once before the first solver substep of an outer step."""
        if substep_index:
            return
        control_force = context.control.joint_f
        model_force = getattr(context.model, "joint_f", None)
        if model_force is None:
            model_force = control_force
        wp.launch(
            _dc_motor,
            dim=self.targets.shape,
            inputs=[
                context.current_state.joint_q,
                context.current_state.joint_qd,
                self.q_indices,
                self.qd_indices,
                self.targets,
                self.kp,
                self.kd,
                self.limits,
                self.speeds,
                model_force,
                control_force,
                self.effort,
                self.velocity,
            ],
            device=context.device,
        )

    def close(self) -> None:
        """Release scene-owned device buffers before teardown or rebuilding."""
        self.q_indices = self.qd_indices = None
        self.kp = self.kd = self.limits = self.speeds = None
        self.targets = self.effort = self.velocity = None
        self._claims = ()
