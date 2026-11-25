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
"""Sample declared physical evidence on the Gym physics clock."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from weakref import proxy
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.geometry._cable_winding import _capsule_pad_gap, _rope_geometry
from embodichain.lab.evaluation._episode_geometry import _connector_frame
from embodichain.lab.evaluation._episode_reduction import _reduce_measurements
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationSample,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.task_program.integrations._configured_measurements import (
        _EpisodeMeasurementPlan,
    )

__all__: list[str] = []


class _EpisodeMeasurements:
    """Own a complete measured episode and publish its named scalar reductions."""

    def __init__(
        self,
        env: EmbodiedEnv,
        plan: _EpisodeMeasurementPlan,
        anchor: np.ndarray | None,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("Episode cable measurements currently require one arena.")
        self.plan = plan
        self.sensor = env.sim.get_sensor(plan.contact_sensor)
        self.cable = env.sim.get_cable_object(plan.cable_uid)
        if self.sensor is None or self.cable is None:
            raise ValueError(
                "Episode measurements require their configured sensor and cable."
            )
        self.substeps = env.cfg.sim_steps_per_control
        if self.substeps % plan.samples_per_control:
            raise ValueError(
                "The contact measurement cadence must divide the physics substeps."
            )
        self.hz = 1 / env.step_dt
        self.lengths = np.linalg.norm(np.diff(self.cable.cfg.points, axis=0), axis=1)
        self.anchor = (
            self.cable.data.segment_state_w[0, 0, :3].cpu().numpy().copy()
            if anchor is None
            else anchor
        )
        self.object_uids: dict[int, tuple[tuple[str, ...], str | None]] = {}
        self.rows: list[dict[str, Any]] = []
        self.values: dict[str, float] = {}
        self.active = True
        self._probe = (
            _WindingProbe(env, self)
            if plan.kind == "cable_winding"
            else _ConnectorProbe(env, self)
        )
        self.begin_control_step()
        self._capture(dropped=0)
        self.sensor.add_sampling_observer(plan.identity, self)

    def close(self) -> None:
        self.active = False
        self.sensor.remove_sampling_observer(self.plan.identity, expected=self)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or 0 in env_ids:
            self.active = False

    def begin_control_step(self) -> None:
        self.substep = 0
        self._probe.begin()

    def before_sample(self) -> None:
        return None

    def after_sample(self, dt: float, token: object) -> None:
        if not self.active or len(self.rows) > self.plan.steps:
            return
        self.substep += 1
        if self.substep % (self.substeps // self.plan.samples_per_control) == 0:
            self._probe.sample()
        if self.substep == self.substeps:
            self._capture(dropped=self.sensor.evidence_dropped_contacts)

    def _capture(self, *, dropped: int) -> None:
        row = self._probe.capture(len(self.rows), dropped)
        if not row["finite"]:
            raise RuntimeError(
                f"Nonfinite physical measurements at control frame {len(self.rows)}."
            )
        self.rows.append(row)
        if len(self.rows) == self.plan.steps + 1:
            self.values = _reduce_measurements(self.rows, self.plan.reductions)

    def actors(self, ids) -> tuple[tuple[tuple[str, ...], str | None], ...]:
        actors = []
        for actor_id in ids:
            if actor_id < 0:
                continue
            if actor_id not in self.object_uids:
                info = self.sensor.get_actor_info(actor_id)
                self.object_uids[actor_id] = (
                    tuple(info.path.strip("/").split("/")),
                    info.link_name,
                )
            actors.append(self.object_uids[actor_id])
        return tuple(actors)


class _WindingProbe:
    def __init__(self, env: EmbodiedEnv, owner: _EpisodeMeasurements) -> None:
        self._owner = proxy(owner)
        self._robot = env.robot
        self._cfg = owner.plan.parameters
        self._arm_ids = self._robot.get_joint_ids(owner.plan.arm)
        self._finger_ids = sorted(set(range(self._robot.dof)) - set(self._arm_ids))
        if not self._arm_ids or len(self._finger_ids) != 2:
            raise ValueError(
                "Cable winding observations require an arm and two gripper joints."
            )
        lengths = owner.lengths
        arc = lengths.sum() - self._cfg["grasp_inset_m"]
        if not 0 < arc < lengths.sum():
            raise ValueError("The winding probe's grasp inset is outside the cable.")
        self._material_index = index = int(np.searchsorted(np.cumsum(lengths), arc))
        self._material_offset = arc - lengths[:index].sum() - lengths[index] / 2
        limits = ElementTree.parse(Path(self._robot.cfg.fpath))
        joints = [
            limits.find(f".//joint[@name='{name}']/limit")
            for name in self._robot.joint_names
        ]
        self._lo = np.array([float(joint.get("lower")) for joint in joints])
        self._hi = np.array([float(joint.get("upper")) for joint in joints])
        self._baseline = None

    def begin(self) -> None:
        self._bilateral = False
        self._robot_post_penetration = 0.0

    def sample(self) -> None:
        owner, cfg = self._owner, self._cfg
        data = owner.sensor.get_data()
        valid = data["is_valid"]
        points: list[list[list[float]]] = [[], []]
        for ids, separation, position in zip(
            data["user_ids"][valid].cpu().tolist(),
            data["distance"][valid].cpu().tolist(),
            data["position"][valid].cpu().tolist(),
        ):
            actors = owner.actors(ids)
            if any(cfg["post_uid"] in path for path, _ in actors) and any(
                self._robot.uid in path for path, _ in actors
            ):
                self._robot_post_penetration = max(
                    self._robot_post_penetration, -separation
                )
            if separation <= cfg["contact_distance_m"] and (
                any(actor < 0 for actor in ids)
                or any(owner.plan.cable_uid in path for path, _ in actors)
            ):
                for i, finger in enumerate(cfg["fingers"]):
                    if any(
                        self._robot.uid in path and link == finger
                        for path, link in actors
                    ):
                        points[i].append(position)
        if self._bilateral or not all(points):
            return
        tail = cfg["tail_segments"]
        state = owner.cable.data.segment_state_w[0, -tail:].cpu().numpy()
        lengths = owner.lengths[-tail:]
        axes = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :, 2]
        starts = state[:, :3] - axes * lengths[:, None] / 2
        delta = axes * lengths[:, None]
        for finger, contacts in zip(cfg["fingers"], points, strict=True):
            point = np.asarray(contacts)
            fraction = np.clip(
                ((point[:, None] - starts) * delta).sum(axis=-1)
                / np.sum(delta * delta, axis=-1),
                0.0,
                1.0,
            )
            closest = starts + fraction[..., None] * delta
            point_gap = (
                np.linalg.norm(point[:, None] - closest, axis=-1).min()
                - cfg["cable_radius_m"]
            )
            pose = self._robot.get_link_pose(finger, to_matrix=True)[0].cpu().numpy()
            if (
                point_gap > cfg["contact_point_gap_m"]
                or _capsule_pad_gap(state, lengths, pose, cfg) > cfg["pad_gap_m"]
            ):
                return
        self._bilateral = True

    def capture(self, frame: int, dropped: int) -> dict[str, Any]:
        owner, cfg = self._owner, self._cfg
        state = owner.cable.data.segment_state_w[0].cpu().numpy()
        qpos = self._robot.get_qpos()[0].cpu().numpy()
        tcp = (
            self._robot.get_link_pose(owner.plan.tool_link, to_matrix=True)[0]
            .cpu()
            .numpy()
        )
        index = self._material_index
        axis = Rotation.from_quat(state[index, 3:7]).as_matrix()[:, 2]
        material = state[index, :3] + self._material_offset * axis
        local = tcp[:3, :3].T @ (material - tcp[:3, 3])
        if self._baseline is None and frame >= cfg["grasp_start"] and self._bilateral:
            self._baseline = local.copy()
        slip = (
            0.0
            if self._baseline is None
            else float(np.linalg.norm(local - self._baseline))
        )
        geometry = _rope_geometry(state, owner.lengths, cfg)
        nodes = geometry.pop("nodes")
        violation = np.maximum(self._lo - qpos, qpos - self._hi)
        return {
            **geometry,
            "frame": frame,
            "free_end_z_m": float(nodes[-1, 2]),
            "tail_tcp_distance_m": float(np.linalg.norm(nodes[-1] - tcp[:3, 3])),
            "grasp_confirmed": self._baseline is not None,
            "material_slip_m": slip,
            "bilateral_rope_contact": self._bilateral,
            "anchor_error_m": float(np.linalg.norm(state[0, :3] - owner.anchor)),
            "arm_joint_limit_violation_rad": float(
                max(0.0, violation[self._arm_ids].max())
            ),
            "finger_joint_limit_violation_m": float(
                max(0.0, violation[self._finger_ids].max())
            ),
            "robot_post_penetration_m": self._robot_post_penetration,
            "dropped_contacts": dropped,
            "finite": bool(
                np.isfinite(state).all()
                and np.isfinite(qpos).all()
                and np.isfinite(tcp).all()
            ),
        }


class _ConnectorProbe:
    def __init__(self, env: EmbodiedEnv, owner: _EpisodeMeasurements) -> None:
        self._owner = proxy(owner)
        self._robot = env.robot
        self._cfg = owner.plan.parameters
        self._connectable = env.sim.get_articulation(self._cfg["connectable"])
        if (
            self._connectable is None
            or env.sim.get_articulation(self._cfg["receiver_uid"]) is None
        ):
            raise ValueError("Connector measurements require the bound articulations.")
        self._latch_index = self._connectable.joint_names.index(
            self._cfg["latch_joint"]
        )
        arm = self._robot.get_joint_ids(owner.plan.arm)
        if len(arm) != len(self._cfg["home_qpos"]):
            raise ValueError("The connector home pose must match the configured arm.")
        fingers = sorted(set(range(self._robot.dof)) - set(arm))
        if len(fingers) != 2:
            raise ValueError("Connector measurements require two gripper joints.")
        self._joint_order = [*arm, *fingers]
        self._insertion_axis = np.asarray(self._cfg["insertion_rotation"])[:, 2]

    def begin(self) -> None:
        self._finger = self._receiver = False
        self._penetration = self._latch_limit = 0.0
        self._press_samples: list[dict[str, float]] = []

    def _angle(self) -> float:
        return float(self._connectable.get_qpos()[0, self._latch_index])

    def sample(self) -> None:
        owner, cfg = self._owner, self._cfg
        data = owner.sensor.get_data()
        valid = data["is_valid"]
        touched = False
        for ids, separation in zip(
            data["user_ids"][valid].cpu().tolist(),
            data["distance"][valid].cpu().tolist(),
        ):
            self._penetration = max(self._penetration, -separation)
            if separation > 0:
                continue
            pair = owner.actors(ids)
            if any(
                cfg["connectable"] in path and link == cfg["latch_link"]
                for path, link in pair
            ):
                touched |= any(
                    self._robot.uid in path and link in cfg["fingers"]
                    for path, link in pair
                )
                self._receiver |= any(
                    cfg["receiver_uid"] in path and link == cfg["receiver_link"]
                    for path, link in pair
                )
        self._finger |= touched
        angle = self._angle()
        lo, hi = cfg["latch_range_rad"]
        self._latch_limit = max(self._latch_limit, angle - hi, lo - angle)
        if touched:
            pose = (
                self._connectable.get_link_pose(
                    cfg["connectable_link"], to_matrix=True
                )[0]
                .cpu()
                .numpy()
            )
            self._press_samples.append(
                {
                    "latch_angle_deg": float(np.rad2deg(angle)),
                    "depth_m": float(
                        self._insertion_axis @ (pose[:3, 3] - cfg["socket_mouth_m"])
                    ),
                }
            )

    def capture(self, frame: int, dropped: int) -> dict[str, Any]:
        owner, cfg = self._owner, self._cfg
        return _connector_frame(
            {**cfg, "control_hz": owner.hz},
            frame=frame,
            plug_pose=self._connectable.get_link_pose(
                cfg["connectable_link"], to_matrix=True
            )[0]
            .cpu()
            .numpy(),
            tcp_pose=self._robot.get_link_pose(owner.plan.tool_link, to_matrix=True)[0]
            .cpu()
            .numpy(),
            robot_qpos=self._robot.get_qpos()[0].cpu().numpy()[self._joint_order],
            latch_angle_rad=self._angle(),
            cable_states=owner.cable.data.segment_state_w[0].cpu().numpy(),
            cable_lengths=owner.lengths,
            anchor_baseline=owner.anchor,
            finger_latch_contact=self._finger,
            latch_socket_contact=self._receiver,
            latch_limit_violation_rad=self._latch_limit,
            penetration_m=self._penetration,
            dropped_contacts=dropped,
            press_samples=self._press_samples,
        )


def _measurement_sample(
    env: EmbodiedEnv, identity: str, name: str
) -> ObservationSample:
    plan = env._episode_measurement_plan
    live = env._episode_measurements
    if plan is None or identity != plan.identity or name not in plan.reductions:
        raise ValueError(f"Unknown configured episode measurement {name!r}.")
    valid = live is not None and live.active and name in live.values
    values = torch.full(
        (env.num_envs, 1),
        live.values[name] if valid else 0.0,
        dtype=torch.float64,
        device=env.device,
    )
    return ObservationSample(
        ObservationDescriptor(
            identity,
            name,
            "episode",
            "scalar",
            (plan.reductions[name]["unit"],),
            (1,),
            values.dtype,
        ),
        values,
        torch.full((env.num_envs,), valid, dtype=torch.bool, device=env.device),
        (None if valid else "Episode measurement is unavailable.",) * env.num_envs,
    )
