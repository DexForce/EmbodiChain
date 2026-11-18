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

"""Shared per-episode measurements for native assembly functors."""

from __future__ import annotations
from typing import Any
import torch
from embodichain.utils.math import trans_matrix_to_xyz_quat
from .functors import (
    _assembly_kinematic_chain,
)
from embodichain.compute.geometry import signed_distance_to_polygon
from .preparation import AssemblyScene
from ._config import assembly_cfg
from ._control import EncoderGravityCompensation

__all__ = ["initialize_assembly", "update_assembly"]


class AssemblyState:
    """Own measurements and transition counters without an environment facade."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._last_control_physics_step = env.sim.physics_step_index
        self.cfg = assembly_cfg(env.cfg.extensions)
        self.scene = AssemblyScene(env.cfg)
        self.arm_dof = len(
            env.robot.get_joint_ids(name=self.cfg.arm_part, remove_mimic=True)
        )
        self.frame_dim = (
            3 * self.arm_dof
            + 9
            + (6 if self.cfg.orientation_representation == "rotation6d" else 3)
        )
        self._assembly_revision = 0
        self._assembly_physics_revision = 0
        self._assembly_state_revision = -1
        self._assembly_state_physics_revision = -1
        self._assembly_reset_rows = torch.ones(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        self._assembly_pending_reset_ids = None
        self._assembly_frame_reset_ids = None
        self._assembly_chain = _assembly_kinematic_chain(
            self.env.cfg.robot, self.env.device, self.cfg.arm_part
        )
        limits = (
            self.env.robot.get_qpos_limits(name=self.cfg.arm_part)[0]
            if self.cfg.proprioceptive_joint_limits is None
            else torch.as_tensor(
                self.cfg.proprioceptive_joint_limits,
                dtype=torch.float32,
                device=self.env.device,
            )
        )
        velocity = (
            self.env.robot.get_qvel_limits(name=self.cfg.arm_part)[0]
            if self.cfg.proprioceptive_velocity_scale is None
            else torch.as_tensor(
                self.cfg.proprioceptive_velocity_scale,
                dtype=torch.float32,
                device=self.env.device,
            )
        )
        if (
            limits.shape != (self.arm_dof, 2)
            or velocity.shape != (self.arm_dof,)
            or not torch.isfinite(limits).all()
            or not torch.isfinite(velocity).all()
            or not (limits[:, 0] < limits[:, 1]).all()
            or not (velocity > 0).all()
        ):
            raise ValueError(
                "Assembly proprioception requires finite scales matching the selected arm joints."
            )
        self._assembly_q_center = limits.mean(-1)
        self._assembly_q_half_range = (limits[:, 1] - limits[:, 0]) * 0.5
        self._assembly_velocity_scale = velocity
        self._assembly_success_hold = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        self._assembly_milestones = torch.zeros(
            (self.env.num_envs, 4), dtype=torch.bool, device=self.env.device
        )
        self._assembly_new_milestones = torch.zeros(
            (self.env.num_envs, 4), dtype=torch.bool, device=self.env.device
        )
        self._assembly_angle = torch.zeros(self.env.num_envs, device=self.env.device)
        self._assembly_previous_yaw = torch.zeros(
            self.env.num_envs, device=self.env.device
        )
        self._assembly_capture_angle = torch.full(
            (self.env.num_envs,), float("nan"), device=self.env.device
        )
        self._assembly_capture_depth = torch.full(
            (self.env.num_envs,), float("nan"), device=self.env.device
        )
        self._assembly_labels = torch.zeros(
            (
                self.env.num_envs,
                self.cfg.contact_points_per_ring
                * (2 if self.cfg.kind == "gear" else 1),
            ),
            dtype=torch.long,
            device=self.env.device,
        )
        self._assembly_metrics: dict[str, torch.Tensor] = {}
        self._assembly_edge_points = torch.as_tensor(
            self.scene._edge_points, dtype=torch.float32, device=self.env.device
        )
        self._assembly_bottom_vertices = torch.as_tensor(
            self.scene._bottom_vertices, dtype=torch.float32, device=self.env.device
        )
        self.controller = EncoderGravityCompensation(env, self.cfg.arm_part)
        self._assembly_robot_base = self.controller.robot_base
        self.scene.assembly_fidelity["gravity_compensation"] = (
            "URDF COM potential gradient from encoders, held at control frequency"
        )
        grasp_sensor = self.env.get_sensor(self.cfg.grasp_sensor)
        self._assembly_pad_actor_ids = grasp_sensor.get_actor_ids(
            self.env.robot.uid, self.scene._assembly_grasp_links
        )
        self._assembly_part_actor_ids = grasp_sensor.get_actor_ids(
            self.scene._moving_uid
        )[:, 0]
        self._grasp_history = grasp_sensor.create_history(
            "bilateral_grasp",
            self._assembly_pad_actor_ids,
            counterpart_ids=self._assembly_part_actor_ids[:, None],
            force_threshold=self.cfg.grasp_force_threshold,
        )
        # Both contact consumers belong to the physics-substep lifecycle.
        # Reset observations read cleared buffers until a solved step exists.
        mating_sensor = self.env.get_sensor(self.cfg.mating_sensor)
        self._mating_history = mating_sensor.create_history(
            "mating_contacts",
            mating_sensor.get_actor_ids(self.scene._moving_uid),
        )
        self.scene.assembly_fidelity["grasp_guard"] = (
            f"bilateral part-pad contact; max air time {self.cfg.grasp_max_air_time:g} s, "
            f"force threshold {self.cfg.grasp_force_threshold:g} N"
        )
        arm_ids = self.env.robot.get_joint_ids(
            name=self.cfg.arm_part, remove_mimic=True
        )
        stiffness = torch.full(
            (self.env.num_envs, self.arm_dof),
            self.cfg.joint_stiffness,
            dtype=torch.float32,
            device=self.env.device,
        )
        damping = torch.full(
            (self.env.num_envs, self.arm_dof),
            self.cfg.joint_damping,
            dtype=torch.float32,
            device=self.env.device,
        )
        if (
            self.env.sim.is_newton_backend
            and self.env.sim.physics.solver_type == "mujoco_warp"
        ):
            from embodichain.lab.sim.physics._joint_drive_view import (
                _JointDriveGainView,
            )

            names = tuple(self.env.robot.joint_names[index] for index in arm_ids)
            view = _JointDriveGainView(self.env.sim, names)
            try:
                view.apply(stiffness, damping)
            finally:
                view.close()
        else:
            self.env.robot.set_joint_drive(
                stiffness=stiffness,
                damping=damping,
                joint_ids=arm_ids,
            )

    def advance(self) -> None:
        """Advance at most once for a control transition with new physics."""
        step = self.env.sim.physics_step_index
        if step != self._last_control_physics_step:
            self._assembly_revision += 1
            self._assembly_physics_revision += 1
            self._last_control_physics_step = step
        self.refresh_assembly_state()

    def _contact_grasp(self) -> torch.Tensor:
        sensor = self.env.get_sensor(self.cfg.grasp_sensor)
        if sensor.dropped_contacts:
            raise RuntimeError(
                "Grasp contact buffer overflow; increase the configured capacity."
            )
        return self._grasp_history.found.all(dim=1) & (
            self._grasp_history.current_air_time <= self.cfg.grasp_max_air_time
        ).all(dim=1)

    def refresh_assembly_state(self) -> None:
        """Compute supervision and milestone transitions once per physical step."""
        if self._assembly_state_revision == self._assembly_revision:
            return
        self._assembly_state_revision = self._assembly_revision
        physical_change = (
            self._assembly_state_physics_revision != self._assembly_physics_revision
        )
        pending = getattr(self, "_assembly_pending_reset_ids", None)
        partial = (
            not physical_change
            and pending is not None
            and 0 < len(pending) < self.env.num_envs
            and hasattr(self, "_assembly_metrics")
            and hasattr(self, "_assembly_labels")
        )
        ids = pending if partial else None
        self._assembly_pending_reset_ids = None

        def select(value: torch.Tensor) -> torch.Tensor:
            return value if ids is None else value[ids]

        count = self.env.num_envs if ids is None else len(ids)
        update_rows = (
            torch.ones(count, dtype=torch.bool, device=self.env.device)
            if physical_change
            else select(self._assembly_reset_rows).clone()
        )
        self._assembly_state_physics_revision = self._assembly_physics_revision
        part = select(
            self.env.sim.get_rigid_object(self.scene._moving_uid).get_local_pose(
                to_matrix=True
            )
        )
        fixture = select(
            self.env.sim.get_rigid_object(self.scene._fixture_uid).get_local_pose(
                to_matrix=True
            )
        )
        # Rigid inverse avoids a general LU solve and its CUDA error-check
        # synchronization on every control observation.
        inverse_rotation = fixture[:, :3, :3].transpose(1, 2)
        relative = torch.zeros_like(part)
        relative[:, :3, :3] = inverse_rotation @ part[:, :3, :3]
        relative[:, :3, 3] = (
            inverse_rotation @ (part[:, :3, 3] - fixture[:, :3, 3]).unsqueeze(-1)
        ).squeeze(-1)
        relative[:, 3, 3] = 1
        points = self._assembly_edge_points.expand(count, -1, -1)
        points = points @ relative[:, :3, :3].transpose(1, 2) + relative[:, None, :3, 3]
        radial = torch.linalg.vector_norm(relative[:, :2, 3], dim=-1)
        tilt = torch.acos(relative[:, 2, 2].clamp(-1, 1))
        sensor = self.env.get_sensor(self.cfg.mating_sensor)
        if sensor.dropped_contacts:
            raise RuntimeError(
                "Assembly contact buffer overflow; increase the configured capacity."
            )
        contacts = sensor.get_data()
        penetration = torch.where(
            select(contacts["is_valid"]),
            (-select(contacts["distance"])).clamp_min(0),
            0,
        ).amax(dim=1)
        grasped = select(self._contact_grasp())
        asset_config = self.scene.assembly_asset.config
        clearance = float(asset_config.get("clearance", 0.0005))
        peg_scale = 1.0
        if self.scene.assembly_kind == "peg":
            peg_scale = float(asset_config.get("scale", 1.0))
            clearance *= peg_scale
        penetration_ok = penetration <= max(
            self.cfg.penetration_tolerance,
            clearance * self.cfg.penetration_clearance_fraction,
        )
        elapsed = select(self.env._elapsed_steps)
        warm = elapsed >= self.cfg.grasp_warmup_steps
        yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
        previous_yaw = select(self._assembly_previous_yaw)
        delta = torch.atan2(
            torch.sin(yaw - previous_yaw),
            torch.cos(yaw - previous_yaw),
        )
        delta[select(self._assembly_reset_rows)] = 0
        angle = select(self._assembly_angle) + delta * update_rows
        next_yaw = torch.where(update_rows, yaw, previous_yaw)
        if ids is None:
            self._assembly_angle.copy_(angle)
            self._assembly_previous_yaw.copy_(next_yaw)
        else:
            self._assembly_angle[ids] = angle
            self._assembly_previous_yaw[ids] = next_yaw
        self._assembly_new_milestones.zero_()
        depth = self.scene._fixture_zmax - points[:, :, 2].amax(dim=1)
        capture = torch.zeros_like(grasped)
        success = torch.zeros_like(grasped)
        if self.scene.assembly_kind == "peg":
            if not hasattr(self, "_hole_polygon"):
                outline = asset_config.get("mating_outline")
                if outline is None:
                    raise ValueError(
                        "Peg task manifests require the generated mating_outline for dependency-free labeling."
                    )
                # Manifests store the generator-space outline. Match the
                # physical collision vertices without changing saved manifests.
                self._hole_polygon = peg_scale * torch.tensor(
                    outline, dtype=torch.float32, device=self.env.device
                )
            # Supervision uses configured samples; success checks every actual
            # convex bottom vertex, so corners between samples cannot escape.
            bottom = self._assembly_bottom_vertices
            bottom_points = (
                bottom[None] @ relative[:, :3, :3].transpose(1, 2)
                + relative[:, None, :3, 3]
            )
            # Match signedDistance/labelEdge in the authors' public
            # static/js/assembly_task.js: segment distance, even-odd interior,
            # and a 1 mm band around both the opening rim and top plane.
            samples = torch.cat([points[:, :, :2], bottom_points[:, :, :2]], dim=1)
            signed_distance = signed_distance_to_polygon(samples, self._hole_polygon)
            edge_distance = signed_distance[:, : points.shape[1]]
            all_inside = (signed_distance[:, points.shape[1] :] <= 0).all(dim=-1)
            depth = self.scene._fixture_zmax - bottom_points[:, :, 2].amax(dim=1)
            band = self.cfg.contact_band
            height = points[:, :, 2] - self.scene._fixture_zmax
            near_surface = height.abs() <= band
            near_edge = edge_distance.abs() <= band
            inside = ((height <= band) & (edge_distance < -band)) | (
                (height <= 0) & near_edge
            )
            contact = near_surface & ((edge_distance > band) | near_edge)
            labels = torch.where(contact, 1, torch.where(inside, 2, 0)).long()
            capture = all_inside & (depth > 0)
            plate_thickness = peg_scale * float(asset_config["plate_thickness"])
            target_bottom_z = self.scene._fixture_zmax - plate_thickness
            # The public task accepts every bottom point within 1 mm of the
            # seated plane. Grasp, penetration and the three-step hold below
            # remain additional project guards, separate from that geometry.
            success = all_inside & (
                (bottom_points[:, :, 2] - target_bottom_z).abs() <= band
            ).all(dim=-1)
        elif self.scene.assembly_kind == "gear":
            cfg = self.scene.assembly_asset.config
            bore = float(cfg["shaft_radius"]) + float(cfg["bore_clearance"])
            axis = torch.tensor(
                [0.0, 0.0, self.scene._fixture_zmax], device=self.env.device
            ).expand(count, -1)
            local_tip = (axis - relative[:, :3, 3]).unsqueeze(1) @ relative[:, :3, :3]
            aligned = torch.linalg.vector_norm(local_tip[:, 0, :2], dim=-1) < float(
                cfg["bore_clearance"]
            )
            capture = aligned & (depth > 0)
            bottom = points[:, :, 2].mean(dim=1)
            success = (
                capture
                & (bottom.abs() < self.cfg.contact_band)
                & (tilt < self.cfg.gear_tilt_tolerance)
            )
            rim = (
                torch.arange(self.cfg.contact_points_per_ring, device=self.env.device)
                * 2
                * torch.pi
                / self.cfg.contact_points_per_ring
            )
            shaft_pts = torch.stack(
                [
                    torch.cos(rim) * float(cfg["shaft_radius"]),
                    torch.sin(rim) * float(cfg["shaft_radius"]),
                    torch.full_like(rim, self.scene._fixture_zmax),
                ],
                dim=-1,
            )
            in_part = (shaft_pts[None] - relative[:, None, :3, 3]) @ relative[:, :3, :3]
            in_bore = torch.linalg.vector_norm(in_part[:, :, :2], dim=-1) < bore
            rim_labels = torch.where(
                in_part[:, :, 2] < self.scene._moving_zmin - self.cfg.contact_band,
                0,
                torch.where(in_bore, 2, 1),
            )
            contact = select(contacts["is_valid"]).any(dim=1)
            tooth_labels = torch.where(
                (depth > 0)[:, None],
                torch.where(
                    contact[:, None],
                    torch.ones_like(points[:, :, 0], dtype=torch.long),
                    torch.full_like(points[:, :, 0], 2, dtype=torch.long),
                ),
                0,
            )
            labels = torch.cat([rim_labels, tooth_labels], dim=1).long()
        else:
            cfg = self.scene.assembly_asset.config
            pitch = float(cfg["pitch"]) * float(cfg.get("scale", 1.0))
            lead = pitch * int(cfg.get("n_starts", 1))
            near_axis = radial < self.cfg.nut_radial_tolerance
            advance = depth.clamp_min(0)
            capture = (
                near_axis
                & (advance > self.cfg.nut_capture_pitch_fraction * pitch)
                & (tilt < self.cfg.nut_tilt_tolerance)
            )
            capture_angle = select(self._assembly_capture_angle)
            capture_depth = select(self._assembly_capture_depth)
            first_capture = capture & ~torch.isfinite(capture_angle) & update_rows
            capture_angle[first_capture] = angle[first_capture]
            capture_depth[first_capture] = advance[first_capture]
            if ids is not None:
                self._assembly_capture_angle[ids] = capture_angle
                self._assembly_capture_depth[ids] = capture_depth
            turned = (angle - capture_angle).abs()
            axial_progress = advance - capture_depth
            expected_progress = lead * turned / (2 * torch.pi)
            # Require new axial advance *after* first capture. Rotating at a
            # fixed captured depth is not threading. The 0.1-pitch band is an
            # explicit nominal tolerance, not an author calibration.
            success = (
                capture
                & (turned >= self.cfg.nut_min_turn)
                & (axial_progress >= pitch * self.cfg.nut_min_advance_fraction)
                & (
                    (axial_progress - expected_progress).abs()
                    <= pitch * self.cfg.nut_advance_tolerance_fraction
                )
            )
            free = (
                points[:, :, 2]
                > self.scene._fixture_zmax + self.cfg.nut_capture_pitch_fraction * pitch
            )
            around = near_axis[:, None] & (points[:, :, 2] < self.scene._fixture_zmax)
            labels = torch.where(
                free, 0, torch.where(capture[:, None], 3, torch.where(around, 2, 1))
            ).long()
        if ids is None:
            self._assembly_labels = labels
        else:
            # Older observations can still hold these tensors after reset.
            # Preserve their terminal values while replacing selected rows.
            next_labels = self._assembly_labels.clone()
            next_labels[ids] = labels
            self._assembly_labels = next_labels
        safe = warm & grasped & penetration_ok
        hold = select(self._assembly_success_hold)
        next_hold = torch.where(success & safe, hold + 1, 0)
        hold = torch.where(update_rows, next_hold, hold)
        if ids is None:
            self._assembly_success_hold = hold
        else:
            self._assembly_success_hold[ids] = hold
        stable_success = hold >= self.cfg.success_hold_steps
        milestones = (
            torch.stack(
                [
                    capture,
                    capture & (depth > self.cfg.milestone_depth),
                    stable_success,
                    torch.zeros_like(success),
                ],
                dim=-1,
            )
            & safe[:, None]
        )
        previous_milestones = select(self._assembly_milestones)
        new_milestones = milestones & ~previous_milestones & update_rows[:, None]
        next_milestones = previous_milestones | (milestones & update_rows[:, None])
        if ids is None:
            self._assembly_new_milestones.copy_(new_milestones)
            self._assembly_milestones.copy_(next_milestones)
        else:
            self._assembly_new_milestones[ids] = new_milestones
            self._assembly_milestones[ids] = next_milestones
        failure = (
            torch.linalg.vector_norm(relative[:, :3, 3], dim=-1)
            > self.cfg.failure_distance
        ) | (~grasped & (elapsed > self.cfg.grasp_failure_after_steps))
        metrics = {
            "radial_error": radial,
            "tilt": tilt,
            "depth": depth,
            "penetration": penetration,
            "grasped": grasped,
            "captured": capture,
            "success": stable_success,
            "fail": failure,
            "relative_pose": trans_matrix_to_xyz_quat(relative),
        }
        if ids is None:
            self._assembly_metrics = metrics
        else:
            next_metrics = {}
            for key, value in metrics.items():
                merged = self._assembly_metrics[key].clone()
                merged[ids] = value
                next_metrics[key] = merged
            self._assembly_metrics = next_metrics
        if ids is None:
            self._assembly_reset_rows.zero_()
        else:
            self._assembly_reset_rows[ids] = False


def initialize_assembly(env: Any, env_ids: Any) -> None:
    """Bind state after ordinary entities and sensors are prepared.

    Args:
        env: Common environment owning the prepared scene and managers.
        env_ids: Startup selection supplied by EventManager; state covers all rows.
    """
    env.assembly_state = AssemblyState(env)


def update_assembly(env: Any, env_ids: Any) -> None:
    """Refresh shared measurements once after each configured control interval.

    Args:
        env: Host containing the startup-bound assembly state.
        env_ids: Event selection; solved physics advances the shared batch.
    """
    env.assembly_state.advance()
