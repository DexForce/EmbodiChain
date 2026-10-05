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

"""Direct UR5/PGI waypoint planning and native physical replay."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.tools.grasp_assemble._geometry import TCP_Z
from scripts.tools.grasp_assemble._config import execution_settings
from scripts.tools.grasp_assemble._layout import motion_metrics, motion_rejection

__all__ = ["RobotSession"]


class RobotSession:
    """Plan smooth waypoint trajectories separately from physical execution."""

    def __init__(self, config: dict, assembly: dict) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            LightCfg,
            RenderCfg,
            RigidObjectCfg,
            physics_cfg_for_backend,
        )
        from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg
        from scripts.tools.grasp_assemble._scene import (
            create_robot_cfg,
            create_assemble_physics,
        )

        self.config, self.assembly = config, assembly
        self._grasp_constraint = None
        self._constraint_events = []
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                physics_dt=0.01,
                num_envs=1,
                width=960,
                height=720,
                physics_cfg=physics_cfg_for_backend("default"),
                render_cfg=RenderCfg(renderer="auto"),
            )
        )
        try:
            self.sim.add_light(
                cfg=LightCfg(
                    uid="assembly_light",
                    light_type="direction",
                    direction=(-0.4, 0.6, -1),
                    intensity=3.0,
                )
            )
            self.robot = self.sim.add_robot(cfg=create_robot_cfg(tcp_z=TCP_Z))
            self.base = self.sim.add_rigid_object(
                cfg=RigidObjectCfg(
                    uid="base",
                    shape=MeshCfg(
                        fpath=assembly["assets"]["base"]["path"],
                        collision=MeshCollisionCfg(approximation="triangle_mesh"),
                    ),
                    body_type="static",
                    init_local_pose=config["T_world_base"],
                )
            )
            self.assemble = self.sim.add_rigid_object(
                cfg=RigidObjectCfg(
                    uid="assemble",
                    shape=MeshCfg(
                        fpath=assembly["assets"]["assemble"]["path"],
                        collision=MeshCollisionCfg(
                            approximation="convex_decomposition",
                            max_hulls=64,
                            acd_method="visacd",
                        ),
                    ),
                    attrs=create_assemble_physics(),
                    init_local_pose=config["T_world_assemble_initial"],
                )
            )
            self.sim.prepare()
            if not np.allclose(
                self.robot.get_local_pose(to_matrix=True)[0].cpu().numpy(),
                np.eye(4),
                atol=1e-6,
            ):
                raise ValueError(
                    "This harness requires the UR5 root at the world origin"
                )
            self.initial_qpos = self.robot.get_qpos().clone()
            self.seed_qpos = self.initial_qpos.clone()
            self.trajectory: dict | None = None
        except BaseException:
            self.close()
            raise

    def set_layout(self, config: dict) -> None:
        """Set both object poses between candidate plans, before physical replay.

        Args:
            config: Candidate-specific world placement and planning settings.
        """
        import torch

        if self._grasp_constraint is not None:
            raise RuntimeError("Cannot change layout while holding the assemble object")
        for body, key in (
            (self.base, "T_world_base"),
            (self.assemble, "T_world_assemble_initial"),
        ):
            body.set_local_pose(
                torch.tensor(config[key], dtype=torch.float32, device=self.sim.device)[
                    None
                ]
            )
            body.clear_dynamics()
        self.config = config
        self.trajectory = None

    def plan(self, candidate: dict, geometry: Any, directory: Path) -> dict:
        """Plan and collision-check a smooth direct waypoint trajectory.

        Args:
            candidate: Validated object-local grasp and named Cartesian waypoints.
            geometry: World geometry for this layout.
            directory: Destination for trajectory arrays and diagnostics.

        Returns:
            Planning acceptance, timing/derivative metrics, and waypoint errors.
        """
        import torch
        from scripts.tools.grasp_assemble._config import motion_settings
        from scripts.tools.grasp_assemble._geometry import interpolate_poses
        from scripts.tools.grasp_assemble._path import select_ik_paths

        started = time.perf_counter()
        self.trajectory = None
        self.config["motion"] = motion_settings(self.config.get("motion"))
        settings = self.config["motion"]
        arm_ids = self.robot.get_joint_ids(name="arm")
        seed = self.seed_qpos[0, arm_ids].cpu().numpy()
        limits = self.robot.get_qpos_limits(name="arm")[0].cpu().numpy()
        waypoints = candidate["waypoints"]
        grip = np.asarray(candidate["T_assemble_tcp"])
        pick = np.asarray(waypoints["grasp"]["T_world_tcp"])
        pre = pick.copy()
        pre[:3, 3] -= pick[:3, 2] * settings["pre_grasp_distance"]
        place = np.asarray(waypoints["place"]["T_world_tcp"])
        retract = np.asarray(candidate["T_world_tcp_retract"])
        routes = {"approach": (pre, pick, False)}
        carry_routes = [
            ("lift", "grasp", "lift"),
            ("rotate", "lift", "rotate"),
        ]
        if "transit" in waypoints:
            carry_routes.extend(
                [
                    ("transfer", "rotate", "transit"),
                    ("approach_target", "transit", "hover"),
                ]
            )
        else:
            carry_routes.append(("transfer", "rotate", "hover"))
        carry_routes.append(("place", "hover", "place"))
        for phase, left, right in carry_routes:
            routes[phase] = (
                np.asarray(waypoints[left]["T_world_assemble"]),
                np.asarray(waypoints[right]["T_world_assemble"]),
                True,
            )
        routes["retract"] = (place, retract, False)
        breakdown = {"ik": 0.0, "timing": 0.0, "collision": 0.0}
        ik_started = time.perf_counter()
        candidate_paths = []
        for rotation_direction in ("shortest", "opposite"):
            for refinement in (1, 2, 4):
                print(
                    f"[grasp] IK search: {rotation_direction}, refinement {refinement}",
                    flush=True,
                )
                tcp_samples, ranges = [], {}
                for phase, (start, end, is_object) in routes.items():
                    samples = interpolate_poses(
                        start,
                        end,
                        settings["cartesian_step"] / refinement,
                        settings["rotation_step_degrees"] / refinement,
                        rotation_direction=(
                            rotation_direction if phase == "rotate" else "shortest"
                        ),
                    )
                    if is_object:
                        samples = [pose @ grip for pose in samples]
                    begin = max(0, len(tcp_samples) - 1)
                    tcp_samples.extend(samples if not tcp_samples else samples[1:])
                    ranges[phase] = (begin, len(tcp_samples))
                solutions = []
                for pose in tcp_samples:
                    valid, q = self.robot.compute_ik(
                        torch.tensor(
                            pose[None], dtype=torch.float32, device=self.sim.device
                        ),
                        name="arm",
                        return_all_solutions=True,
                    )
                    solutions.append(q[0, valid[0]].cpu().numpy())
                paths = select_ik_paths(
                    solutions,
                    seed,
                    limits,
                    12.0,
                    joint_cost_weights=settings["joint_cost_weights"],
                )
                if paths:
                    candidate_paths.extend(
                        (path, ranges, rotation_direction) for path in paths[:8]
                    )
                    break
        breakdown["ik"] = time.perf_counter() - ik_started
        if not candidate_paths:
            return {
                "accepted": False,
                "reason": "No continuous IK branch path found within joint limits and 12-degree knot increments",
                "planning_seconds": time.perf_counter() - started,
                "planning_stage_seconds": breakdown.copy(),
            }
        failures = []
        # Different physical IK branches can have very different obstacle clearance.
        for branch_index, (joint_path, ranges, rotation_direction) in enumerate(
            candidate_paths
        ):
            timing_started = time.perf_counter()
            try:
                self.trajectory, timing = self._time_path(joint_path, ranges, candidate)
            except ValueError as error:
                failures.append(
                    {
                        "reason": f"Timing failed: {error}",
                        "rotation_direction": rotation_direction,
                    }
                )
                continue
            finally:
                breakdown["timing"] += time.perf_counter() - timing_started
            endpoint_errors = self._endpoint_errors(candidate)
            if any(
                v["position_m"] > 0.003 or v["rotation_degrees"] > 1
                for v in endpoint_errors.values()
            ):
                failures.append(
                    {"reason": "Waypoint FK error", "waypoint_errors": endpoint_errors}
                )
                continue
            metrics = motion_metrics(
                self.trajectory["positions"][0, :, arm_ids].cpu().numpy(),
                [self.robot.joint_names[i] for i in arm_ids],
                joint_cost_weights=settings["joint_cost_weights"],
            )
            metrics.update(
                duration_seconds=float(self.trajectory["dt"].sum()),
                peak_velocity_rad_s=max(
                    t["validation"]["peak_velocity"] for t in timing.values()
                ),
                peak_acceleration_rad_s2=max(
                    t["validation"]["peak_acceleration"] for t in timing.values()
                ),
                peak_jerk_rad_s3=max(
                    t["validation"]["peak_jerk"] for t in timing.values()
                ),
            )
            q = self.trajectory["positions"][0, :, arm_ids].cpu().numpy()
            reason = motion_rejection(metrics, self.config)
            if np.any(q < limits[:, 0] - 1e-5) or np.any(q > limits[:, 1] + 1e-5):
                reason = "Joint spline exceeds physical joint limits"
            if reason:
                failures.append({"reason": reason, "motion_metrics": metrics})
                continue
            print(
                f"[grasp] Collision check: branch {branch_index + 1}/{len(candidate_paths)} "
                f"({rotation_direction}), {len(q)} trajectory samples",
                flush=True,
            )
            collision_started = time.perf_counter()
            check = self._check_trajectory(candidate, geometry)
            breakdown["collision"] += time.perf_counter() - collision_started
            if not check["accepted"]:
                failures.append(check | {"motion_metrics": metrics})
                continue
            self.initial_qpos = self.trajectory["positions"][:, 0].clone()
            for target in (False, True):
                self.robot.set_qpos(self.initial_qpos, target=target)
            self.robot.clear_dynamics()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "trajectory.npz"
            np.savez_compressed(
                path,
                **{
                    key: value.cpu().numpy()
                    for key, value in self.trajectory.items()
                    if key != "phases"
                },
                initial_qpos=self.initial_qpos.cpu().numpy(),
            )
            return check | {
                "trajectory": str(path),
                "planner": "continuous arm blocks: IK + TOPPRA with a smooth speed clock",
                "phase_ranges": self.trajectory["phases"],
                "waypoints": waypoints,
                "waypoint_errors": endpoint_errors,
                "motion_metrics": metrics,
                "timing": timing,
                "ik_samples": len(joint_path),
                "ik_branch_index": branch_index,
                "rotation_direction": rotation_direction,
                "planning_seconds": time.perf_counter() - started,
                "planning_stage_seconds": breakdown.copy(),
                "physical_success": None,
                "initial_robot_qpos": self.initial_qpos.cpu().numpy().tolist(),
                "initialization": "Robot starts at the open pre-grasp pose; no home-to-pre-grasp motion is planned",
                "self_collision_checked": False,
            }
        self.trajectory = None
        return {
            "accepted": False,
            "reason": "No checked smooth IK branch trajectory passed",
            "branch_failures": failures,
            "planning_seconds": time.perf_counter() - started,
            "planning_stage_seconds": breakdown.copy(),
        }

    def _time_path(
        self, q: np.ndarray, ranges: dict, candidate: dict
    ) -> tuple[dict, dict]:
        """Time continuous arm blocks separated only by gripper changes."""
        import torch
        from scripts.tools.grasp_assemble._trajectory import time_parameterize

        settings = self.config["motion"]
        control_dt = self.sim.sim_config.physics_dt
        arm_ids = self.robot.get_joint_ids(name="arm")
        finger_ids = [
            self.robot.joint_names.index(f"gripper_finger{i}_joint_1") for i in (1, 2)
        ]
        template = self.seed_qpos[0].cpu().numpy().copy()
        channels = {
            key: []
            for key in ("positions", "velocities", "accelerations", "jerks", "dt")
        }
        phases, reports, size = {}, {}, 0

        def append(name: str, values: dict) -> None:
            nonlocal size
            start = max(0, size - 1)
            skip = int(size > 0)
            for key in channels:
                channels[key].append(values[key][skip:])
            size += len(values["dt"]) - skip
            phases[name] = {"start": start, "stop": size}

        def hand(
            name: str, pose: np.ndarray, start_q: float, end_q: float, hold: float = 0
        ) -> None:
            duration = np.ceil(settings["hand_duration"] / control_dt) * control_dt
            t = np.linspace(0, 1, round(duration / control_dt) + 1)
            clock = t**3 * (10 - 15 * t + 6 * t * t)
            pos = np.repeat(template[None], len(t), axis=0)
            pos[:, arm_ids] = pose
            pos[:, finger_ids] = (start_q + (end_q - start_q) * clock)[:, None]
            values = {
                "positions": pos,
                "dt": np.r_[0.0, np.full(len(t) - 1, control_dt)],
            }
            for key, derivative in (
                ("velocities", 30 * t * t * (1 - t) ** 2 / duration),
                ("accelerations", 60 * t * (1 - t) * (1 - 2 * t) / duration**2),
                ("jerks", (60 - 360 * t + 360 * t * t) / duration**3),
            ):
                arr = np.zeros_like(pos)
                arr[:, finger_ids] = ((end_q - start_q) * derivative)[:, None]
                values[key] = arr
            if hold:
                count = int(np.ceil(hold / control_dt))
                for key in values:
                    tail = (
                        np.full(count, control_dt)
                        if key == "dt"
                        else np.repeat(
                            (pos[-1] if key == "positions" else np.zeros_like(pos[-1]))[
                                None
                            ],
                            count,
                            axis=0,
                        )
                    )
                    values[key] = np.concatenate([values[key], tail])
            append(name, values)

        closed = candidate["gripper_close_qpos"]
        blocks = {
            "approach": ("approach",),
            "carry": tuple(
                name for name in ranges if name not in ("approach", "retract")
            ),
            "retract": ("retract",),
        }
        for name, members in blocks.items():
            begin, end = ranges[members[0]][0], ranges[members[-1]][1]
            timed = time_parameterize(
                q[begin:end],
                control_dt=control_dt,
                velocity_limit=settings["velocity_limit"],
                acceleration_limit=settings["acceleration_limit"],
                jerk_limit=settings["jerk_limit"],
                duration_scale=settings["duration_scale"],
            )
            reports[name] = {
                key: timed[key] for key in ("duration", "toppra_duration", "validation")
            }
            values = {"dt": timed["dt"]}
            for key in ("positions", "velocities", "accelerations", "jerks"):
                data = (
                    np.repeat(template[None], len(timed["dt"]), axis=0)
                    if key == "positions"
                    else np.zeros((len(timed["dt"]), len(template)))
                )
                data[:, arm_ids] = timed[key]
                if key == "positions":
                    data[:, finger_ids] = (
                        0 if name in ("approach", "retract") else closed
                    )
                values[key] = data
            block_start = max(0, size - 1)
            append(name, values)
            reports[name]["phases"] = list(members)
            reports[name]["waypoint_passages"] = {}
            for phase in members:
                left, right = ranges[phase]
                first = int(timed["waypoint_indices"][left - begin])
                last = int(timed["waypoint_indices"][right - begin - 1])
                phases[phase] = {
                    "start": block_start + first,
                    "stop": block_start + last + 1,
                }
                reports[name]["waypoint_passages"][phase] = {
                    "time_seconds": float(timed["waypoint_times"][right - begin - 1]),
                    "sample_index": block_start + last,
                    "arm_speed_rad_s": float(np.linalg.norm(timed["velocities"][last])),
                }
            if name == "approach":
                hand("close", q[end - 1], 0, closed, settings["grasp_hold_seconds"])
            elif name == "carry":
                hand("release", q[end - 1], closed, 0)
        result = {
            key: torch.tensor(
                np.concatenate(value)[None], dtype=torch.float32, device=self.sim.device
            )
            for key, value in channels.items()
        }
        result["phases"] = phases
        return result, reports

    def _endpoint_errors(self, candidate: dict) -> dict:
        """Measure named waypoints at their nearest planned control sample."""
        from scipy.spatial.transform import Rotation

        errors = {}
        arm_ids = self.robot.get_joint_ids(name="arm")
        endpoints = [
            ("grasp", "approach"),
            ("lift", "lift"),
            ("rotate", "rotate"),
        ]
        if "transit" in candidate["waypoints"]:
            endpoints.extend([("transit", "transfer"), ("hover", "approach_target")])
        else:
            endpoints.append(("hover", "transfer"))
        endpoints.append(("place", "place"))
        for role, phase in endpoints:
            step = self.trajectory["phases"][phase]["stop"] - 1
            pose = (
                self.robot.compute_fk(
                    self.trajectory["positions"][:, step, arm_ids],
                    name="arm",
                    to_matrix=True,
                )[0]
                .cpu()
                .numpy()
            )
            target = np.asarray(candidate["waypoints"][role]["T_world_tcp"])
            errors[role] = {
                "position_m": float(np.linalg.norm(pose[:3, 3] - target[:3, 3])),
                "rotation_degrees": float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            pose[:3, :3].T @ target[:3, :3]
                        ).magnitude()
                    )
                ),
            }
        return errors

    def _joint_positions(self) -> Any:
        """Resolve passive mimic columns before full-chain FK or playback."""
        positions = self.trajectory["positions"].clone()
        for child, parent, scale, offset in zip(
            getattr(self.robot, "mimic_ids", ()),
            getattr(self.robot, "mimic_parents", ()),
            getattr(self.robot, "mimic_multipliers", ()),
            getattr(self.robot, "mimic_offsets", ()),
        ):
            positions[..., child] = positions[..., parent] * scale + offset
        return positions

    def _check_trajectory(self, candidate: dict, geometry: Any) -> dict:
        import fcl
        import trimesh
        from scripts.tools.assemble._collision import _triangle_object

        positions = self._joint_positions()[0]
        q_by_name = {
            name: positions[:, i] for i, name in enumerate(self.robot.joint_names)
        }
        fk = self.robot.pk_chain.forward_kinematics(q_by_name)
        root = self.robot.get_local_pose(to_matrix=True)[0].cpu().numpy()
        parts = []
        ground_bounds = {}
        for name in self.robot.link_names:
            vertices, faces = self.robot.get_link_vert_face(name)
            if vertices.numel() == 0 or faces.numel() == 0:
                continue
            mesh = trimesh.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy(), process=False
            )
            matrices = root @ fk[name].get_matrix().cpu().numpy()
            center = mesh.bounds.mean(axis=0)
            half_extents = np.diff(mesh.bounds, axis=0)[0] / 2
            # The transformed local AABB encloses the complete mesh. Most
            # links are high enough that its lower z bound proves clearance.
            ground_bounds[name] = (
                matrices[:, 2, :3] @ center
                + matrices[:, 2, 3]
                - np.abs(matrices[:, 2, :3]) @ half_extents
            )
            parts.append(
                (
                    name,
                    mesh,
                    _triangle_object(mesh),
                    matrices,
                )
            )
        base = _triangle_object(geometry.base_world)
        count = positions.shape[0]
        close_start = self.trajectory["phases"]["close"]["start"]
        initial_assemble = geometry.assemble_mesh.copy()
        initial_assemble.apply_transform(geometry.initial)
        assemble_obj = _triangle_object(initial_assemble)
        # All planned robot samples are checked against the base. Full self
        # collision and continuous swept volumes remain outside this checker.
        for step in range(count):
            for name, mesh, obj, matrices in parts:
                pose = matrices[step]
                obj.setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
                if fcl.collide(
                    base, obj, fcl.CollisionRequest(), fcl.CollisionResult()
                ):
                    return {
                        "accepted": False,
                        "reason": f"Robot link {name} hits base at joint sample {step}",
                    }
                if step < close_start and fcl.collide(
                    assemble_obj, obj, fcl.CollisionRequest(), fcl.CollisionResult()
                ):
                    return {
                        "accepted": False,
                        "reason": f"Robot link {name} hits initial assemble object before closure at joint sample {step}",
                    }
                if (
                    name not in {"base_link", "arm_base_link"}
                    and ground_bounds[name][step] < -0.002 + 1e-9
                    and (mesh.vertices @ pose[2, :3] + pose[2, 3]).min() < -0.002
                ):
                    return {
                        "accepted": False,
                        "reason": f"Robot link {name} hits ground at joint sample {step}",
                    }
        # Reuse batched full-chain FK for the carried object, and retain the
        # placed object in the collision world after release. Only the two
        # fingers may intentionally contact the object before fully opening.
        carry_start = self.trajectory["phases"]["lift"]["start"]
        release_start = self.trajectory["phases"]["release"]["start"]
        fingers_open = self.trajectory["phases"]["release"]["stop"] - 1
        tcp_offset = np.eye(4)
        tcp_offset[2, 3] = TCP_Z
        tcp_matrices = (
            root @ fk["gripper_base_link_1"].get_matrix().cpu().numpy() @ tcp_offset
        )
        inverse_grasp = np.linalg.inv(np.asarray(candidate["T_assemble_tcp"]))
        carried_poses = tcp_matrices @ inverse_grasp
        placed_pose = np.asarray(candidate["T_world_assemble_target"])
        moving_assemble = _triangle_object(geometry.assemble_mesh)
        contact_links = {"gripper_finger1_link_1", "gripper_finger2_link_1"}
        for step in range(close_start, count):
            carrying = carry_start <= step < release_start
            if step < carry_start:
                assemble_pose, assemble_state = geometry.initial, "initial"
            elif carrying:
                assemble_pose, assemble_state = carried_poses[step], "carried"
            else:
                assemble_pose, assemble_state = placed_pose, "placed"
            moving_assemble.setTransform(
                fcl.Transform(assemble_pose[:3, :3], assemble_pose[:3, 3])
            )
            for name, _, obj, matrices in parts:
                if name in contact_links and step < fingers_open:
                    continue
                pose = matrices[step]
                obj.setTransform(fcl.Transform(pose[:3, :3], pose[:3, 3]))
                if fcl.collide(
                    moving_assemble, obj, fcl.CollisionRequest(), fcl.CollisionResult()
                ):
                    return {
                        "accepted": False,
                        "reason": (
                            f"Robot link {name} hits {assemble_state} assemble object "
                            f"at joint sample {step}"
                        ),
                    }
            if carrying:
                reason = geometry._world_clearance(
                    tcp_matrices[step], candidate["gripper_clear_qpos"]
                )
                if (
                    geometry.assemble_mesh.vertices @ assemble_pose[:3, :3].T
                    + assemble_pose[:3, 3]
                )[:, 2].min() < -0.002:
                    reason = "Carried assemble object intersects the ground"
                if reason or geometry._overlap(geometry.object_check, assemble_pose):
                    return {
                        "accepted": False,
                        "reason": reason
                        or f"Carried assemble object hits base at joint sample {step}",
                    }
        return {
            "accepted": True,
            "joint_samples": count,
            "checks": "IK, sampled robot-link/base and ground, robot/initial assemble object, carried and placed assemble object, sampled carried assemble object and gripper/base; only finger/assemble object contact is exempt until fully open; no continuous or self-collision proof",
        }

    def _attach_grasp(self, candidate: dict) -> None:
        """Weld the measured grasp to the physical palm without moving the assemble object."""
        from scipy.spatial.transform import Rotation

        if self._grasp_constraint is not None:
            raise RuntimeError("A grasp constraint is already active")
        palm = (
            self.robot.get_link_pose("gripper_base_link_1", to_matrix=True)[0]
            .cpu()
            .numpy()
        )
        assemble = self.assemble.get_local_pose(to_matrix=True)[0].cpu().numpy()
        tcp_offset = np.eye(4)
        tcp_offset[2, 3] = TCP_Z
        measured = np.linalg.inv(assemble) @ palm @ tcp_offset
        expected = np.asarray(candidate["T_assemble_tcp"])
        distance = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
        angle = float(
            np.rad2deg(
                Rotation.from_matrix(measured[:3, :3].T @ expected[:3, :3]).magnitude()
            )
        )
        hand_ids = [
            self.robot.joint_names.index(f"gripper_finger{i}_joint_1") for i in (1, 2)
        ]
        fingers = self.robot.get_qpos()[0, hand_ids].cpu().numpy()
        contact_error = float(
            np.max(np.abs(fingers - np.asarray(candidate["finger_contact_qpos"])))
        )
        if contact_error > 0.006:
            raise RuntimeError(
                f"Refusing grasp attachment: fingers are {contact_error:.4f} m from expected contact"
            )
        if distance > 0.01 or angle > 10:
            raise RuntimeError(
                f"Refusing grasp attachment: measured TCP drift {distance:.4f} m / {angle:.2f} deg"
            )
        # The public rigid-object constraint facade does not accept robot links.
        # Borrow native endpoints only in this single-arena Default adapter.
        palm_body = self.robot._entities[0].render_articulation.get_physical_body(
            "gripper_base_link_1"
        )
        assemble_body = self.assemble._entities[0].native().get_rigid_body()
        name = "grasp_assemble_hold"
        frame = (np.linalg.inv(palm) @ assemble).astype(np.float32)
        handle = self.sim.get_env(0).create_fixed_constraint_bodies(
            name, palm_body, assemble_body, frame, np.eye(4, dtype=np.float32)
        )
        if handle is None:
            raise RuntimeError("Failed to create the temporary grasp constraint")
        self._grasp_constraint = name
        self._constraint_events.append(
            {
                "event": "attached",
                "phase": "grasp_closed",
                "measured_T_assemble_tcp": measured.tolist(),
                "position_drift_m": distance,
                "rotation_drift_degrees": angle,
                "finger_qpos": fingers.tolist(),
            }
        )
        print("[grasp] Grasp closed: temporary grasp constraint attached", flush=True)

    def _release_grasp(self, phase: str = "cleanup") -> None:
        if self._grasp_constraint is not None:
            self.sim.get_env(0).remove_constraint(self._grasp_constraint)
            self._grasp_constraint = None
            self._constraint_events.append({"event": "released", "phase": phase})
            print(f"[grasp] Temporary grasp constraint released: {phase}", flush=True)

    def show(self) -> None:
        """Open the native viewer at the planned scene's inspection camera."""
        self.sim.open_window()
        self.sim.get_world().get_windows().set_look_at(
            eye=(-1.15, -1.0, 0.95), look_at=(-0.35, 0, 0.25), up=(0, 0, 1)
        )

    def replay(self, candidate: dict, directory: Path, headless: bool = False) -> dict:
        """Replay direct waypoints with the configured grasp mode and measure error.

        Args:
            candidate: The candidate just accepted by plan().
            directory: Video destination.
            headless: Record without opening a native window.

        Returns:
            Observed final error and success, separate from planning acceptance.
        """
        from scipy.spatial.transform import Rotation
        from embodichain.lab.sim.motion.execution import (
            play_joint_trajectory,
            JointTrajectoryPlaybackCfg,
        )

        if self.trajectory is None:
            raise RuntimeError("Plan the grasp before replay")
        if not headless:
            self.show()
        settings = execution_settings(self.config.get("execution"))
        mode = settings["grasp_mode"]
        self._constraint_events = []
        recording = self.sim.start_window_record(
            save_path=str(directory / "pick_place.mp4"),
            fps=max(
                1,
                min(
                    10,
                    int(
                        220
                        / (
                            float(self.trajectory["dt"].sum())
                            + (
                                settings["release_hold_steps"]
                                + settings["settle_steps"]
                            )
                            * self.sim.sim_config.physics_dt
                        )
                    ),
                ),
            ),
            max_memory=512,
            use_sim_time=True,
            look_at=((-1.15, -1.0, 0.95), (-0.35, 0, 0.25), (0, 0, 1)),
        )
        try:
            positions = self._joint_positions()
            phase_observations = []
            begin = 0
            # Keep the held-object motion in a single playback call. Restarting
            # playback at lift/rotate/hover would force its endpoint velocities
            # to zero even though the planned carry block is continuous.
            boundaries = (
                ("grasp_closed", self.trajectory["phases"]["lift"]["start"]),
                ("before_release", self.trajectory["phases"]["release"]["start"]),
                ("retracted", self.trajectory["positions"].shape[1] - 1),
            )
            for phase, end in boundaries:
                intervals = self.trajectory["dt"][:, begin : end + 1].clone()
                intervals[:, 0] = 0
                play_joint_trajectory(
                    self.sim,
                    self.robot,
                    positions=positions[:, begin : end + 1],
                    dt=intervals,
                    cfg=JointTrajectoryPlaybackCfg(
                        joint_command_mode="position_velocity"
                    ),
                    joint_ids=list(range(self.robot.get_qpos().shape[-1])),
                )
                if phase == "before_release":
                    # Hold the closed grasp at the placement endpoint while
                    # the position controller catches up, before detaching.
                    self.sim.update(step=settings["release_hold_steps"])
                tcp = (
                    self.robot.compute_fk(
                        self.robot.get_qpos(name="arm"), name="arm", to_matrix=True
                    )[0]
                    .cpu()
                    .numpy()
                )
                assemble = self.assemble.get_local_pose(to_matrix=True)[0].cpu().numpy()
                phase_observations.append(
                    {
                        "phase": phase,
                        "T_world_assemble": assemble.tolist(),
                        "measured_T_assemble_tcp": (
                            np.linalg.inv(assemble) @ tcp
                        ).tolist(),
                        "robot_qpos": self.robot.get_qpos()[0].cpu().numpy().tolist(),
                    }
                )
                if phase == "grasp_closed":
                    self.assemble.clear_dynamics()
                    if mode == "fixed_constraint":
                        self._attach_grasp(candidate)
                elif phase == "before_release":
                    self._release_grasp("before_release")
                begin = end
            self.sim.update(step=settings["settle_steps"])
            observed = self.assemble.get_local_pose(to_matrix=True)[0].cpu().numpy()
            target = np.asarray(candidate["T_world_assemble_target"])
            position_error = float(np.linalg.norm(observed[:3, 3] - target[:3, 3]))
            angle_error = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        observed[:3, :3].T @ target[:3, :3]
                    ).magnitude()
                )
            )
            return {
                "trajectory_completed": True,
                "physical_success": position_error <= 0.015
                and angle_error <= settings["rotation_tolerance_degrees"],
                "pose_tolerances": {
                    "position_m": 0.015,
                    "rotation_degrees": settings["rotation_tolerance_degrees"],
                },
                "grasp_mode": mode,
                "execution_settings": settings,
                "constraint_events": self._constraint_events.copy(),
                "constraint_active_at_end": self._grasp_constraint is not None,
                "position_error_m": position_error,
                "rotation_error_degrees": angle_error,
                "observed_T_world_assemble": observed.tolist(),
                "phase_observations": phase_observations,
                "video": str(directory / "pick_place.mp4") if recording else None,
            }
        finally:
            self._release_grasp()
            if recording:
                self.sim.stop_window_record()

    def close(self) -> None:
        """Release this session before the caller flushes native cleanup."""
        self._release_grasp()
        self.sim.destroy(exit_process=False)
