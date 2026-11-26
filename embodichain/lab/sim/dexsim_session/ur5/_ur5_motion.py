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

"""Bind the qualified UR5 deployment to the motion-service contract."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.lab.sim.atomic_skills.state import HeldObjectState
from embodichain.lab.sim.dexsim_session._motion_jobs import (
    JobRejected,
    MotionJobs,
    Work,
)
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenerator,
    MotionGenOptions,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlannerCfg,
)
from embodichain.lab.sim.motion.planners.utils import PlanState

from .ur5_pick_place import _Episode, _PHYSICS_DT

__all__: list[str] = []

_PROFILE = "ur5/trapezoidal@1"
_CHECKS = {
    "joint_position_limits": True,
    "joint_velocity_limits": True,
    "collision": "not_checked",
}


class Ur5Motion:
    """Keep the exact configured semantic runtime across separate MCP jobs."""

    def __init__(self, episode: _Episode, output: Path) -> None:
        self.episode = episode
        self.jobs = MotionJobs(episode.context, output, self.progress)
        self.state = "ready"
        self.grasp: dict[str, Any] | None = None
        self.plans: dict[str, dict[str, Any]] = {}
        self.generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=TrapezoidalPlannerCfg(
                    robot_uid=episode.robot.uid,
                    sim_instance_id=episode.manager.instance_id,
                )
            )
        )
        self.program_plan = f"ecprogram_{uuid4().hex}"

    def progress(self) -> dict[str, Any]:
        e = self.episode
        return {
            "phase": e.phase,
            "physics_step": e.physics_step,
            "control_step": e.control_step,
            "simulation_time_s": e.physics_step * _PHYSICS_DT,
        }

    def capabilities(self) -> dict[str, Any]:
        e = self.episode
        e.manager.validate()
        return {
            "session_id": e.context.session_id,
            "topology_revision": e.scene.topology_revision,
            "robot_ref": e.robot.uid,
            "control_parts": {
                name: e.robot.control_parts[name] for name in ("arm", "eef")
            },
            "kinematics": {
                "control_part": "arm",
                "frame": "arena",
                "flange_link": e.arm.end_link_name,
            },
            "planner_profiles": {
                _PROFILE: {
                    "joint_waypoints": True,
                    "tcp_waypoints": True,
                    "collision_check": False,
                }
            },
            "skills": {
                "pick_up": {"object_refs": [e.cube.path]},
                "place": {
                    "target_refs": e.destinations or ["platform_top"],
                    "requires_grasp": True,
                },
            },
            "programs": [e.compiled.program_id],
            "control_dt_s": e.period,
            "writer_authorized": True,
            "motion_world_state": self.state,
            **self.progress(),
        }

    def operations(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "fk": self.fk,
            "ik": self.ik,
            "plan_motion": self.plan_motion,
            "execute_motion": self.execute_motion,
            "start_skill": self.start_skill,
            "get_grasp": self.get_grasp,
            "prepare_program": self.prepare_program,
            "start_run": self.start_run,
            "cancel_run": self.jobs.cancel,
        }

    def close(self) -> None:
        self.jobs.close()
        segments = getattr(self, "_segments", None)
        if segments is not None:
            segments.close()

    def _arm_joint_fields(self, part: str) -> dict[str, Any]:
        self.episode.manager.validate()
        if part != "arm":
            raise ValueError("This UR5 profile exposes kinematics for arm")
        robot = self.episode.robot
        names = [robot.joint_names[i] for i in robot.get_joint_ids(part)]
        return {"joint_names": names, "units": ["rad"] * len(names)}

    def _joint_vector(self, part: str, joints: dict[str, Any] | None) -> torch.Tensor:
        fields = self._arm_joint_fields(part)
        e = self.episode
        if joints is None:
            return e.robot.get_qpos(part).clone()
        if (
            joints.get("joint_names") != fields["joint_names"]
            or joints.get("units") != fields["units"]
        ):
            raise ValueError("Joint names/order or units differ from the bound robot")
        qpos = torch.as_tensor(
            joints["qpos"], dtype=torch.float32, device=e.robot.device
        )
        if (
            qpos.shape != (len(fields["joint_names"]),)
            or not torch.isfinite(qpos).all()
        ):
            raise ValueError("Expected one finite value per arm joint")
        self._check_limits(qpos)
        return qpos.unsqueeze(0)

    def _check_limits(self, qpos: torch.Tensor) -> None:
        lower, upper = self.episode.robot.get_qpos_limits("arm")[0].unbind(-1)
        if not torch.isfinite(qpos).all() or ((qpos < lower) | (qpos > upper)).any():
            raise ValueError("ARM_JOINT_LIMIT")

    @staticmethod
    def _pose(target: dict[str, Any]) -> torch.Tensor:
        pos = np.asarray(target.get("position_m"), dtype=np.float64)
        quat = np.asarray(target.get("quaternion_xyzw"), dtype=np.float64)
        if (
            target.get("frame") != "arena"
            or pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(pos).all()
            or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1.0, rtol=0, atol=1e-3)
        ):
            raise ValueError(
                "Expected an arena-frame position in metres and a unit xyzw quaternion"
            )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
        transform[:3, 3] = pos
        return torch.as_tensor(transform, dtype=torch.float32).unsqueeze(0)

    def fk(
        self, control_part: str, joints: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        qpos = self._joint_vector(control_part, joints)
        e = self.episode
        pose = e.robot.compute_fk(qpos, name=control_part, to_matrix=True)[0].numpy()
        return {
            "frame": "arena",
            "position_m": pose[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
            "flange_link": e.arm.end_link_name,
            "link_from_tcp": e.link_from_tcp.tolist(),
            "physics_step": e.physics_step,
        }

    def ik(
        self,
        control_part: str,
        target: dict[str, Any],
        seed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fields = self._arm_joint_fields(control_part)
        seed_qpos = self._joint_vector(control_part, seed)
        ok, qpos = self.episode.robot.compute_ik(
            self._pose(target), joint_seed=seed_qpos, name=control_part
        )
        if not bool(ok.all()):
            return {"success": False, **fields, "qpos": None}
        self._check_limits(qpos)
        return {"success": True, **fields, "qpos": qpos[0].tolist()}

    def _snapshot(self) -> dict[str, Any]:
        e = self.episode
        e.manager.validate()
        return {
            "session_id": e.context.session_id,
            "topology_revision": e.scene.topology_revision,
            "physics_step": e.physics_step,
            "robot_qpos": e.robot.get_qpos()[0].tolist(),
            "drive_qpos": e.robot.get_qpos(target=True)[0].tolist(),
            "robot_pose": e.robot.get_local_pose(to_matrix=True)[0].tolist(),
            "cube_pose": e.cube.get_world_pose().tolist(),
            "platform_pose": e.platform.get_world_pose().tolist(),
        }

    def plan_motion(
        self, control_part: str, planner_profile: str, waypoints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        fields = self._arm_joint_fields(control_part)
        if planner_profile != _PROFILE or not 1 <= len(waypoints) <= 12:
            raise ValueError(
                "Expected the qualified planner profile and 1–12 waypoints"
            )
        if self.jobs.active is not None:
            raise ValueError("WORLD_BUSY")
        e = self.episode
        start = self._joint_vector(control_part, None)
        states = [PlanState.from_qpos(start)]
        for waypoint in waypoints:
            if "frame" in waypoint:
                previous = states[-1].qpos
                target = self.ik(
                    control_part, waypoint, {**fields, "qpos": previous[0].tolist()}
                )
                if not target["success"]:
                    return {"success": False, "error": "IK_UNREACHABLE"}
                waypoint = target
            states.append(
                PlanState.from_qpos(self._joint_vector(control_part, waypoint))
            )
        raw = self.generator.generate(
            states,
            MotionGenOptions(
                control_part=control_part, velocity_limit=0.5, acceleration_limit=1.0
            ),
        )
        if not bool(torch.as_tensor(raw.success).all()) or raw.positions is None:
            return {"success": False, "error": "PLANNING_FAILED"}
        positions, _, dt, _ = retime_to_control_grid(raw.positions, raw.dt, e.period)
        self._check_limits(positions)
        durations = dt[:, 1:].clamp_min(1e-9).unsqueeze(-1)
        velocity = (positions[:, 1:] - positions[:, :-1]).abs() / durations
        limits = e.robot.get_qvel_limits(control_part).abs()
        if (velocity > limits[:, None, :] + 1e-5).any():
            raise ValueError("RETIME_VELOCITY_LIMIT")
        snapshot = self._snapshot()
        plan_id = f"ecplan_{uuid4().hex}"
        artifact = self.jobs.output / f"{plan_id}.json"
        artifact.write_text(
            json.dumps(
                {
                    **fields,
                    "snapshot": snapshot,
                    "positions": positions[0].tolist(),
                    "arrival_dt_s": dt[0].tolist(),
                }
            ),
            encoding="utf-8",
        )
        if len(self.plans) >= 16:
            del self.plans[next(iter(self.plans))]
        self.plans[plan_id] = {"positions": positions, "snapshot": snapshot}
        return {
            "success": True,
            "motion_plan_id": plan_id,
            "trajectory": str(artifact),
            "duration_s": float(dt.sum()),
            "samples": positions.shape[1],
            "session_id": snapshot["session_id"],
            "physics_step": snapshot["physics_step"],
            "checks": _CHECKS,
            **fields,
        }

    def execute_motion(self, motion_plan_id: str, command_id: str) -> dict[str, Any]:
        return self.jobs.start(
            command_id,
            "motion",
            {"motion_plan_id": motion_plan_id},
            lambda _folder: self._play(motion_plan_id),
        )

    def _play(self, motion_plan_id: str) -> Work:
        saved = self.plans.get(motion_plan_id)
        if saved is None or self._snapshot() != saved["snapshot"]:
            raise JobRejected(
                "STALE_PLAN",
                "The robot, Scene or physics clock differs from the plan snapshot",
            )
        if self.state != "ready":
            raise JobRejected(
                "TASK_STATE_CONFLICT", "Motion playback requires a ready, empty gripper"
            )
        e = self.episode
        ids = e.robot.get_joint_ids("arm")
        command = e.robot.get_qpos(target=True).clone()
        for point in saved["positions"][0, 1:]:
            command[0, ids] = point
            yield e.advance("motion", command)
        final_error = float((e.robot.get_qpos("arm")[0] - command[0, ids]).abs().max())
        return {"passed": final_error <= 0.03, "final_joint_error_rad": final_error}

    def prepare_program(self, program_id: str) -> dict[str, Any]:
        self.episode.manager.validate()
        if program_id != self.episode.compiled.program_id:
            raise ValueError("Unknown registered Task Program")
        return {
            "plan_id": self.program_plan,
            "session_id": self.episode.context.session_id,
            "program_id": program_id,
        }

    def start_run(self, plan_id: str, command_id: str) -> dict[str, Any]:
        if plan_id != self.program_plan:
            raise ValueError("Unknown Task Program plan")
        return self.jobs.start(
            command_id, "program", {"plan_id": plan_id}, self._program
        )

    def start_skill(
        self,
        skill_id: str,
        object_ref: str,
        command_id: str,
        grasp_ref: str | None = None,
        target_ref: str | None = None,
    ) -> dict[str, Any]:
        e = self.episode
        if object_ref != e.cube.path:
            raise ValueError("Only the registered cube is bound to this deployment")
        if skill_id == "pick_up" and grasp_ref is None and target_ref is None:
            make_work = self._pick
        elif (
            skill_id == "place"
            and grasp_ref
            and target_ref in (e.destinations or ["platform_top"])
        ):

            def make_work(_folder):
                if e.live_paths is not None:
                    e.select_platform(target_ref)
                return (yield from self._place(grasp_ref))

        else:
            raise ValueError(
                "Expected pick_up(cube) or place(cube, grasp_ref, platform_top)"
            )
        return self.jobs.start(
            command_id,
            "skill",
            dict(
                skill_id=skill_id,
                object_ref=object_ref,
                grasp_ref=grasp_ref,
                target_ref=target_ref,
            ),
            make_work,
        )

    def get_grasp(self, grasp_ref: str) -> dict[str, Any]:
        self.episode.manager.validate()
        grasp = self.grasp
        if grasp is None or grasp_ref != grasp["grasp_ref"] or self.state != "held":
            return {"grasp_ref": grasp_ref, "verified": False, "code": "GRASP_LOST"}
        reading = self.episode.observe()
        rel = np.linalg.solve(reading["tcp_pose"], reading["cube_pose"])
        old = np.asarray(grasp["cube_in_tcp"])
        displacement = float(np.linalg.norm(rel[:3, 3] - old[:3, 3]))
        rotation = Rotation.from_matrix(old[:3, :3].T @ rel[:3, :3]).magnitude()
        verified = (
            displacement <= 0.006
            and rotation <= np.deg2rad(10.0)
            and self._closed(reading)
        )
        if not verified:
            self.grasp = None
            self.state = "grasp_lost"
            return {"grasp_ref": grasp_ref, "verified": False, "code": "GRASP_LOST"}
        return {**grasp, "verified": True, "physics_step": reading["physics_step"]}

    def _closed(self, reading: dict[str, Any]) -> bool:
        e = self.episode
        fraction = (reading["qpos"][e.grip_index] - e.grip_open) / (
            e.grip_close - e.grip_open
        )
        return fraction >= 0.3

    def _stop_partial_skill(self, previous_state: str) -> None:
        if self.state == previous_state:
            self.state = "interrupted"
            self.grasp = None
            self.episode.close_recording()
            segments = getattr(self, "_segments", None)
            if segments is not None:
                segments.close()

    def _pick(self, output: Path) -> Work:
        try:
            return (yield from self._perform_pick(output))
        finally:
            self._stop_partial_skill("picking")

    def _perform_pick(self, output: Path) -> Work:
        if self.state != "ready":
            raise JobRejected(
                "TASK_STATE_CONFLICT",
                "PickUp requires a fresh episode and an empty gripper",
            )
        e = self.episode
        self.state = "picking"
        start_z = e.cube.get_world_pose()[2, 3]
        e.record(output)
        self._bridge = e.adapter.create_bridge(e.compiled)
        self._segments = iter(self._bridge)
        pick = next(self._segments)
        assert pick.name == "pick"
        accepted = yield from e.run_segment(pick)
        held = list(e.recent)
        if not accepted or len(held) < 12:
            raise JobRejected(
                "GRASP_UNVERIFIED", "The Pick stage lacks a complete lift observation"
            )
        relative = np.asarray(
            [np.linalg.solve(row["tcp_pose"], row["cube_pose"]) for row in held]
        )
        lift = float(held[-1]["cube_pose"][2][3] - start_z)
        if (
            not all(
                self._closed(row) and row["cube_pose"][2][3] - start_z >= 0.03
                for row in held
            )
            or np.linalg.norm(relative[-1, :3, 3]) > 0.12
            or np.linalg.norm(relative[:, :3, 3] - relative[-1, :3, 3], axis=-1).max()
            > 0.005
        ):
            raise JobRejected(
                "GRASP_UNVERIFIED",
                "The cube must maintain its measured TCP relation after lifting",
            )
        # Replace the projected Pick effect with the verified current transform.
        runtime = self._bridge._runtime
        state = runtime.task_state
        [(resource, relation)] = state.held_objects.items()
        observed = HeldObjectState(
            semantics=relation.semantics,
            object_to_eef=torch.as_tensor(
                np.linalg.inv(relative[-1]), dtype=torch.float32
            ).unsqueeze(0),
            grasp_xpos=e.tcp().unsqueeze(0),
            env_mask=relation.env_mask,
        )
        runtime.adopt_verified_task_state(
            replace(state, held_objects={resource: observed})
        )
        proof = output / "grasp-evidence.json"
        self.grasp = {
            "evidence": str(proof),
            "grasp_ref": f"ecgrasp_{uuid4().hex}",
            "object_ref": e.cube.path,
            "session_id": e.context.session_id,
            "robot_ref": e.robot.uid,
            "cube_in_tcp": relative[-1].tolist(),
            "lift_m": lift,
            "verified": True,
        }
        proof.write_text(
            json.dumps({"grasp": self.grasp, "samples": held}, allow_nan=False),
            encoding="utf-8",
        )
        self.state = "held"
        e.phase = "holding"
        return {"passed": True, **self.grasp}

    def _place(self, grasp_ref: str) -> Work:
        try:
            return (yield from self._perform_place(grasp_ref))
        finally:
            self._stop_partial_skill("placing")

    def _perform_place(self, grasp_ref: str) -> Work:
        if not self.get_grasp(grasp_ref)["verified"]:
            raise JobRejected(
                "GRASP_LOST", "The requested grasp no longer qualifies for Place"
            )
        self.state = "placing"
        self.grasp = None
        e = self.episode
        if e.live_paths is not None:
            verified = self._bridge._runtime.task_state
            self._segments.close()
            self._bridge = e.place_bridge(verified)
            self._segments = iter(self._bridge)
        place = next(self._segments)
        assert place.name == "place"
        accepted = yield from e.run_segment(place)
        assert next(self._segments, None) is None
        completed = (
            accepted
            and self._bridge.program_completed
            and bool(self._bridge.completion_mask.all())
        )
        if completed:
            yield from e.settle()
        result = e.finish(program_passed=completed)
        self.state = "complete"
        e.phase = "complete"
        return result

    def _program(self, output: Path) -> Work:
        picked = yield from self._pick(output)
        return (yield from self._place(picked["grasp_ref"]))
