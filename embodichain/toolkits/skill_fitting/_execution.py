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

"""Direct native-skill planning and Gym execution for fitted recording bundles."""

from __future__ import annotations

from dataclasses import fields
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from embodichain.data_pipeline.skill_fitting import file_digest
from ._bundle import _contained
from ._robot import RobotProfile

__all__: list[str] = []


def _connection(
    start: torch.Tensor,
    stop: torch.Tensor,
    profile: RobotProfile,
    dt: float,
    minimum: float,
) -> torch.Tensor:
    """A zero-endpoint-velocity quintic within joint derivative bounds."""
    delta = (stop - start).abs()
    velocity = delta.new_tensor(profile.velocity_limits)
    acceleration = delta.new_tensor(profile.acceleration_limits)
    duration = max(
        minimum,
        float((1.875 * delta / velocity).max()),
        float(torch.sqrt(5.774 * delta / acceleration).max()),
    )
    count = max(2, math.ceil(duration / dt) + 1)
    if count > 200_000:
        raise ValueError("Connection exceeds execution command budget.")
    t = torch.linspace(0, 1, count, device=start.device, dtype=start.dtype)[:, None]
    path = start + (stop - start) * (10 * t**3 - 15 * t**4 + 6 * t**5)
    path[0], path[-1] = start, stop
    return path


def _invocation(engine: Any, span: dict[str, Any], objects: dict[str, Any]) -> Any:
    """Decode only fitted built-in goal/option types; never import config code."""
    from embodichain.lab.sim.atomic_skills import (
        GraspGoal,
        ObjectSemantics,
        PickUpMotionGoal,
        SlideMotionGoal,
        TwistMotionGoal,
        HandOverMotionGoal,
        PickUpOptions,
        SlideOptions,
        TwistOptions,
        HandOverOptions,
        MotionPolicy,
        RecoveryPolicy,
    )
    from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance

    device = engine.device
    tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
    goal_data = dict(span["goal"])
    kind = goal_data.pop("kind")
    skill = span["skill_id"]
    expected = {
        "pick_up": {"object_grasp", "pickup_motion"},
        "slide": {"slide_motion"},
        "twist": {"twist_motion"},
        "hand_over": {"handover_motion"},
    }
    if skill not in expected or kind not in expected[skill]:
        raise ValueError("Unsupported native goal/skill pair.")
    if kind == "object_grasp":
        if goal_data.keys() != {"object_id", "object_to_tcp"}:
            raise ValueError("Invalid object-bound grasp declaration.")
        obj = objects[goal_data["object_id"]]
        pose = obj.get_local_pose(to_matrix=True)[0]
        semantics = ObjectSemantics(
            entity_id=obj.uid,
            geometry={},
            affordance=AntipodalAffordance(
                mesh_vertices=obj.get_vertices(env_ids=[0], scale=True)[0],
                mesh_triangles=obj.get_triangles(env_ids=[0])[0],
            ),
        )
        goal = GraspGoal(
            semantics,
            grasp_xpos=pose @ tensor(goal_data["object_to_tcp"]),
            object_pose=pose,
        )
    else:
        cls = {
            "pickup_motion": PickUpMotionGoal,
            "slide_motion": SlideMotionGoal,
            "twist_motion": TwistMotionGoal,
            "handover_motion": HandOverMotionGoal,
        }[kind]
        if goal_data.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown native goal field.")
        goal = cls(
            **{
                k: (
                    None if v is None else tuple(v) if k == "axis_origin" else tensor(v)
                )
                for k, v in goal_data.items()
            }
        )
    cls = {
        "pick_up": PickUpOptions,
        "slide": SlideOptions,
        "twist": TwistOptions,
        "hand_over": HandOverOptions,
    }[skill]
    options = dict(span["options"])
    if options.keys() - {f.name for f in fields(cls)}:
        raise ValueError("Unknown native skill option.")
    for key in (
        "approach_direction",
        "obj_upright_direction",
        "grasp_frame_to_eef",
        "fixed_object_to_eef",
    ):
        if options.get(key) is not None:
            options[key] = tensor(options[key])
    if "downstream_object_target_poses" in options:
        options["downstream_object_target_poses"] = tuple(
            tensor(v) for v in options["downstream_object_target_poses"]
        )
    policy = dict(span["motion_policy"])
    policy.pop("control_dt", None)
    return engine.make_invocation(
        skill,
        goal,
        control_parts=span["control_parts"],
        skill_options=cls(**options),
        motion_policy=MotionPolicy(**policy),
        recovery_policy=RecoveryPolicy(max_replans=0, max_skill_retries=0),
    )


def _bound_command_path(
    planned: torch.Tensor,
    previous: torch.Tensor,
    profile: RobotProfile,
    participants: dict[str, dict[str, str]],
) -> torch.Tensor:
    """Keep unbound joints at their last command, including loaded grippers.

    Native plans carry measured positions for unbound joints. Feeding those
    back as targets would release a gripper's preload while another arm acts.
    """
    parts = {part for endpoints in participants.values() for part in endpoints.values()}
    names = {name for part in parts for name in profile.control_parts[part]}
    joint_ids = [i for i, name in enumerate(profile.joint_names) if name in names]
    command = previous.expand(planned.shape[0], -1).clone()
    command[:, joint_ids] = planned[:, joint_ids]
    return command


def execute_sequence(
    env: Any,
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
    plan_transform_factory: Callable[[int, Any], Any] | None = None,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    """Execute one object-grounded sequence with live physics and no replay PT."""
    from tensordict import TensorDict
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.compute.trajectory import retime_to_control_grid
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenCfg,
    )
    from embodichain.lab.sim.motion.planners import TrapezoidalPlannerCfg
    from embodichain.lab.sim.atomic_skills import (
        AtomicSkillEngine,
        ControlPartCommandProfile,
    )

    target = env.unwrapped
    path = manifest_path.resolve(strict=True)
    root = path.parent
    destination = root if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(path.read_text())
    allowed = {
        "schema",
        "execution",
        "task_config",
        "robot_profile",
        "layout",
        "joint_names",
        "step_dt",
        "timeline",
        "physical_success",
        "scene_collision",
        "initial_qpos",
        "connection",
        "files",
    }
    if data.keys() - allowed or data.get("schema") != "native-skill-sequence/v1":
        raise ValueError("Unsupported native skill sequence schema.")
    for filename, digest in data["files"].items():
        if file_digest(_contained(root, filename)) != digest:
            raise ValueError(f"Runtime bundle resource changed: {filename}")
    profile = RobotProfile.load(_contained(root, data["robot_profile"]))
    if target.num_envs != 1 or set(target.robot.joint_names) != set(
        profile.joint_names
    ):
        raise ValueError("Native fitted sequence requires its exact B=1 robot.")
    if target.robot.control_parts != profile.control_parts:
        raise ValueError("Robot control parts differ from the fitted profile.")
    dt = float(target.step_dt)
    if not math.isclose(dt, data["step_dt"], rel_tol=1e-6):
        raise ValueError("Runtime command clock differs from exported sequence.")
    connection = data["connection"]
    if (
        connection.keys() != {"kind", "minimum_duration_s"}
        or connection["kind"] != "quintic_joint"
    ):
        raise ValueError("Unsupported transition connection.")
    minimum = connection["minimum_duration_s"]
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("Connection duration must be positive and finite.")
    robot = target.robot
    order = [profile.joint_names.index(n) for n in robot.joint_names]
    inverse = [robot.joint_names.index(n) for n in profile.joint_names]
    active = [
        profile.joint_names.index(n) for n in target.expert_action_spec.joint_names
    ]
    tensor = lambda value: torch.as_tensor(
        value, dtype=torch.float32, device=robot.device
    )
    last = tensor(data["initial_qpos"])
    if last.shape != (len(profile.joint_names),) or not torch.isfinite(last).all():
        raise ValueError("Invalid sequence initial posture.")
    outcome = dict(
        schema="native-skill-sequence-execution/v1",
        completed=False,
        physical_success="unknown",
        unbound_joint_policy="hold_previous_command",
        step_dt=dt,
        spans=[],
        command_count=0,
    )
    previous_reset = getattr(target, "_replay_no_auto_reset", False)
    target._replay_no_auto_reset = True
    started = time.perf_counter()
    recorder = None

    def submit(q: torch.Tensor) -> None:
        nonlocal last
        if outcome["command_count"] >= 200_000:
            raise ValueError("Sequence exceeds execution command budget.")
        _, _, term, trunc, _ = env.step(
            ControllerAction(TensorDict({"qpos": q[active][None]}, batch_size=[1]))
        )
        if recorder is not None:
            recorder.append(q[active])
        last = q.clone()
        outcome["command_count"] += 1
        if bool(term.any() or trunc.any()):
            raise RuntimeError("Environment terminated before sequence completion.")

    try:
        env.reset(options={"save_data": False})
        robot.set_qpos(last[order][None])
        robot.set_qpos(last[order][None], target=True)
        robot.clear_dynamics()
        motion = MotionGenerator(
            MotionGenCfg(planner_cfg=TrapezoidalPlannerCfg(robot_uid=robot.uid))
        )
        commands = {
            part: ControlPartCommandProfile.joint_positions(
                **{k: tensor(v) for k, v in values.items()}
            )
            for part, values in profile.hand_commands.items()
        }
        engine = AtomicSkillEngine(motion, control_profiles=commands)
        objects = {
            obj["goal"]["object_id"]: target.sim.get_rigid_object(
                obj["goal"]["object_id"]
            )
            for obj in data["timeline"]
            if obj["kind"] == "atomic_skill" and obj["goal"]["kind"] == "object_grasp"
        }
        if record_trajectory:
            from ._sequence_trace import _SequenceTrace

            recorder = _SequenceTrace(
                destination / "trajectory.h5", target, profile, dt
            )
        for index, span in enumerate(data["timeline"]):
            print(
                f"Native sequence {index + 1}/{len(data['timeline'])}: {span['skill_id']}",
                flush=True,
            )
            record = dict(
                index=index,
                kind=span["kind"],
                skill_id=span["skill_id"],
                command_start=outcome["command_count"],
            )
            outcome["spans"].append(record)
            if span["kind"] == "atomic_skill":
                for obj in objects.values():
                    if obj is None:
                        raise ValueError(
                            "Native skill target is absent from the live scene."
                        )
                invocation = _invocation(engine, span, objects)
                context = engine.initial_context(control_dt=dt)
                transform = (
                    None
                    if plan_transform_factory is None
                    else plan_transform_factory(index, engine)
                )
                plan = engine.plan(invocation, context, plan_transform=transform)
                if not bool(plan.plan_success.all()) or plan.joint_trajectory is None:
                    raise RuntimeError(
                        f"Native {span['skill_id']} planning failed: {plan.diagnostics}"
                    )
                trajectory = plan.joint_trajectory
                scale = span["duration_scale"]
                if (
                    type(scale) not in (int, float)
                    or not math.isfinite(scale)
                    or not 1 <= scale <= 16
                ):
                    raise ValueError("Native duration_scale must be in [1, 16].")
                positions, _, _, _ = retime_to_control_grid(
                    trajectory.positions, trajectory.dt * scale, dt
                )
                qpos = _bound_command_path(
                    positions[0, :, inverse], last, profile, span["control_parts"]
                )
                identifier = span["goal"].get("object_id")
                if identifier:
                    record["object_id"] = identifier
                    record["object_start_z_m"] = float(
                        objects[identifier].get_local_pose()[0, 2]
                    )
                record["native_waypoints"] = len(qpos)
                record["expansion"] = plan.diagnostics.metadata.get("expansion")
            elif span["kind"] == "demo_skill":
                filename = _contained(root, span["tensor"])
                if file_digest(filename) != span["tensor_sha256"]:
                    raise ValueError("Demo tensor checksum mismatch.")
                source = torch.load(filename, map_location="cpu", weights_only=True)
                if (
                    source["joint_names"] != list(profile.joint_names)
                    or source["layout"] != "DOF x n_frame"
                ):
                    raise ValueError("Demo tensor joint layout differs from profile.")
                qpos = source["qpos"].T.to(device=robot.device, dtype=torch.float32)
                times = source["timestamps"].to(
                    device=robot.device, dtype=torch.float64
                )
                if (
                    qpos.ndim != 2
                    or qpos.shape != (len(times), len(profile.joint_names))
                    or len(times) == 0
                    or not torch.isfinite(qpos).all()
                    or not torch.isfinite(times).all()
                    or bool((torch.diff(times) <= 0).any())
                ):
                    raise ValueError("Invalid DemoSkill samples/clock.")
                if len(times) > 1:
                    intervals = torch.cat((times.new_zeros(1), times.diff()))
                    qpos = retime_to_control_grid(qpos[None], intervals[None], dt)[0][0]
                record["expansion"] = "excluded_demo_skill"
                record["demo_command_sha256"] = hashlib.sha256(
                    qpos.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
            else:
                raise ValueError("Unknown sequence span kind.")
            bridge = _connection(last, qpos[0], profile, dt, minimum)
            record["connection_waypoints"] = len(bridge)
            for q in bridge[1:]:
                submit(q)
            # The bridge endpoint is exactly the first path command.
            record["path_command_start"] = outcome["command_count"] - 1
            for q in qpos[1:]:
                submit(q)
            record["path_command_stop"] = outcome["command_count"]
            # Let the final command take effect and expose measured tracking error.
            for _ in range(10):
                submit(last)
            record["command_stop"] = outcome["command_count"]
            record["max_final_joint_error"] = float(
                (robot.get_qpos()[0, inverse] - last).abs().max()
            )
            if record.get("object_id"):
                record["object_end_z_m"] = float(
                    objects[record["object_id"]].get_local_pose()[0, 2]
                )
                record["observed_lift_m"] = (
                    record["object_end_z_m"] - record["object_start_z_m"]
                )
        outcome["completed"] = True
    except BaseException as error:
        outcome["error"] = str(error)
        raise
    finally:
        outcome["elapsed_seconds"] = time.perf_counter() - started
        try:
            if recorder is not None:
                recorder.close(completed=outcome["completed"])
            (destination / "execution.json").write_text(json.dumps(outcome, indent=2))
        finally:
            target._replay_no_auto_reset = previous_reset
    print(
        f"Native sequence completed; execution report: {destination / 'execution.json'}",
        flush=True,
    )
    return outcome
