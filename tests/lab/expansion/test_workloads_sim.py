# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _probe(output: str) -> None:
    import hashlib
    import json
    import torch
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        ExpansionEngine,
        ExpansionJobCfg,
        ExpansionSession,
        ExpansionSessionCfg,
        EpisodeCandidate,
        PreparedEpisodeWorkItem,
        ExpansionWorkloadCfg,
        WorkloadExecutionCfg,
        WorkloadProgramExecutor,
        load_expansion_episode,
    )
    from embodichain.lab.expansion.contracts import (
        ExpertEpisode,
        SceneCase,
        ValidationCheck,
        ValidationResult,
    )
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    discover_task_packages()
    execute_init_hooks()
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 4
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sensor, cfg.enable_sensor, cfg.dataset = [], False, None
    cfg.init_rollout_buffer = cfg.record_trajectory = False
    cfg.task_program.program.count = 1
    env = None
    try:
        env = REGISTERED_ENVS[load_config(path)["id"]].make(cfg=cfg)
        env.reset(seed=0)
        robot = env.robot
        program = env.compile_task_program(cfg.task_program)
        from embodichain.lab.sim.motion.motion_generator import (
            MotionGenerator,
            MotionGenCfg,
            MotionGenOptions,
        )
        from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
            TrapezoidalPlannerCfg,
        )
        from embodichain.lab.sim.motion.planners.utils import PlanState
        from embodichain.lab.expansion._planning_batches import batched_planning

        requests = {}
        for row in (0, 3):
            generator = MotionGenerator(
                MotionGenCfg(
                    planner_cfg=TrapezoidalPlannerCfg(
                        robot_uid=robot.uid, sim_instance_id=env.sim.instance_id
                    )
                ),
                env_ids=(row,),
            )
            start = robot.get_qpos(name="arm")[row : row + 1].clone()
            goal = start.clone()
            goal[:, 0] += 0.01 * (row + 1)
            states = [PlanState.from_qpos(goal)]
            options = MotionGenOptions(
                strategy="ik_interp",
                interpolation_mode="fixed_grid",
                ik_seed_mode="sequential",
                sample_count=12,
                interpolation_dt=0.04,
                start_qpos=start,
                control_part="arm",
            )
            requests[row] = (generator, states, options)
        reference = {
            row: generator.generate(states, options)
            for row, (generator, states, options) in requests.items()
        }

        class Request:
            def __init__(self, row):
                self.row = row

            def next(self):
                generator, states, options = requests[self.row]
                return generator.generate(states, options)

        before = env.sim.physics_step_index
        with batched_planning(robot, 0) as planning:
            planned = planning.gather({row: Request(row) for row in requests})
            assert (
                planning.stats["motion_calls"] == 1
                and planning.stats["max_motion_group"] == 2
            ), planning.stats
            for row, original in reference.items():
                torch.testing.assert_close(
                    planned[row].success, original.success, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    planned[row].positions, original.positions, rtol=0, atol=1e-6
                )
                torch.testing.assert_close(planned[row].dt, original.dt, rtol=0, atol=0)
        assert env.sim.physics_step_index == before
        from embodichain.lab.expansion.collision import _MeasuredCollision

        checks = _MeasuredCollision(
            env,
            str(root / "examples/expansion/scene_robot_feasibility/search.ur5.yaml"),
            Path(output) / "checks",
        )

        class Host:
            capacity = env.num_envs

            def preflight(self, items):
                assert all(item.compatibility_key == "one-world" for item in items)

            def validate_plan(self, item):
                assert torch.isfinite(robot.get_qpos()[item.env_id]).all()
                return ValidationResult(
                    (
                        ValidationCheck(
                            "initial_geometry",
                            "passed" if checks.initial[item.env_id] else "failed",
                        ),
                    )
                )

        sink = ArtifactEpisodeSink(output)
        session = ExpansionSession(
            ExpansionSessionCfg.from_mapping(
                {
                    "collection": {
                        "target_committed_episodes": 4,
                        "max_proposals": 8,
                        "max_rollout_attempts": 8,
                    },
                    "execution": {"ready_high_watermark": 4, "ready_low_watermark": 0},
                    "persistence": {
                        "pending_episode_limit": 4,
                        "pending_max_bytes": 32 * 1024 * 1024,
                    },
                }
            )
        )
        engine = ExpansionEngine(
            ExpansionJobCfg(batch_size=4), session=session, host=Host(), sink=sink
        )

        class Source:
            def __init__(self):
                self.buffers, self.assignments, self.adapters = {}, {}, []
                self.finished = []

            def prepare(self, assignments):
                assert len(SimulationManager._instances) == 1
                self.reset_holds = getattr(self, "reset_holds", {})
                for assignment in assignments:
                    self.reset_holds[assignment.env_id] = robot.get_qpos(target=True)[
                        assignment.env_id
                    ].clone()

            def activate(self, assignment, observation):
                row = assignment.env_id
                # The physical settling ticks must keep the authored drive
                # target, rather than repeatedly holding gravity-displaced qpos.
                torch.testing.assert_close(
                    robot.get_qpos(target=True)[row],
                    self.reset_holds[row],
                    rtol=0,
                    atol=0,
                )
                checks.prepare(env_ids=(row,))
                from copy import deepcopy

                self.scalar_monitors = getattr(self, "scalar_monitors", {})
                self.scalar_monitors[row] = deepcopy(checks.monitors[row])
                key = f"{assignment.workload_id}-{assignment.ordinal}"
                state = robot.get_qpos()[row].detach().cpu().clone()
                initial = hashlib.sha256(state.numpy().tobytes()).hexdigest()
                session.register_case(
                    SceneCase(key, initial, "scene", program.program_id, robot.uid),
                    robot.get_qpos_limits()[row].cpu(),
                    joint_names=tuple(robot.joint_names),
                )
                identity, _ = session.propose(
                    key,
                    initial,
                    source_id=program.program_id,
                    source_revision="v1",
                    template_id=key,
                    operator_id="episode",
                )
                item = PreparedEpisodeWorkItem(
                    EpisodeCandidate(identity, "{}"), "one-world", row
                )
                native = engine.admit_row(item)
                assert native is not None
                self.assignments[row] = native
                self.buffers[row] = {"actions": [], "states": [state]}
                adapter = env.task_program_adapter.for_env_ids((row,))
                self.adapters.append(adapter)
                return adapter.create_bridge(program, isolate_policy_sessions=True)

            def started(self, assignment, boundary):
                engine.mark_row_started(self.assignments[assignment.env_id])

            def observe(self, assignments, action, observation):
                rejected = []
                checks.sample(tuple(a.env_id for a in assignments), action)
                for assignment in assignments:
                    scalar = self.scalar_monitors[assignment.env_id]
                    scalar.observe(env, action)
                    assert scalar.validation(env) == checks.terminal[assignment.env_id]
                for assignment in assignments:
                    row = assignment.env_id
                    buffer = self.buffers[row]
                    values = (
                        action.value
                        if isinstance(action.value, torch.Tensor)
                        else action.value["qpos"]
                    )
                    buffer["actions"].append(values[row].detach().cpu().clone())
                    buffer["states"].append(
                        robot.get_qpos()[row].detach().cpu().clone()
                    )
                    if (
                        assignment.workload_id == "first"
                        and assignment.ordinal == 0
                        and len(buffer["actions"]) == 12
                    ):
                        rejected.append(row)
                return rejected

            def complete(self, result):
                row = result.assignment.env_id
                native = self.assignments.pop(row)
                data = self.buffers.pop(row)
                count = result.program.lengths[0]
                assert (
                    len(data["actions"]) == count and len(data["states"]) == count + 1
                )
                assert (
                    result.world_control_end - result.world_control_start + 1 == count
                )
                self.finished.append(
                    (
                        result.assignment.workload_id,
                        result.assignment.ordinal,
                        count,
                        result.program.accepted[0],
                    )
                )
                if not result.program.accepted[0]:
                    return engine.complete_row(native, None)
                episode = ExpertEpisode(
                    native.item.candidate.identity,
                    {"joint_positions": torch.stack(data["states"])},
                    torch.stack(data["actions"]),
                    torch.arange(count + 1, dtype=torch.float64) * env.step_dt,
                    "qpos",
                    ValidationResult(
                        (
                            ValidationCheck("task_success", "passed"),
                            ValidationCheck(
                                "path_collision", checks.results[row].status
                            ),
                            *checks.terminal[row].checks,
                        )
                    ),
                    native.episode_id,
                    native.commit_id,
                )
                return engine.complete_row(native, episode)

        source = Source()
        owner = id(env.sim.get_world())
        done = []
        executor = WorkloadProgramExecutor(
            env,
            [
                ExpansionWorkloadCfg("first", 2, 4),
                ExpansionWorkloadCfg("second", 1, 2),
                ExpansionWorkloadCfg("queued", 1, 2),
            ],
            source,
            WorkloadExecutionCfg(
                workload_count=2, batch_size=2, max_steps=2500, settling_steps=5
            ),
            on_workload_done=lambda name, counts: done.append((name, counts)),
        )
        result = executor.run()
        assert result["complete"], (result, source.finished)
        assert result["num_envs"] == 4 and result["refills"] > 0
        assert result["counts"]["first"] == {
            "attempts": 3,
            "rollouts": 3,
            "committed": 2,
        }
        assert len(done) == 3 and len(sink.inspect()) == 4
        assert session.snapshot()["counts"]["committed"] == 4
        assert (
            id(env.sim.get_world()) == owner and len(SimulationManager._instances) == 1
        )
        assert next(entry for entry in source.finished if entry[:2] == ("first", 0))[
            2:
        ] == (13, False)
        for receipt in sink.inspect():
            episode = load_expansion_episode(receipt.storage_id)
            assert len(episode.timestamps) == len(episode.actions) + 1
            assert episode.validation.accepted
        with pytest.raises(RuntimeError, match="single-use"):
            executor.run()
        for adapter in source.adapters:
            adapter.close()
        checks.close()
        (Path(output) / "workload-result.json").write_text(json.dumps(result, indent=2))
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _facade_probe(output: str) -> None:
    from copy import deepcopy
    import hashlib
    import json
    import torch
    import yaml
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        ExpansionRunCfg,
        GymExpansionHostExtensions,
        GymExpansionWorkload,
        WorkloadExecutionCfg,
        run_gym_expansion_workloads,
        load_expansion_episode,
    )
    from embodichain.lab.expansion.collision import _MeasuredCollision
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    task = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place"
    )
    deployment = task / "task.ur5.yaml"
    path = Path(output)
    path.mkdir(exist_ok=True)
    program = yaml.safe_load((task / "task_program/program.yaml").read_text())
    program["program"]["repeat"]["count"] = 1
    override = path / "program.yaml"
    override.write_text(yaml.safe_dump(program))

    def absolute_paths(value, base):
        if isinstance(value, dict):
            return {key: absolute_paths(item, base) for key, item in value.items()}
        if isinstance(value, list):
            return [absolute_paths(item, base) for item in value]
        if isinstance(value, str) and value.endswith((".yaml", ".yml", ".json")):
            return str((base / value).resolve())
        return value

    raw = absolute_paths(yaml.safe_load(deployment.read_text()), deployment.parent)
    component = Path(raw["embodiment"]["component"])
    embodiment = absolute_paths(yaml.safe_load(component.read_text()), component.parent)
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    base_cfg = config_to_cfg(
        yaml.safe_load(deployment.read_text()), source_path=deployment
    )
    contact = deepcopy(
        base_cfg.robot.gripper_bindings["eef"]["defaults"]["root_from_contact_frame"]
    )
    contact[0][3] += 0.001
    embodiment["grippers"]["eef"]["root_from_contact_frame"] = contact
    component_override = path / "embodiment.yaml"
    component_override.write_text(yaml.safe_dump(embodiment))
    raw["id"] = "TaskProgramRepeatedPickPlace-ContactFrame-v1"
    raw["embodiment"]["component"] = str(component_override)
    policy_deployment = path / "contact-frame.yaml"
    policy_deployment.write_text(yaml.safe_dump(raw))
    config_to_cfg(raw, source_path=policy_deployment)
    discover_task_packages()
    execute_init_hooks()
    workloads = []
    for name, target in (("pair", 2), ("queued", 1)):
        selected_deployment = deployment if name == "pair" else policy_deployment
        digest = hashlib.sha256(selected_deployment.read_bytes()).hexdigest()
        cfg = ExpansionRunCfg(
            deployment=str(selected_deployment),
            program=str(override),
            output_dir=str(path / name),
            target_episodes=target,
            max_attempts=4,
            max_steps=2500,
            max_wall_time_s=240,
            settling_steps=5,
        )
        report = {
            "job_id": name,
            "semantic_id": name,
            "execution_id": name,
            "recipe_revision": name,
            "input_files": {"deployment": digest},
            "runtime_versions": {},
        }
        workloads.append(GymExpansionWorkload(name, cfg, report, {}))

    observed_bindings = []

    def validator(env, output):
        binding = deepcopy(env.robot.cfg.gripper_bindings)
        observed_bindings.append((env.robot.cfg, binding))
        factory = REGISTERED_ENVS[raw["id"]].task_program_adapter_factory
        with pytest.raises(ValueError, match="actual hand installation"):
            factory.create_adapter(env)
        assert env.robot.cfg.gripper_bindings == binding
        return _MeasuredCollision(
            env,
            str(root / "examples/expansion/scene_robot_feasibility/search.ur5.yaml"),
            output,
        )

    result = run_gym_expansion_workloads(
        workloads,
        WorkloadExecutionCfg(
            workload_count=1, batch_size=2, max_steps=2500, settling_steps=5
        ),
        extensions=GymExpansionHostExtensions(validator_factory=validator),
        output_dir=path / "execution",
    )
    assert result["complete"], result
    assert result["num_envs"] == 2 and result["world"]["world_count"] == 1
    assert not SimulationManager._instances
    assert all(cfg.gripper_bindings == binding for cfg, binding in observed_bindings)
    for name, expected in (("pair", 2), ("queued", 1)):
        group = result["workloads"][name]
        assert group["complete"] and group["committed_episodes"] == expected
        receipts = ArtifactEpisodeSink(path / name).inspect()
        assert len(receipts) == len(group["lerobot"]["datasets"]) == expected
        for receipt in receipts:
            episode = load_expansion_episode(receipt.storage_id)
            assert episode.validation.accepted
            assert len(episode.timestamps) == len(episode.actions) + 1
            execution = episode.metadata["execution"]
            assert execution["world_control_end"] - execution[
                "world_control_start"
            ] + 1 == len(episode.actions)
            assert execution["world"] == result["world"]
    (path / "facade-result.json").write_text(json.dumps(result, indent=2))
    # A real drive change remains incompatible even though contact conventions
    # above are allowed. Reject before constructing another World.
    embodiment["simulation"]["joint_drive_props"]["stiffness"][
        "gripper_finger1_joint_1"
    ] += 1
    hardware_component = path / "different-hardware.yaml"
    hardware_component.write_text(yaml.safe_dump(embodiment))
    raw["id"] += "-Hardware"
    raw["embodiment"]["component"] = str(hardware_component)
    hardware_deployment = path / "different-hardware-task.yaml"
    hardware_deployment.write_text(yaml.safe_dump(raw))
    rejected = GymExpansionWorkload(
        "different",
        ExpansionRunCfg(
            **{
                **workloads[-1].cfg.to_dict(),
                "deployment": str(hardware_deployment),
                "output_dir": str(path / "different"),
            }
        ),
        {
            **workloads[-1].report,
            "input_files": {
                "deployment": hashlib.sha256(
                    hardware_deployment.read_bytes()
                ).hexdigest()
            },
        },
        {},
    )
    with pytest.raises(ValueError, match="physical environment"):
        run_gym_expansion_workloads(
            [workloads[0], rejected],
            WorkloadExecutionCfg(
                workload_count=1, batch_size=2, max_steps=2500, settling_steps=5
            ),
            output_dir=path / "rejected",
        )
    assert not SimulationManager._instances

    # Expiry during settling retires only the unstarted group. A peer must
    # still collect/export a real receipt, and the aggregate stays incomplete.
    from dataclasses import replace

    budget_groups = [
        replace(
            workloads[0],
            workload_id=name,
            cfg=ExpansionRunCfg(
                **{
                    **workloads[0].cfg.to_dict(),
                    "output_dir": str(path / name),
                    "target_episodes": 1,
                    "max_attempts": 1,
                    "max_wall_time_s": budget,
                }
            ),
        )
        for name, budget in (("expired", 1e-9), ("peer", 240.0))
    ]
    budget_result = run_gym_expansion_workloads(
        budget_groups,
        WorkloadExecutionCfg(
            workload_count=2, batch_size=1, max_steps=2500, settling_steps=5
        ),
        extensions=GymExpansionHostExtensions(
            validator_factory=lambda env, output: _MeasuredCollision(
                env,
                str(
                    root / "examples/expansion/scene_robot_feasibility/search.ur5.yaml"
                ),
                output,
            )
        ),
        output_dir=path / "budget-execution",
    )
    assert not budget_result["complete"]
    assert budget_result["counts"]["expired"] == {
        "attempts": 1,
        "rollouts": 0,
        "committed": 0,
    }
    assert budget_result["workloads"]["peer"]["complete"]
    assert len(ArtifactEpisodeSink(path / "peer").inspect()) == 1
    assert len(budget_result["workloads"]["peer"]["lerobot"]["datasets"]) == 1
    assert not ArtifactEpisodeSink(path / "expired").inspect()
    records = list((path / "expired" / "attempts").glob("*/attempt.json"))
    assert len(records) == 1
    expired_record = json.loads(records[0].read_text())
    assert expired_record["status"] == "initial_rejected"
    assert expired_record["reason"] == "wall_time_exhausted"
    assert not SimulationManager._instances


def _reuse_probe(output: str) -> None:
    """A real rejected installation opens reuse, then only one receipt wins."""
    import json
    import numpy as np
    import torch
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        ExpansionRunCfg,
        GymExpansionHostExtensions,
        GymExpansionWorkload,
        WorkloadExecutionCfg,
        run_gym_expansion_workloads,
        load_expansion_episode,
    )
    from embodichain.lab.expansion.collision import _MeasuredCollision
    from embodichain.lab.sim.scene.expansion import SceneVariant
    from embodichain.lab.sim.scene.feasibility import (
        JointSceneCandidate,
        RobotDeploymentCandidate,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.expansion._planning_batches import PlanningBatches
    from embodichain.utils.utility import load_config
    import yaml

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    task = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place"
    )
    path = Path(output)
    path.mkdir(exist_ok=True)
    raw = load_config(task / "task.ur5.yaml")

    def absolute(value, base):
        if isinstance(value, dict):
            return {k: absolute(v, base) for k, v in value.items()}
        if isinstance(value, list):
            return [absolute(v, base) for v in value]
        if isinstance(value, str) and value.endswith((".yaml", ".yml", ".json")):
            return str((base / value).resolve())
        return value

    raw = absolute(raw, task)
    program = yaml.safe_load((task / "task_program/program.yaml").read_text())
    program["program"]["repeat"]["count"] = 1
    program_path = path / "program.yaml"
    program_path.write_text(yaml.safe_dump(program))
    raw["task_program"]["program"] = str(program_path)
    deployment = path / "deployment.yaml"
    deployment.write_text(yaml.safe_dump(raw))

    def scenes(env):
        nominal = env.robot.get_local_pose(to_matrix=True)[0].cpu().tolist()

        class Source:
            def candidate(self, ordinal):
                pose = np.array(nominal)
                if ordinal == 0:
                    pose[2, 3] -= 0.6
                elif ordinal == 2:
                    pose[0, 3] += 0.02
                return JointSceneCandidate(
                    "retry",
                    SceneVariant("retry", 2, ordinal, ()),
                    RobotDeploymentCandidate({env.robot.uid: pose.tolist()}),
                    "installation",
                    (),
                )

        return Source()

    cfg = ExpansionRunCfg(
        deployment=str(deployment),
        output_dir=str(path / "group"),
        batch_size=1,
        target_episodes=1,
        max_attempts=6,
        max_steps=2500,
        max_wall_time_s=180,
        settling_steps=5,
        seed=2,
    )
    workload = GymExpansionWorkload(
        "retry",
        cfg,
        {
            "job_id": "retry",
            "semantic_id": "retry",
            "execution_id": "retry",
            "recipe_revision": "v1",
            "input_files": {},
            "runtime_versions": {},
        },
        {},
        GymExpansionHostExtensions(scene_source_factory=scenes),
    )
    original_gather = PlanningBatches.gather
    boundary = {"injected": False, "executed": False}

    def gather(broker, cursors):
        actions = original_gather(broker, cursors)
        if not boundary["injected"] and any(
            c.finished and c.accepted for c in cursors.values()
        ):
            for row, cursor in cursors.items():
                if row not in actions or cursor.finished:
                    continue
                # Exercise the real host-stop path before quota completion;
                # the controlled administrative stop is not fake physics.
                cursor.abort("test_stop_before_peer_quota")
                hold = cursor.next()
                assert hold is not None
                actions[row] = hold
                boundary.update(
                    injected=True,
                    row=row,
                    cursor=cursor,
                    clock=cursor.bridge.clock.step_index,
                    hold=hold,
                )
                break
        return actions

    def validator(env, output):
        original_step = env.step

        def step(action, *args, **kwargs):
            pending = boundary["injected"] and not boundary["executed"]
            if pending:
                assert boundary["cursor"].bridge.clock.step_index == boundary["clock"]
                actual = (
                    action.value
                    if isinstance(action.value, torch.Tensor)
                    else action.value["qpos"]
                )
                expected = boundary["hold"].value
                expected = (
                    expected if isinstance(expected, torch.Tensor) else expected["qpos"]
                )
                torch.testing.assert_close(
                    actual[boundary["row"]], expected[0], rtol=0, atol=0
                )
            result = original_step(action, *args, **kwargs)
            if pending:
                boundary["executed"] = True
            return result

        env.step = step
        return _MeasuredCollision(
            env,
            str(root / "examples/expansion/scene_robot_feasibility/search.ur5.yaml"),
            output,
        )

    PlanningBatches.gather = gather
    try:
        result = run_gym_expansion_workloads(
            (workload,),
            WorkloadExecutionCfg(
                workload_count=1,
                batch_size=2,
                max_steps=2500,
                settling_steps=5,
                seed=2,
                reuse_idle_rows=True,
            ),
            extensions=GymExpansionHostExtensions(validator_factory=validator),
            output_dir=path / "execution",
        )
    finally:
        PlanningBatches.gather = original_gather
    assert boundary["injected"] and boundary["executed"]
    assert result["complete"] and result["reused_rows"] > 0, result
    assert result["counts"]["retry"]["committed"] == 1
    assert result["cancelled"]["retry"] > 0
    assert result["num_envs"] == 2 and result["world"]["world_count"] == 1
    receipts = ArtifactEpisodeSink(path / "group").inspect()
    assert len(receipts) == 1
    episode = load_expansion_episode(receipts[0].storage_id)
    assert (
        episode.validation.accepted
        and len(episode.timestamps) == len(episode.actions) + 1
    )
    records = [
        json.loads(p.read_text())
        for p in (path / "group/attempts").glob("*/attempt.json")
    ]
    assert next(r for r in records if r["ordinal"] == 0)["status"] in {
        "initial_rejected",
        "rejected",
    }
    cancelled = [
        p
        for p in (path / "group/attempts").glob("*/attempt.json")
        if json.loads(p.read_text())["status"] == "cancelled"
    ]
    assert cancelled
    for entry in cancelled:
        record = json.loads(entry.read_text())
        assert record["reason"] == "quota_fulfilled_by_peer"
        if "partial_rollout" in record:
            assert record["env_id"] in (0, 1)
            if record["steps"]:
                assert record["last_controller_metadata"].get("bridge_action_kind")
            with np.load(entry.parent / record["partial_rollout"]["path"]) as data:
                assert len(data["qpos"]) == record["steps"] + 1
                assert len(data.get("actions", [])) == record["steps"]
    assert not SimulationManager._instances


def _declarative_probe(output: str) -> None:
    import json
    from embodichain.lab.expansion import (
        run_task_program_expansion,
        ArtifactEpisodeSink,
        load_expansion_episode,
    )
    from embodichain.lab.sim import SimulationManager

    root = Path(__file__).resolve().parents[3]
    result = run_task_program_expansion(
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml",
        Path(__file__).parent / "fixtures/task_program_job.yaml",
        output,
    )
    assert result["complete"], result
    execution = json.loads((Path(output) / "workload-execution.json").read_text())
    assert execution["world"]["world_count"] == 1 and execution["num_envs"] == 2
    assert not SimulationManager._instances
    for operator, group in result["operators"].items():
        receipts = ArtifactEpisodeSink(group["directory"]).inspect()
        assert len(receipts) == len(group["lerobot"]["datasets"]) == 1
        episode = load_expansion_episode(receipts[0].storage_id)
        assert episode.validation.accepted
        assert len(episode.timestamps) == len(episode.actions) + 1
        if operator == "joint_residual":
            assert episode.metadata["expansion"]


@pytest.mark.requires_sim
@pytest.mark.parametrize("mode", ["executor", "facade", "declarative", "reuse"])
def test_rolling_workloads_keep_native_receipts_and_world_boundaries(tmp_path, mode):
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path), mode],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "SDL_VIDEODRIVER": "offscreen", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]


if __name__ == "__main__":
    {
        "executor": _probe,
        "facade": _facade_probe,
        "declarative": _declarative_probe,
        "reuse": _reuse_probe,
    }[sys.argv[2] if len(sys.argv) > 2 else "executor"](sys.argv[1])
