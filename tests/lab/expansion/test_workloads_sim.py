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

            def activate(self, assignment, observation):
                row = assignment.env_id
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


@pytest.mark.requires_sim
def test_rolling_workloads_keep_native_receipts_and_world_boundaries(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "SDL_VIDEODRIVER": "offscreen", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]


if __name__ == "__main__":
    _probe(sys.argv[1])
