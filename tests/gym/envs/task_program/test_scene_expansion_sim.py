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

"""Native Task Program scene preparation, workspace, and restoration smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_RUN_CHILD = (
    "import os, runpy, sys; from pathlib import Path; "
    "module = runpy.run_path(sys.argv[1]); "
    "module['_run_scene_expansion_smoke'](Path(sys.argv[2])); "
    "sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
)


def _run_scene_expansion_smoke(output: Path) -> None:
    """Exercise real B=1 CPU physics with no renderer and measured placement."""
    import numpy as np
    import torch

    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
    from embodichain.lab.sim.motion.workspace.cfg import RobotWorkspaceCfg
    from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
    from embodichain.lab.task_program.integrations.scene_expansion import (
        execute_scene_variants,
    )
    from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
        SimulationSceneExpansionHost,
    )
    from embodichain.lab.sim.scene.workspace import (
        RobotSceneWorkspace,
    )
    from embodichain.utils.utility import load_config

    path = (
        _REPOSITORY_ROOT
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.seed = 0
    cfg.num_envs = cfg.sim_cfg.num_envs = 1
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sim_cfg.device = "cpu"
    cfg.light.direct = []
    cfg.light.indirect = None
    cfg.sensor = []
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    assert cfg.task_program is not None
    # One existing Pick/Place cycle is sufficient to verify this integration.
    cfg.task_program.program.count = 1
    env = None
    try:
        env = REGISTERED_ENVS["TaskProgramRepeatedPickPlace-v1"].make(cfg=cfg)
        env.reset(seed=0)
        cache = output.with_suffix(".npz")
        np.savez(
            cache,
            reachable_points=np.zeros((1, 3), dtype=np.float32),
            joint_configurations=env.robot.get_qpos(name="arm").cpu().numpy(),
        )
        workspace_cfg = RobotWorkspaceCfg(cache_path=str(cache))
        workspace_cfg.validate()
        env.robot.cfg.workspace_cfg = {"arm": workspace_cfg}
        workspace = RobotSceneWorkspace(env.robot, control_part="arm")
        samples = workspace.sample_object_poses(
            "cube", torch.eye(4), num_samples=1, seed=0
        )
        assert len(samples) == 1
        workspace_check = workspace.check_object_pose(
            samples[0].pose_change, torch.eye(4), joint_seed=samples[0].joint_seed
        )
        assert workspace_check.passed

        cube = env.sim.get_rigid_object("cube")
        proposed = cube.get_local_pose(to_matrix=True)[0].clone()
        proposed[0, 3] += 0.015
        variant = SceneVariant(
            "repeated-pick-place:scene-expansion-smoke",
            0,
            0,
            (ScenePoseChange("cube", proposed.tolist()),),
        )

        def initial_checks() -> ValidationResult:
            actual = cube.get_local_pose(to_matrix=True)[0]
            valid = (
                bool(torch.isfinite(actual).all())
                and float(torch.linalg.vector_norm(actual[:2, 3] - proposed[:2, 3]))
                < 0.01
            )
            return ValidationResult(
                (ValidationCheck("initial.layout", "passed" if valid else "failed"),)
            )

        host = SimulationSceneExpansionHost(
            env, initial_validator=lambda row: initial_checks(), settle_steps=30
        )

        def measured_placement() -> ValidationResult:
            actual = cube.get_local_pose(to_matrix=True)[0]
            target_xy = torch.tensor([-0.40, 0.48])
            xy_error = float(torch.linalg.vector_norm(actual[:2, 3].cpu() - target_xy))
            height = float(actual[2, 3])
            passed = xy_error < 0.06 and abs(height - 0.025) < 0.04
            return ValidationResult(
                (
                    ValidationCheck(
                        "placement.measured",
                        "passed" if passed else "failed",
                        metrics={"xy_error": xy_error, "height": height},
                    ),
                )
            )

        result = execute_scene_variants(
            env,
            {0: variant},
            prepare_scene=host,
            measured_validator=lambda row: measured_placement(),
            trace_actions=True,
        )[0]
        execution_trace = env._active_task_program_bridge.execution_trace
        assert len(execution_trace) == int(env._traj_steps[0])
        assert all(row["step"] == index for index, row in enumerate(execution_trace))
        assert all(
            row["plan_index"] >= 0
            for row in execution_trace
            if row["kind"] == "runtime_command"
        )
        snapshot = host.initial_states[0]
        assert snapshot is not None
        recorded_first_pose = env._traj_buffer[
            "states", "rigid_objects", "cube", "pose"
        ][0, 0].cpu()
        captured_cube = next(
            entity
            for entity in snapshot.to_metadata()["entities"]
            if entity["uid"] == "cube"
        )
        assert torch.allclose(
            recorded_first_pose, torch.tensor(captured_cube["pose"]), atol=1e-6, rtol=0
        )
        restored = host.restore_initial_states({0: snapshot})[0]
        assert int(env._traj_steps[0]) == 0
        assert (
            restored.metadata["restored_from_initial_state_id"]
            == snapshot.initial_state_id
        )
        report = {
            "scene": result.to_metadata(),
            "execution": result.execution.to_metadata(),
            "workspace_accepted": workspace_check.passed,
            "restoration_accepted": restored.validation.accepted,
            "first_frame_matches_initial_state": True,
            "recorded_steps_after_restore": int(env._traj_steps[0]),
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert result.accepted, report
        assert restored.validation.accepted, report
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _run_batch_scene_expansion_smoke(
    backend: str, env_ids: tuple[int, ...], capacity: int
) -> None:
    """Prepare, restore and execute two native rows with independent acceptance."""
    import torch

    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
    from embodichain.lab.sim.scene.expansion import SceneVariant
    from embodichain.lab.task_program.integrations.scene_expansion import (
        execute_scene_variants,
    )
    from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
        SimulationSceneExpansionHost,
    )
    from embodichain.utils.utility import load_config

    path = _REPOSITORY_ROOT / f"examples/expansion/engine/task.{backend}.yaml"
    raw = load_config(path)
    cfg = config_to_cfg(raw, source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = capacity
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    if backend == "default":
        cfg.sim_cfg.device = "cpu"
    cfg.task_program.program.count = 1
    cfg.sensor = []
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    cfg.light.direct = []
    cfg.light.indirect = None
    env = None
    try:
        env = REGISTERED_ENVS[raw["id"]].make(cfg=cfg)
        env.reset(seed=0)
        variants = {row: SceneVariant("native-batch", 7, row, ()) for row in env_ids}

        def initial(row: int) -> ValidationResult:
            finite = bool(torch.isfinite(env.robot.get_qpos()[row]).all())
            return ValidationResult(
                (ValidationCheck("initial.finite", "passed" if finite else "failed"),)
            )

        host = SimulationSceneExpansionHost(
            env, env_ids=env_ids, initial_validator=initial
        )
        prepared = host(variants)
        assert all(value.validation.accepted for value in prepared.values())
        snapshots = host.initial_states
        clock = env.sim.simulation_time
        cube = env.sim.get_rigid_object("cube")
        cube.set_velocity(
            lin_vel=torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], device=env.device),
            ang_vel=torch.zeros((2, 3), device=env.device),
            env_ids=list(env_ids),
        )
        restored = host.restore_initial_states(snapshots)
        assert all(value.validation.accepted for value in restored.values())
        assert env.sim.simulation_time == clock

        def measured(row: int) -> ValidationResult:
            accepted = bool(env.is_task_success()[row]) and row == env_ids[-1]
            return ValidationResult(
                (ValidationCheck("measured.row", "passed" if accepted else "failed"),)
            )

        result = execute_scene_variants(
            env,
            variants,
            prepare_scene=lambda rows: restored,
            measured_validator=measured,
        )
        assert all(value.program_success for value in result.values())
        assert result[env_ids[-1]].accepted and not result[env_ids[0]].accepted
        assert env.episode_context.filter_recordable(env_ids).tolist() == [env_ids[-1]]
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "backend,env_ids,capacity",
    [
        ("default", (1, 0), 2),
        pytest.param("newton", (1, 0), 2, marks=pytest.mark.gpu),
        ("default", (3, 1), 4),
    ],
)
def test_native_batch_scene_expansion_prepares_restores_and_isolates_rows(
    backend: str, env_ids: tuple[int, ...], capacity: int
) -> None:
    command = "import os,runpy,sys; runpy.run_path(sys.argv[1])['_run_batch_scene_expansion_smoke'](sys.argv[2], tuple(map(int, sys.argv[3].split(','))), int(sys.argv[4])); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            __file__,
            backend,
            ",".join(map(str, env_ids)),
            str(capacity),
        ],
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.slow
def test_native_scene_expansion_records_and_restores_measured_task(
    tmp_path: Path,
) -> None:
    """Qualify the production host and Robot FK/IK on a real rigid scene."""
    report_path = tmp_path / "scene-expansion.json"
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        (str(_REPOSITORY_ROOT), child_env.get("PYTHONPATH", ""))
    )
    child_env["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", _RUN_CHILD, __file__, str(report_path)],
        cwd=_REPOSITORY_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scene"]["accepted"] is True
    assert report["scene"]["program_success"] is True
    assert report["execution"]["length"] > 0
    assert report["workspace_accepted"] is True
    assert report["restoration_accepted"] is True
    assert report["first_frame_matches_initial_state"] is True
    assert report["recorded_steps_after_restore"] == 0
