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

"""Measured asset and language switching through the shared Pick/Place expert."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_CONFIG = _ROOT / (
    "extensions/embodichain_tasks/configs/tasks/manipulation/"
    "repeated_pick_place/task.dual_ur5.assets.yaml"
)


def _run_asset_episode(asset: str, output: str) -> None:
    """Run native resources in a disposable process, retaining measured evidence."""
    import torch

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import make
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    env = None
    exit_code = 1
    try:
        payload = load_config(_CONFIG)
        payload["env"]["events"]["select_object"]["params"]["asset"] = asset
        payload.update(
            num_envs=1,
            headless=True,
            renderer="no-render",
            device="cpu",
            enable_sensor=False,
            seed=0,
        )
        cfg = config_to_cfg(payload, source_path=_CONFIG)
        cfg.dataset = None
        cfg.filter_dataset_saving = True
        cfg.init_rollout_buffer = False
        cfg.record_trajectory = True
        cfg.trajectory_auto_save = False
        env = make(payload["id"], cfg=cfg).unwrapped
        env.reset(seed=0)
        result = execute_demo_episode(env)
        obj = env.sim.get_rigid_object("cube")
        report = {
            "episode": result.to_metadata(),
            "recording": env._demo_episode_metadata[0],
            "position": obj.get_local_pose(to_matrix=True)[0, :3, 3].tolist(),
            "linear_speed": torch.linalg.vector_norm(obj.body_data.lin_vel[0]).item(),
            "shape": type(obj.cfg.shape).__name__,
            "source_instruction": cfg.task_program.instruction,
        }
        Path(output).write_text(json.dumps(report), encoding="utf-8")
        exit_code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "asset,label,shape",
    [
        ("cube", "cube", "CubeCfg"),
        ("rubiks_cube", "Rubik's cube", "MeshCfg"),
    ],
)
def test_asset_switch_preserves_physical_goal_and_recorded_language(
    asset: str,
    label: str,
    shape: str,
    tmp_path: Path,
) -> None:
    """Both choices reach the same measured goal with matching task/segment text."""
    command = (
        "import runpy, sys; module = runpy.run_path(sys.argv[1]); "
        "module['_run_asset_episode'](sys.argv[2], sys.argv[3])"
    )
    output = tmp_path / "outcome.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(Path(__file__).resolve()),
            asset,
            str(output),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert report["shape"] == shape
    assert report["episode"]["completed"]
    assert report["episode"]["success"] == [True]
    assert math.dist(report["position"], [0.55, 0.05, 0.775]) < 0.04
    assert report["linear_speed"] < 0.03
    recording = report["recording"]
    assert recording["assets"]["cube"]["asset"] == asset
    assert recording["assets"]["cube"]["label"] == label
    assert (
        label in recording["instruction"] and "{cube}" not in recording["instruction"]
    )
    assert len(recording["segments"]) == 6
    for segment in recording["segments"]:
        assert (
            label in segment["instruction"] and "{cube}" not in segment["instruction"]
        )
    assert "{cube}" in report["source_instruction"]
