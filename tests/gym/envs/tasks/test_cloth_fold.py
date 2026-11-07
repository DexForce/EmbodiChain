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


"""Physical acceptance and installed-path registration for cloth folding."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("config_first", (True, False))
def test_cloth_fold_registration_order(tmp_path: Path, config_first: bool) -> None:
    package = (
        Path(__file__).resolve().parents[4] / "extensions/embodichain_tasks/configs"
    )
    config = package / "tasks/manipulation/cloth_fold/task.w1.yaml"
    script = """
import sys
from dataclasses import replace
from pathlib import Path

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
    get_env_spec,
)
from embodichain.utils.utility import load_config

path = Path(sys.argv[1])
env_id = "TaskProgramClothFold-W1-v1"
assert env_id not in REGISTERED_ENVS
if sys.argv[2] == "True":
    cfg = config_to_cfg(load_config(path), source_path=path)
    assert get_env_spec(env_id).cls is EmbodiedEnv
    discover_task_packages()
else:
    discover_task_packages()
    assert env_id not in REGISTERED_ENVS
    cfg = config_to_cfg(load_config(path), source_path=path)

spec = get_env_spec(env_id)
assert spec.cls is EmbodiedEnv
assert config_to_cfg(load_config(path), source_path=path).final_acceptance == cfg.final_acceptance
assert cfg.robot.uid == "w1"
assert cfg.task_program.program_id == "cloth_single_fold"
assert cfg.kinematic_replay.load()[0].shape == (991, 24)
if sys.argv[2] == "True":
    cfg.extensions = {"task_program": None}
    try:
        spec.make(cfg=cfg)
    except ValueError as exc:
        assert "Extensions cannot override" in str(exc)
    else:
        raise AssertionError("Extensions bypassed configured acceptance.")
    cfg.extensions = None
    surface = cfg.replay_surface
    cfg.replay_surface = replace(surface, mesh_sha256="0" * 64)
    try:
        spec.make(cfg=cfg)
    except ValueError as exc:
        assert "registered replay mesh" in str(exc)
    else:
        raise AssertionError("A different mesh passed construction.")
    cfg.replay_surface = surface
cfg.final_acceptance = replace(cfg.final_acceptance, max_distance_m=0.020)
try:
    spec.make(cfg=cfg)
except ValueError as exc:
    assert "registered final acceptance" in str(exc)
else:
    raise AssertionError("A different acceptance passed construction.")
"""
    robot = tmp_path / "w1_v030" / "robot.urdf"
    robot.parent.mkdir()
    robot.touch()
    subprocess.run(
        [sys.executable, "-c", script, str(config), str(config_first)],
        cwd=tmp_path,
        env={**os.environ, "EMBODICHAIN_DATA_ROOT": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )


def test_fold_remeasures_both_corners_after_program_completion() -> None:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs._final_acceptance import _FinalAcceptance

    env = object.__new__(EmbodiedEnv)
    env._num_envs = 1
    nodes = torch.zeros((1, 4, 3))
    cloth = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=nodes))
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_deformable_object=lambda _: cloth
    )
    env.cfg = SimpleNamespace(task_program=object())
    env._final_acceptance = _FinalAcceptance(
        "cloth", (((0,), (1,)), ((2,), (3,))), 0.015
    )
    bridge = SimpleNamespace(
        program_completed=False, completion_mask=torch.tensor([True])
    )
    env._active_task_program_bridge = bridge
    nodes[0, 1, 0] = 0.015
    nodes[0, 3, 1] = 0.015
    assert not env.is_task_success().item()
    bridge.program_completed = True
    assert env.is_task_success().item()
    nodes[0, 3, 1] = 0.01501
    assert not env.is_task_success().item()
    nodes[0, 3, 1] = float("nan")
    assert not env.is_task_success().item()
