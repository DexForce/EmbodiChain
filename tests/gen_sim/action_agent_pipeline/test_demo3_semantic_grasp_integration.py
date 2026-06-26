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

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DEXSIM_GRASP_TESTS") != "1",
    reason="Set RUN_DEXSIM_GRASP_TESTS=1 to run DexSim semantic grasp integration tests.",
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEMO3_CONFIG_DIR = (
    _REPO_ROOT / "gym_project/action_agent_pipeline/configs/demo3_text"
)
_DEMO3_CONFIG_DIR = (
    Path(
        os.environ.get(
            "RUN_DEXSIM_GRASP_CONFIG_DIR",
            str(_DEFAULT_DEMO3_CONFIG_DIR),
        )
    )
    .expanduser()
    .resolve()
)
_MIN_LIFT_M = float(os.environ.get("RUN_DEXSIM_GRASP_MIN_LIFT_M", "0.04"))
_MAX_EEF_DISTANCE_M = float(
    os.environ.get("RUN_DEXSIM_GRASP_MAX_EEF_DISTANCE_M", "0.25")
)
_POST_GRASP_HOLD_STEPS = int(os.environ.get("RUN_DEXSIM_GRASP_HOLD_STEPS", "10"))
_PICK_UP_SPEC_RE = re.compile(
    r'"atomic_action_class"\s*:\s*"PickUp".*?'
    r'"robot_name"\s*:\s*"(?P<robot_name>[^"]+)".*?'
    r'"obj_name"\s*:\s*"(?P<obj_name>[^"]+)"',
    re.DOTALL,
)


def _load_demo3_gym_config() -> dict:
    return json.loads(
        (_DEMO3_CONFIG_DIR / "fast_gym_config.json").read_text(encoding="utf-8")
    )


def _configured_rigid_object_uids() -> set[str]:
    return {
        rigid_object["uid"]
        for rigid_object in _load_demo3_gym_config().get("rigid_object", [])
    }


def _configured_grasp_targets() -> list[tuple[str, str]]:
    atom_actions_text = (_DEMO3_CONFIG_DIR / "atom_actions.txt").read_text(
        encoding="utf-8"
    )
    targets = [
        (match.group("robot_name"), match.group("obj_name"))
        for match in _PICK_UP_SPEC_RE.finditer(atom_actions_text)
    ]
    rigid_object_uids = _configured_rigid_object_uids()
    stale_targets = [
        (robot_name, obj_name)
        for robot_name, obj_name in targets
        if obj_name not in rigid_object_uids
    ]
    assert not stale_targets, (
        "atom_actions.txt references pick_up objects that are not present in "
        f"fast_gym_config.json: stale_targets={stale_targets}, "
        f"rigid_object_uids={sorted(rigid_object_uids)}."
    )
    return targets


def _configured_grasp_target_for(*keywords: str) -> tuple[str, str]:
    lower_keywords = tuple(keyword.lower() for keyword in keywords)
    matches = [
        (robot_name, obj_name)
        for robot_name, obj_name in _configured_grasp_targets()
        if all(keyword in obj_name.lower() for keyword in lower_keywords)
    ]
    assert matches, (
        f"No configured grasp target matching keywords={keywords}. "
        f"grasp_targets={_configured_grasp_targets()}."
    )
    assert (
        len(matches) == 1
    ), f"Ambiguous grasp target matching keywords={keywords}: {matches}."
    return matches[0]


def _write_runtime_gym_config(tmp_path: Path) -> Path:
    gym_config = _load_demo3_gym_config()
    gym_config["env"]["events"] = {}
    gym_config["env"]["dataset"] = {}
    gym_config["sensor"] = []

    runtime_config_path = tmp_path / "demo3_semantic_grasp_gym_config.json"
    runtime_config_path.write_text(
        json.dumps(gym_config, indent=2),
        encoding="utf-8",
    )
    return runtime_config_path


def _make_env(tmp_path: Path):
    import gymnasium

    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from embodichain.utils.utility import load_config

    # Import registers AtomicActionsAgent-v3.
    from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware import (  # noqa: F401
        agent_env,
    )

    args = argparse.Namespace(
        num_envs=1,
        device=os.environ.get("RUN_DEXSIM_GRASP_DEVICE", "cpu"),
        headless=True,
        renderer=os.environ.get("RUN_DEXSIM_GRASP_RENDERER", "hybrid"),
        arena_space=float(os.environ.get("RUN_DEXSIM_GRASP_ARENA_SPACE", "5.0")),
        gpu_id=int(os.environ.get("RUN_DEXSIM_GRASP_GPU_ID", "0")),
        gym_config=str(_write_runtime_gym_config(tmp_path)),
        action_config=None,
        preview=False,
        filter_visual_rand=True,
        filter_dataset_saving=True,
    )
    env_cfg, gym_config, _ = build_env_cfg_from_args(args)
    agent_config_path = _DEMO3_CONFIG_DIR / "agent_config.json"
    return gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=load_config(agent_config_path),
        agent_config_path=str(agent_config_path),
        task_name="Demo3_Text",
    )


def _object_xyz(env, obj_name: str) -> torch.Tensor:
    pose = env.sim.get_rigid_object(obj_name).get_local_pose(to_matrix=True).squeeze(0)
    return pose[:3, 3].detach().cpu()


def _arm_eef_xyz(env, robot_name: str) -> torch.Tensor:
    left_pose, right_pose = env.get_current_xpos_agent()
    pose = left_pose if "left" in robot_name else right_pose
    return pose[:3, 3].detach().cpu()


def _hold_last_action(env, actions: list, steps: int) -> None:
    if steps <= 0 or not actions:
        return
    last_action = actions[-1]
    for _ in range(steps):
        env.step(last_action)


def _assert_semantic_grasp_lifts_object(
    tmp_path: Path,
    *,
    robot_name: str,
    obj_name: str,
) -> None:
    import torch

    from embodichain.gen_sim.action_agent_pipeline.runtime.atom_actions import (
        execute_parallel_atomic_actions,
    )

    gym_env = _make_env(tmp_path)
    env = gym_env.unwrapped
    try:
        gym_env.reset()
        z_before = float(_object_xyz(env, obj_name)[2])
        action_spec = {
            "atomic_action_class": "PickUp",
            "robot_name": robot_name,
            "control": "arm",
            "target_object": {
                "obj_name": obj_name,
                "affordance": "antipodal",
            },
            "cfg": {
                "pre_grasp_distance": 0.08,
                "lift_height": 0.14,
                "sample_interval": 80,
            },
        }
        result = execute_parallel_atomic_actions(
            left_arm_action=action_spec if "left" in robot_name else None,
            right_arm_action=action_spec if "right" in robot_name else None,
            env=env,
            return_result=True,
            allow_grasp_annotation=True,
            force_grasp_reannotate=bool(
                int(os.environ.get("RUN_DEXSIM_GRASP_FORCE_REANNOTATE", "0"))
            ),
        )
        _hold_last_action(env, result["actions"], _POST_GRASP_HOLD_STEPS)

        obj_xyz = _object_xyz(env, obj_name)
        eef_xyz = _arm_eef_xyz(env, robot_name)
        lift = float(obj_xyz[2] - z_before)
        eef_distance = float(torch.linalg.norm(obj_xyz - eef_xyz))

        assert lift >= _MIN_LIFT_M, (
            f"{obj_name} semantic grasp did not lift enough: lift={lift:.4f}m, "
            f"required={_MIN_LIFT_M:.4f}m, obj_xyz={obj_xyz.tolist()}, "
            f"eef_xyz={eef_xyz.tolist()}."
        )
        assert eef_distance <= _MAX_EEF_DISTANCE_M, (
            f"{obj_name} is too far from {robot_name} after grasp: "
            f"distance={eef_distance:.4f}m, "
            f"required<={_MAX_EEF_DISTANCE_M:.4f}m, "
            f"obj_xyz={obj_xyz.tolist()}, eef_xyz={eef_xyz.tolist()}."
        )
    finally:
        gym_env.close()


def test_demo3_semantic_grasp_lifts_orange(tmp_path: Path) -> None:
    robot_name, obj_name = _configured_grasp_target_for("orange", "1")
    _assert_semantic_grasp_lifts_object(
        tmp_path,
        robot_name=robot_name,
        obj_name=obj_name,
    )


def test_demo3_semantic_grasp_lifts_can(tmp_path: Path) -> None:
    robot_name, obj_name = _configured_grasp_target_for("can")
    _assert_semantic_grasp_lifts_object(
        tmp_path,
        robot_name=robot_name,
        obj_name=obj_name,
    )
