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

"""Shared environment construction and identity for the ACT example."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from embodichain.utils.config_paths import resolve_config_path

TASK = resolve_config_path(
    "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/task.franka.visual.yaml",
    base_dir=Path(__file__).parent,
).resolve()
CONFIG_ROOT = TASK.parents[3]
TASK_PACKAGE = CONFIG_ROOT.parent / "embodichain_tasks"
if not TASK_PACKAGE.is_dir():
    TASK_PACKAGE = CONFIG_ROOT.parent
IDENTITY_FILES = {
    "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/task.franka.visual.yaml": TASK,
    "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/env.visual.yaml": TASK.parent
    / "env.visual.yaml",
    "extensions/embodichain_tasks/configs/components/embodiments/franka_panda_rgb_224.yaml": CONFIG_ROOT
    / "components/embodiments/franka_panda_rgb_224.yaml",
    "extensions/embodichain_tasks/embodichain_tasks/manipulation/push_cube.py": TASK_PACKAGE
    / "manipulation/push_cube.py",
}


def task_identity() -> dict[str, str]:
    """Hash the task's physical configuration and expert implementation."""
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in IDENTITY_FILES.items()
    }


def make_env(device: str, *, camera: bool, save_path: Path | None = None) -> Any:
    """Build a single physical task instance with optional LeRobot recording."""
    import gymnasium as gym
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import discover_task_packages

    config = _resolve_gym_components(load_config(TASK), base_dir=TASK.parent).config
    config.update(
        device=device,
        gpu_id=int(device.split(":")[-1]) if device.startswith("cuda:") else 0,
        num_envs=1,
        enable_sensor=camera,
        renderer="hybrid" if camera else "no-render",
        headless=True,
    )
    if save_path is not None:
        if not camera:
            raise ValueError("Demonstration recording requires the RGB camera")
        config["env"]["dataset"] = {
            "lerobot": {
                "func": "LeRobotRecorder",
                "mode": "save",
                "params": {
                    "save_path": str(save_path),
                    "use_videos": False,
                    "image_writer_threads": 2,
                    "extra": {
                        "scene_type": "tabletop",
                        "task_description": "visual_push_cube",
                        "data_type": "sim",
                    },
                    "action_contract": {
                        "version": 1,
                        "representation": "joint_position_parallel_gripper",
                    },
                },
            }
        }
    discover_task_packages()
    cfg = config_to_cfg(config, manager_modules=get_manager_modules())
    cfg.init_rollout_buffer = save_path is not None
    cfg.filter_dataset_saving = save_path is None
    cfg.rollout_action_semantics = "policy_request"
    return gym.make(config["id"], cfg=cfg)


def close_env(env: Any) -> None:
    """Close the Gym environment and drain deferred simulation cleanup."""
    from embodichain.lab.sim import SimulationManager

    if env is not None:
        env.close()
    SimulationManager.flush_cleanup_queue()


def schema_for_env(env: Any) -> dict[str, Any]:
    """Build the explicit ordered recording schema from the live manager."""
    raw = env.unwrapped
    names = [raw.robot.joint_names[index] for index in raw.active_joint_ids]
    actions = [
        name
        for descriptor in raw.action_manager.descriptors
        for name in descriptor.term.feature_names
    ]
    return {
        "observation": {"names": names, "dtype": "float32"},
        "action": {
            "names": actions,
            "dtype": "float32",
            "semantics": "joint_position_parallel_gripper@1",
            "step_dt": float(raw.step_dt),
        },
    }
