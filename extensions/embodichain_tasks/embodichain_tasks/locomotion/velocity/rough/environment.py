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

"""Build rough-terrain velocity environments from frozen scene assets."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules

from .config import load_config

__all__ = ["load_rough_env_cfg"]


def load_rough_env_cfg(
    robot: str,
    asset_bundle: str | Path,
    *,
    num_envs: int = 1,
    seed: int = 42,
    training_randomization: bool = False,
    use_mujoco_contacts: bool | None = None,
    initial_state_file: str | Path | None = None,
) -> EmbodiedEnvCfg:
    """Load G1 Rough with shared evaluation or training terrain.

    Args:
        robot: ``g1``.
        asset_bundle: Directory containing the exported robot USDs and, for G1,
            ``terrain.npz`` and ``terrain.json`` with evaluation origins or the
            complete training terrain and curriculum definition.
        num_envs: Independent physics worlds.
        seed: Environment random seed.
        training_randomization: Enable observation noise, startup mass/COM
            events, random reset and interval velocity pushes.
        use_mujoco_contacts: Select native contacts or Newton contacts while
            retaining MJWarp dynamics. ``None`` keeps the packaged setting.
        initial_state_file: Shared deterministic benchmark states. Rows repeat
            in environment ID order and apply again on episode resets.

    Returns:
        The resolved EmbodiChain environment configuration.

    Raises:
        ValueError: The robot selector is unknown or bundle metadata is invalid.
        FileNotFoundError: A required scene asset is missing.
    """
    tasks = {"g1": "g1_rough"}
    if robot not in tasks:
        raise ValueError(f"Unknown robot selector: {robot!r}")
    bundle = Path(asset_bundle).expanduser().resolve()
    asset = bundle / f"{robot}.usda"
    if not asset.is_file():
        raise FileNotFoundError(asset)
    resource = files("embodichain_tasks.configs").joinpath(
        "tasks", "locomotion", "velocity", tasks[robot], "env.newton.yaml"
    )
    data = yaml.safe_load(resource.read_text())
    data["num_envs"] = num_envs
    data["robot"]["fpath"] = str(asset)
    solver_cfg = data["physics_config"]["solver_cfg"]
    mj_target = (
        solver_cfg.setdefault("mujoco_options", {})
        if solver_cfg.get("solver_type") == "dexuni"
        else solver_cfg
    )
    if use_mujoco_contacts is None:
        use_mujoco_contacts = mj_target["use_mujoco_contacts"]
    mj_target["use_mujoco_contacts"] = use_mujoco_contacts
    if use_mujoco_contacts:
        data["physics_config"]["collision_cfg"] = None
        solver_cfg.pop("collision_options", None)
    # Newton contact mode sets dynamic friction to 0.8, while native MJWarp
    # contact mode retains the imported sliding friction of 1.0.
    data["robot"]["attrs"]["material_props"]["dynamic_friction"] = (
        1.0 if use_mujoco_contacts else 0.8
    )
    extension = data["env"]["extensions"]
    extension["deterministic_reset"] = not training_randomization
    extension["randomization"] = training_randomization
    if initial_state_file is not None:
        if training_randomization:
            raise ValueError(
                "Shared initial states require deterministic benchmark mode."
            )
        states = json.loads(Path(initial_state_file).read_text())
        task = load_config(tasks[robot])
        if (
            states.get("schema") != "rough_velocity.initial_states@1"
            or states.get("robot") != robot
            or states.get("joint_names") != list(task.joint_names)
        ):
            raise ValueError(
                "Initial states must match the target robot and policy joint order."
            )
        rows = states["initial_states"]
        if not rows:
            raise ValueError("Initial state list must be nonempty.")
        limits = np.asarray(task.data["robot"]["soft_joint_pos_limits"])
        for row in rows:
            for key, size in (
                ("root_pose", 7),
                ("root_velocity", 6),
                ("qpos", task.action_dim),
                ("qvel", task.action_dim),
            ):
                values = np.asarray(row[key])
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid initial state field: {key}")
            if not np.isclose(np.linalg.norm(row["root_pose"][3:]), 1.0, atol=1e-6):
                raise ValueError(
                    "Initial root quaternion must be normalized in XYZW order."
                )
            position = np.asarray(row["qpos"])
            if np.any(position < limits[:, 0] - 1e-7) or np.any(
                position > limits[:, 1] + 1e-7
            ):
                raise ValueError("Initial joint positions exceed the task soft limits.")
        extension["initial_states"] = rows
    for term in data["env"]["observations"].values():
        term["params"]["enable_corruption"] = training_randomization
    if robot == "g1":
        terrain = bundle / "terrain.npz"
        if not terrain.is_file():
            raise FileNotFoundError(terrain)
        extension["terrain"] = str(terrain)
        metadata = json.loads((bundle / "terrain.json").read_text())
        if training_randomization:
            required_keys = (
                "terrain_origins",
                "max_init_terrain_level",
                "generator",
            )
            if not all(key in metadata for key in required_keys):
                raise ValueError(
                    "G1 training requires terrain.json with terrain_origins, "
                    "max_init_terrain_level, and generator."
                )
            extension["terrain_course"] = metadata
        else:
            extension["origin"] = metadata["env_origins"][0]
    cfg = config_to_cfg(data, manager_modules=get_manager_modules())
    cfg.seed = seed
    return cfg
