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

"""Build reference environments from explicitly supplied frozen scene assets."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules

from .config import load_config

__all__ = ["load_reference_env_cfg"]


def load_reference_env_cfg(
    robot: str,
    reference_bundle: str | Path,
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
        reference_bundle: Directory containing the exported robot USDs and,
            for G1, terrain.npz and terrain.json with evaluation origins or the
            complete reference training terrain and curriculum definition.
        num_envs: Independent physics worlds.
        seed: Environment random seed.
        training_randomization: Enable reference observation noise, startup
            mass/COM events, random reset and interval velocity pushes.
        use_mujoco_contacts: Select native contacts or Newton contacts while
            retaining MJWarp dynamics. ``None`` uses the EA reference setting.
        initial_state_file: Shared deterministic benchmark states. Rows repeat
            in environment ID order and apply again on episode resets.

    Returns:
        The resolved EmbodiChain environment configuration.

    Raises:
        ValueError: The robot selector is unknown.
        FileNotFoundError: A required reference asset is missing.
    """
    tasks = {"g1": "g1_isaac_rough"}
    if robot not in tasks:
        raise ValueError(f"Unknown reference robot: {robot!r}")
    bundle = Path(reference_bundle).expanduser().resolve()
    asset = bundle / f"{robot}.usda"
    if not asset.is_file():
        raise FileNotFoundError(asset)
    resource = files("embodichain_tasks.configs").joinpath(
        "tasks", "locomotion", "velocity", tasks[robot], "env.newton.yaml"
    )
    data = yaml.safe_load(resource.read_text())
    data["num_envs"] = num_envs
    data["robot"]["fpath"] = str(asset)
    if use_mujoco_contacts is None:
        use_mujoco_contacts = data["physics_config"]["solver_cfg"][
            "use_mujoco_contacts"
        ]
    data["physics_config"]["solver_cfg"]["use_mujoco_contacts"] = use_mujoco_contacts
    if use_mujoco_contacts:
        data["physics_config"]["collision_cfg"] = None
    # The EA startup material event changes Newton mu to 0.8, while its native
    # MJWarp model retains the imported sliding friction of 1.0.
    data["robot"]["attrs"]["material_props"]["dynamic_friction"] = (
        1.0 if use_mujoco_contacts else 0.8
    )
    extension = data["env"]["extensions"]
    extension["reference_deterministic_reset"] = not training_randomization
    extension["reference_randomization"] = training_randomization
    if initial_state_file is not None:
        if training_randomization:
            raise ValueError(
                "Shared initial states require deterministic benchmark mode."
            )
        states = json.loads(Path(initial_state_file).read_text())
        task = load_config(tasks[robot])
        if (
            states.get("schema") != "isaac_velocity.initial_states@1"
            or states.get("robot") != robot
            or states.get("joint_names") != list(task.joint_names)
        ):
            raise ValueError(
                "Initial states must match the reference robot and policy joint order."
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
                raise ValueError(
                    "Initial joint positions exceed the reference soft limits."
                )
        extension["reference_initial_states"] = rows
    for term in data["env"]["observations"].values():
        term["params"]["enable_corruption"] = training_randomization
    if robot == "g1":
        terrain = bundle / "terrain.npz"
        if not terrain.is_file():
            raise FileNotFoundError(terrain)
        extension["reference_terrain"] = str(terrain)
        metadata = json.loads((bundle / "terrain.json").read_text())
        if training_randomization:
            if metadata.get("schema") != "isaac_velocity.rough_training_terrain@1":
                raise ValueError(
                    "G1 training requires the complete reference Rough terrain bundle."
                )
            extension["reference_terrain_course"] = metadata
        else:
            extension["reference_origin"] = metadata["env_origins"][0]
    cfg = config_to_cfg(data, manager_modules=get_manager_modules())
    cfg.seed = seed
    return cfg
