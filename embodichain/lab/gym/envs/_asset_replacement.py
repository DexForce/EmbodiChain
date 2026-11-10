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

"""Prepared Task Program assets applied by the existing replacement event."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import re
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.task_program.integrations.asset_materialization import (
    MaterializedAssetDeployment,
    load_asset_requirements_from_deployment,
    validate_materialized_deployment,
)
from embodichain.utils.configclass import class_to_dict

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg

__all__: list[str] = []


def _declaration(value: Any) -> str:
    """Compare owned declarations, including relocated content-addressed files."""

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            if "func" in item and (
                item.get("mode") == "save"
                or str(item["func"]).endswith("record_camera_data")
            ):
                item = deepcopy(item)
                if "save_path" in item.get("params", {}):
                    item["params"]["save_path"] = "<collection-recording-directory>"
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, str):
            # Materialization names local resources by their complete content
            # digest. Different deployment directories do not change identity.
            match = re.search(r"/(resources/[0-9a-f]{64}/.*)$", item)
            return match[1] if match else item
        if hasattr(item, "tolist"):
            return normalize(item.tolist())
        return item

    return json.dumps(
        normalize(class_to_dict(value)), sort_keys=True, default=str, allow_nan=False
    )


def _check_environment(
    current: EmbodiedEnvCfg, selected: EmbodiedEnvCfg, replaced: set[str]
) -> None:
    """Require the robot, scene layout and execution/recording schema to stay fixed."""
    for name in (
        "num_envs",
        "robot",
        "sensor",
        "light",
        "rigid_object_group",
        "articulation",
        "control_parts",
        "expert_trajectory",
        "max_episode_steps",
        "target_control_frequency",
        "ignore_terminations",
        "enable_sensor",
        "objective",
        "active_joint_ids",
        "actions",
        "observations",
        "rewards",
        "events",
        "dataset",
    ):
        if _declaration(getattr(current, name, None)) != _declaration(
            getattr(selected, name, None)
        ):
            raise ValueError(f"Asset replacement cannot change environment {name}")
    for name in ("physics_cfg", "arena_space", "render_cfg", "headless", "device"):
        if _declaration(getattr(current.sim_cfg, name)) != _declaration(
            getattr(selected.sim_cfg, name)
        ):
            raise ValueError(f"Asset replacement cannot change simulation {name}")
    if current.target_control_frequency is None and (
        current.sim_steps_per_control != selected.sim_steps_per_control
    ):
        raise ValueError("Asset replacement cannot change the control period")
    for group in ("rigid_object", "background"):
        old = {obj.uid: obj for obj in getattr(current, group)}
        new = {obj.uid: obj for obj in getattr(selected, group)}
        if set(old) != set(new):
            raise ValueError("Asset replacement must preserve the scene's object UIDs")
        for uid in set(old) - replaced:
            if _declaration(old[uid]) != _declaration(new[uid]):
                raise ValueError(
                    f"Asset replacement cannot change unselected object {uid}"
                )
    if current.task_program.program_id != selected.task_program.program_id:
        raise ValueError("Asset replacement must preserve the logical Task Program")


def _refresh_manager_bindings(
    env: EmbodiedEnv, selected: EmbodiedEnvCfg, replaced: set[str]
) -> None:
    """Reinitialize only terms whose explicit selectors address replaced bodies."""
    from embodichain.lab.gym.envs.managers import Functor
    from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

    def affected(value: Any) -> bool:
        if isinstance(value, SceneEntityCfg):
            return value.uid in replaced
        if isinstance(value, dict):
            return any(affected(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(affected(item) for item in value)
        return False

    for name, min_argc in (("events", 2), ("observations", 2), ("rewards", 4)):
        manager_name = {
            "events": "event_manager",
            "observations": "observation_manager",
            "rewards": "reward_manager",
        }[name]
        manager = getattr(env, manager_name, None)
        template = getattr(selected, name, None)
        if manager is None or template is None:
            continue
        items = (
            template.items() if isinstance(template, dict) else vars(template).items()
        )
        for term_name, declaration in items:
            if declaration is None or not affected(declaration.params):
                continue
            current = manager.get_functor_cfg(term_name)
            fresh = deepcopy(declaration)
            scope = (
                manager._functor_seed_scope(
                    phase="initialize",
                    mode=fresh.mode,
                    functor_name=term_name,
                )
                if name == "events"
                else nullcontext()
            )
            with scope:
                manager._resolve_common_functor_cfg(term_name, fresh, min_argc=min_argc)
            # Keep the config object referenced by all manager mode/reset lists.
            # Unrelated terms, including recorder instances, retain their state.
            current.params = fresh.params
            current.func = fresh.func
            if isinstance(current.func, Functor):
                current.func.cfg = current


def _apply_asset_deployment(
    env: EmbodiedEnv,
    deployment: MaterializedAssetDeployment,
) -> None:
    """Replace only selected rigid bodies and install a fresh semantic adapter.

    Call at a cleared episode boundary. All file/configuration checks precede
    native mutation; consumers bind afterward to the new native objects. Failures
    stop collection, since the backend does not offer rollback.
    """
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import get_env_spec
    from embodichain.utils.utility import load_config

    if not isinstance(deployment, MaterializedAssetDeployment):
        raise TypeError("deployment must be MaterializedAssetDeployment")
    if env.num_envs != 1:
        raise ValueError("Prepared asset replacement requires B=1")
    if getattr(env, "_scene_expansion_attempt_active", False):
        raise RuntimeError("Asset replacement requires an inactive episode")
    for name in ("_traj_steps", "_demo_steps", "rollout_steps"):
        steps = getattr(env, name, None)
        if isinstance(steps, torch.Tensor) and bool(torch.any(steps != 0).item()):
            raise RuntimeError("Asset replacement requires cleared episode buffers")
    validate_materialized_deployment(deployment)
    manifest = json.loads(deployment.manifest_path.read_text(encoding="utf-8"))
    requirements = load_asset_requirements_from_deployment(deployment.deployment_path)
    config = load_config(deployment.deployment_path)
    selected = config_to_cfg(config, source_path=deployment.deployment_path)
    # Physical UIDs are explicit scene bindings; logical role names need not
    # equal native object names.
    registration = get_env_spec(config["id"]).task_program_registration
    bindings = {
        binding.entity_id: binding.simulation_uid
        for binding in registration.scene_binding.rigid_objects
    }
    replaced = {bindings[role.entity_id] for role in requirements.roles.values()}
    _check_environment(env.cfg, selected, replaced)
    candidate_objects = {
        obj.uid: obj for obj in (*selected.rigid_object, *selected.background)
    }
    replacements = [candidate_objects[uid] for uid in sorted(replaced)]
    preserved = {
        uid: env.sim.get_rigid_object(uid)
        for uid in candidate_objects.keys() - replaced
    }
    robot = env.robot
    factory = get_env_spec(config["id"]).task_program_adapter_factory
    env.sim.replace_rigid_objects(replacements)
    if env.robot is not robot or any(
        env.sim.get_rigid_object(uid) is not value for uid, value in preserved.items()
    ):
        raise RuntimeError("Asset replacement changed an unselected scene object")
    for group in ("rigid_object", "background"):
        setattr(
            env.cfg,
            group,
            [
                candidate_objects[obj.uid] if obj.uid in replaced else obj
                for obj in getattr(env.cfg, group)
            ],
        )
    env.cfg.task_program = selected.task_program
    # The policy port binds native entities eagerly even though planning scene
    # construction is lazy. Construct it only after the new bodies exist.
    adapter = factory.create_adapter(env)
    env._task_program_adapter = adapter
    env._active_task_program_bridge = None
    _refresh_manager_bindings(env, selected, replaced)
    objective = getattr(env, "physical_objective", None)
    if objective is not None and objective.cfg.object_uid in replaced:
        env._physical_objective_object = env.sim.get_rigid_object(
            objective.cfg.object_uid
        )
    from embodichain.data_pipeline.recording import build_recording_provenance

    provenance = build_recording_provenance(env)
    env._recording_provenance = provenance
    for manager_name in ("dataset_manager", "event_manager"):
        manager = getattr(env, manager_name, None)
        for terms in getattr(manager, "_mode_functor_cfgs", {}).values():
            for term in terms:
                if hasattr(term.func, "_recording_provenance"):
                    term.func._recording_provenance = deepcopy(provenance)
    env._asset_generalization_manifest = deepcopy(manifest)
    env._asset_generalization_replacement = {
        "replaced_uids": sorted(replaced),
        "preserved_uids": sorted(preserved),
        "robot_preserved": True,
    }
    metadata = getattr(env, "_demo_episode_metadata", None)
    if metadata:
        metadata[0].update(deepcopy(provenance))
        metadata[0]["asset_generalization"] = {"manifest": deepcopy(manifest)}
