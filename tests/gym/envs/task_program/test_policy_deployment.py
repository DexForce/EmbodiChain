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

"""Declaration-only policy configuration and control authority contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import get_env_spec
from embodichain.lab.task_program.integrations.progress import (
    _validate_policy_action_coverage,
)

_TASK = (
    Path(__file__).resolve().parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.policy.cobotmagic.yaml"
)


def _load(name="ValidPolicyFixture-v1"):
    config = yaml.safe_load(_TASK.read_text())
    config["id"] = name
    return config


def test_policy_loader_registers_rl_with_declarations_and_content_identity():
    config = _load()
    cfg = config_to_cfg(config, source_path=_TASK)
    spec = get_env_spec(config["id"])
    assert spec.supports_rl
    assert cfg.task_program_execution == "policy"
    assert (
        spec.task_program_adapter_factory.registration.catalog.preflight(
            cfg.task_program, strict_contracts=True
        ).segment_count
        == 2
    )
    # The same ID cannot hide a changed ownership declaration or reward scale.
    changed = deepcopy(config)
    changed["task_program"]["action_owners"].pop("secondary_manipulator")
    with pytest.raises(ValueError, match="every declared"):
        config_to_cfg(changed, source_path=_TASK)
    assert config_to_cfg(config, source_path=_TASK).task_program_execution == "policy"
    changed = deepcopy(config)
    changed["rl_task"]["success"]["stages"]["params"]["state"] = "current"
    with pytest.raises(ValueError, match="already has a different"):
        config_to_cfg(changed, source_path=_TASK)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_rl", "require rl_task"),
        ("missing_owners", "every declared"),
        ("expert_owner", "every declared"),
        ("expert_mode", "cannot share"),
        ("missing_success", "requires.*success"),
    ],
)
def test_policy_preflight_rejects_incomplete_or_conflicting_authority(mutation, match):
    config = _load(f"BadPolicy{mutation}-v1")
    if mutation == "missing_rl":
        config.pop("rl_task")
    elif mutation == "missing_owners":
        config["task_program"].pop("action_owners")
    elif mutation == "expert_owner":
        config["task_program"]["action_owners"]["primary_manipulator"] = "expert"
    elif mutation == "expert_mode":
        config["task_program"]["execution"] = "expert"
    else:
        config["rl_task"]["success"] = {}
    with pytest.raises(ValueError, match=match):
        config_to_cfg(config, source_path=_TASK)


@pytest.mark.parametrize(
    "name,value",
    [
        ("task_program_execution", "expert"),
        ("_after_observation", None),
        ("task_program_progress", None),
    ],
)
def test_policy_extensions_cannot_override_validated_authority(tmp_path, name, value):
    config = _load(f"BadPolicyExtension{name}-v1")
    physical = yaml.safe_load((_TASK.parent / "env.policy.yaml").read_text())
    physical["env"]["extensions"] = {name: value}
    (tmp_path / "env.yaml").write_text(yaml.safe_dump(physical))
    config["environment"]["component"] = str(tmp_path / "env.yaml")
    with pytest.raises(ValueError, match="extensions cannot override"):
        config_to_cfg(config, source_path=_TASK)


def test_mimic_followers_are_not_required_policy_action_producers():
    endpoint = SimpleNamespace(control_part="eef")
    registration = SimpleNamespace(
        robot_profile_binding=SimpleNamespace(
            resources=[SimpleNamespace(endpoints=[endpoint])]
        )
    )
    robot = SimpleNamespace(
        joint_names=["driver", "follower"],
        get_joint_ids=lambda *, name, remove_mimic: [0] if remove_mimic else [0, 1],
    )
    manager = SimpleNamespace(
        descriptors=[SimpleNamespace(term=SimpleNamespace(joint_names=["driver"]))]
    )
    _validate_policy_action_coverage(registration, robot, manager)
    manager.descriptors = []
    with pytest.raises(ValueError, match="driver"):
        _validate_policy_action_coverage(registration, robot, manager)


@pytest.mark.parametrize(
    "unsupported,match",
    [
        ("post", "post-policies"),
        ("temporal", "instantaneous"),
        ("implicit", "instruction"),
    ],
)
def test_unsupported_policy_subset_fails_static_preflight(tmp_path, unsupported, match):
    config = _load(f"UnsupportedPolicy{unsupported}-v1")
    program = yaml.safe_load(
        (_TASK.parent / "task_program/program.measured.yaml").read_text()
    )
    if unsupported == "post":
        program["program"][0]["post"] = [{"kind": "wait_stable", "entity": "block_2"}]
    elif unsupported == "implicit":
        program["program"] = {"pick": {"object": "block_2", "grasp": "block_2_grasp"}}
    else:
        evaluation = yaml.safe_load(
            (_TASK.parent / "task_program/evaluation.measured.yaml").read_text()
        )
        evaluation["graph"]["rules"]["lifted"] = {
            "kind": "hold_for",
            "ref": "below_clearance",
            "end": 3,
            "steps": 2,
        }
        path = tmp_path / "evaluation.yaml"
        path.write_text(yaml.safe_dump(evaluation))
        config["evaluation"] = {"component": str(path)}
    config["task_program"]["program"] = {"inline": program}
    with pytest.raises(ValueError, match=match):
        config_to_cfg(config, source_path=_TASK, _register_env=False)


def test_strict_inspector_checks_policy_authority_without_registering(monkeypatch):
    import runpy
    import gymnasium
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS

    inspect = runpy.run_path(
        str(
            _TASK.parents[7]
            / ".agents/skills/add-task-program/scripts/inspect_deployment.py"
        )
    )["inspect_deployment"]
    before = (set(REGISTERED_ENVS), set(gymnasium.envs.registry))
    assert inspect(_TASK, strict_contracts=True)["execution"] == "policy"
    assert before == (set(REGISTERED_ENVS), set(gymnasium.envs.registry))
    config = _load("InvalidPolicyInspector-v1")
    config.pop("rl_task")
    monkeypatch.setitem(inspect.__globals__, "load_config", lambda path: config)
    with pytest.raises(ValueError, match="require rl_task"):
        inspect(_TASK, strict_contracts=True)
    assert before == (set(REGISTERED_ENVS), set(gymnasium.envs.registry))
