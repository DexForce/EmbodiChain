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

"""Verify canonical protocol ownership without changing serialized behavior."""

from __future__ import annotations

from types import ModuleType

from embodichain.gen_sim.action_agent_pipeline import contracts as legacy_contracts
from embodichain.gen_sim.action_agent_pipeline import semantics as legacy_semantics
from embodichain.gen_sim.action_agent_pipeline.config import defaults
from embodichain.gen_sim.action_agent_pipeline.domain import object_semantics
from embodichain.gen_sim.action_agent_pipeline.generation import relation_language
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    NominalGraphStep,
    build_nominal_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.protocol import (
    actions,
    artifacts,
    success,
    tasks,
)

_LEGACY_CONTRACT_EXPORTS = frozenset(
    {
        "ACTION_AGENT_ENV_ID",
        "AGENT_CONFIG_FILENAME",
        "ARM_ACTION_KEYS",
        "ATOM_ACTIONS_FILENAME",
        "ATOMIC_ACTION_CLASSES",
        "BASIC_BACKGROUND_FILENAME",
        "COMPILED_GRAPH_FILENAME",
        "CONTROL_ARM",
        "CONTROL_HAND",
        "DEFAULT_VIEWER_CAMERA_UID",
        "DUAL_ARM_NAME",
        "FAST_GYM_CONFIG_FILENAME",
        "LEFT_ARM_ACTION_KEY",
        "LEFT_ARM_NAME",
        "MANIPULATION_INTENTS",
        "MAX_COORDINATED_PAYLOADS",
        "OBJECT_ORIENTATION_AXES",
        "OBJECT_ORIENTATION_GOALS",
        "POSE_REFERENCES",
        "RELATIVE_RELATIONS",
        "RIGHT_ARM_ACTION_KEY",
        "RIGHT_ARM_NAME",
        "ROBOTIQ_ARG2F_140_CLOSE_QPOS",
        "ROBOTIQ_ARG2F_140_OPEN_QPOS",
        "SIDE_RELATIONS",
        "SUCCESS_TERM_ALIASES",
        "SUCCESS_TERM_TYPES",
        "SUPPORTED_CONTROLS",
        "SuccessTerm",
        "TASK_GRAPH_FILENAME",
        "TASK_PROMPT_FILENAME",
        "TASK_ROUTE_ARRANGEMENT_LINE",
        "TASK_ROUTE_OBJECT_MANIPULATION",
        "TASK_ROUTE_STACKING",
        "TASK_ROUTE_UNSUPPORTED",
        "TASK_ROUTES",
    }
)
_LEGACY_SEMANTIC_EXPORTS = frozenset(
    {
        "BOTTLE_LIKE_KEYWORDS",
        "CONTAINER_LIKE_KEYWORDS",
        "CUP_LIKE_KEYWORDS",
        "FLAT_CARRIER_KEYWORDS",
        "RELATIVE_RELATION_PHRASES",
        "ROD_LIKE_KEYWORDS",
        "SHORT_BOTTLE_LIKE_KEYWORDS",
        "SHORT_CUP_LIKE_KEYWORDS",
        "UPRIGHTABLE_KEYWORDS",
        "relative_relation_phrase",
    }
)


def test_legacy_contract_exports_are_identity_aliases() -> None:
    canonical_exports = _public_exports(actions, artifacts, success, tasks)
    canonical_exports.update(
        {
            "ROBOTIQ_ARG2F_140_CLOSE_QPOS": (defaults.ROBOTIQ_ARG2F_140_CLOSE_QPOS),
            "ROBOTIQ_ARG2F_140_OPEN_QPOS": defaults.ROBOTIQ_ARG2F_140_OPEN_QPOS,
        }
    )

    assert set(canonical_exports) == _LEGACY_CONTRACT_EXPORTS
    assert set(legacy_contracts.__all__) == _LEGACY_CONTRACT_EXPORTS
    for name, canonical_value in canonical_exports.items():
        assert getattr(legacy_contracts, name) is canonical_value


def test_legacy_semantic_exports_are_identity_aliases() -> None:
    canonical_exports = _public_exports(object_semantics, relation_language)

    assert set(canonical_exports) == _LEGACY_SEMANTIC_EXPORTS
    assert set(legacy_semantics.__all__) == _LEGACY_SEMANTIC_EXPORTS
    for name, canonical_value in canonical_exports.items():
        assert getattr(legacy_semantics, name) is canonical_value


def test_protocol_sets_are_internally_consistent() -> None:
    assert set(success.SUCCESS_TERM_ALIASES.values()) <= success.SUCCESS_TERM_TYPES
    assert set(relation_language.RELATIVE_RELATION_PHRASES) == tasks.RELATIVE_RELATIONS
    assert tasks.SIDE_RELATIONS == tasks.RELATIVE_RELATIONS - {"inside", "on"}


def test_robotiq_fallbacks_preserve_historical_tuple_contract() -> None:
    expected_open = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected_close = (0.7, -0.7, 0.7, -0.7, -0.7, 0.7)

    assert defaults.ROBOTIQ_ARG2F_140_OPEN_QPOS == expected_open
    assert defaults.ROBOTIQ_ARG2F_140_CLOSE_QPOS == expected_close
    assert isinstance(defaults.ROBOTIQ_ARG2F_140_OPEN_QPOS, tuple)
    assert isinstance(defaults.ROBOTIQ_ARG2F_140_CLOSE_QPOS, tuple)
    assert len(defaults.ROBOTIQ_ARG2F_140_OPEN_QPOS) == 6
    assert len(defaults.ROBOTIQ_ARG2F_140_CLOSE_QPOS) == 6


def test_artifact_and_integration_identifiers_remain_stable() -> None:
    assert artifacts.FAST_GYM_CONFIG_FILENAME == "fast_gym_config.json"
    assert artifacts.AGENT_CONFIG_FILENAME == "agent_config.json"
    assert artifacts.TASK_GRAPH_FILENAME == "task_graph.json"
    assert artifacts.ACTION_AGENT_ENV_ID == "AtomicActionsAgent-v3"
    assert artifacts.DEFAULT_VIEWER_CAMERA_UID == "cam_high"
    assert actions.LEFT_ARM_ACTION_KEY == "left_arm_action"
    assert actions.RIGHT_ARM_ACTION_KEY == "right_arm_action"
    assert tasks.TASK_ROUTE_ARRANGEMENT_LINE == "arrangement_line"
    assert success.SuccessTerm.OBJECTS_ORDERED == "objects_ordered"


def test_generated_agent_and_task_graph_protocol_fields_remain_stable() -> None:
    agent_config = make_agent_config()
    graph = build_nominal_task_graph(
        task_name="protocol_regression",
        steps=[
            NominalGraphStep(
                semantic="Move left arm",
                left_arm_action={"atomic_action_class": "MoveJoints"},
            )
        ],
    )

    assert (
        agent_config["TaskAgent"]["precomputed_task_graph"]
        == artifacts.TASK_GRAPH_FILENAME
    )
    assert set(graph["edges"][0]) == {
        "id",
        "source",
        "target",
        actions.LEFT_ARM_ACTION_KEY,
        actions.RIGHT_ARM_ACTION_KEY,
    }


def _public_exports(*modules: ModuleType) -> dict[str, object]:
    """Collect explicitly public objects from canonical owner modules."""
    return {
        name: getattr(module, name) for module in modules for name in module.__all__
    }
