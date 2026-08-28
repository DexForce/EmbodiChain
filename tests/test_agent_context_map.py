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

from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_AGENT_CONTEXT_ROOT = _REPOSITORY_ROOT / "agent_context"
_MAP_PATH = _AGENT_CONTEXT_ROOT / "MAP.yaml"
_REQUIRED_TOPIC_FIELDS = {
    "id",
    "title",
    "aliases",
    "keywords",
    "paths",
    "source_of_truth",
    "related_topics",
    "status",
}


def _load_context_map() -> dict:
    with _MAP_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _topics_by_id() -> dict[str, dict]:
    return {topic["id"]: topic for topic in _load_context_map()["topics"]}


def test_topic_ids_are_unique() -> None:
    topics = _load_context_map()["topics"]
    topic_ids = [topic["id"] for topic in topics]

    assert len(topic_ids) == len(set(topic_ids))


def test_topic_entries_use_the_required_schema() -> None:
    errors: list[str] = []
    for topic in _load_context_map()["topics"]:
        missing = _REQUIRED_TOPIC_FIELDS - set(topic)
        if missing:
            errors.append(
                f"{topic.get('id', '<missing-id>')}: missing {sorted(missing)}"
            )
        if topic.get("status") not in {"active", "deprecated"}:
            errors.append(
                f"{topic.get('id', '<missing-id>')}: invalid status "
                f"{topic.get('status')!r}"
            )

    assert errors == []


def test_related_topics_reference_registered_ids() -> None:
    topics = _topics_by_id()
    invalid_relations = [
        f"{topic_id} -> {related_id}"
        for topic_id, topic in topics.items()
        for related_id in topic["related_topics"]
        if related_id not in topics
    ]

    assert invalid_relations == []


def test_simulation_and_rl_topics_cover_their_primary_entry_points() -> None:
    topics = _topics_by_id()

    assert {
        "embodichain/lab/sim/__init__.py",
        "embodichain/lab/sim/sim_manager.py",
        "embodichain/lab/gym/envs/base_env.py",
    } <= set(topics["simulation-system"]["source_of_truth"])
    assert {
        "embodichain/__main__.py",
        "embodichain/learning/rl/train.py",
        "embodichain/learning/rl/utils/trainer.py",
        "embodichain_tasks/configs/tasks/",
    } <= set(topics["rl-learning"]["source_of_truth"])


def test_simulation_and_rl_topics_have_operational_sections() -> None:
    topics = _topics_by_id()
    required_sections = {
        "## Entry Points",
        "## Invariants",
        "## Common Failure Modes",
    }
    missing_sections: list[str] = []

    for topic_id in ("simulation-system", "rl-learning"):
        context_path = _AGENT_CONTEXT_ROOT / topics[topic_id]["paths"][0]
        content = context_path.read_text(encoding="utf-8")
        for section in required_sections:
            if section not in content:
                missing_sections.append(f"{topic_id}: {section}")

    assert missing_sections == []


def test_representative_navigation_terms_have_one_owner() -> None:
    expected_owners = {
        "simulation manager": "simulation-system",
        "SimulationManager": "simulation-system",
        "viser": "sim-visualization",
        "rl config": "rl-learning",
        "train-rl": "rl-learning",
        "ik solver": "ik-solvers",
    }
    topics = _load_context_map()["topics"]
    actual_owners: dict[str, set[str]] = {}

    for term in expected_owners:
        normalized_term = term.casefold()
        actual_owners[term] = {
            topic["id"]
            for topic in topics
            if normalized_term
            in {
                candidate.casefold()
                for candidate in [*topic["aliases"], *topic["keywords"]]
            }
        }

    assert actual_owners == {
        term: {topic_id} for term, topic_id in expected_owners.items()
    }


def test_project_context_adapters_reference_the_canonical_skill() -> None:
    canonical_path = ".agents/skills/project-dev-context/SKILL.md"
    adapter_paths = (
        _REPOSITORY_ROOT / ".claude/skills/project-dev-context/SKILL.md",
        _REPOSITORY_ROOT / ".github/copilot/project-dev-context.md",
    )
    missing_references = [
        str(path.relative_to(_REPOSITORY_ROOT))
        for path in adapter_paths
        if canonical_path not in path.read_text(encoding="utf-8")
    ]

    assert missing_references == []
