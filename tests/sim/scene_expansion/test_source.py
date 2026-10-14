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

from collections.abc import Iterator
from dataclasses import MISSING

import pytest

from embodichain.lab.sim.scene_expansion import (
    FixedSceneCandidateSource,
    SceneExpansionCfg,
    ScenePoseChange,
)


def _change(uid: str = "cube", x: float = 0.0) -> ScenePoseChange:
    return ScenePoseChange(
        uid, ((1, 0, 0, x), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )


def test_fixed_source_consumes_only_budgeted_prefix_and_repeats_it() -> None:
    consumed = []

    def generate() -> Iterator[tuple[ScenePoseChange, ...]]:
        for index in range(3):
            consumed.append(index)
            yield (_change(x=index / 10),)

    candidates = generate()
    source = FixedSceneCandidateSource(
        SceneExpansionCfg(seed=23, max_candidates=2, movable_entity_ids=("cube",)),
        "scene:revision-1",
        candidates,
    )

    assert consumed == [0, 1]
    assert len(source) == 2
    assert [variant.ordinal for variant in source] == [0, 1]
    assert {variant.seed for variant in source} == {23}
    assert tuple(source) == tuple(source) == source.variants
    assert next(candidates)[0].arena_pose[0][3] == pytest.approx(0.2)


def test_source_owns_config_and_candidate_containers() -> None:
    movable = ["cube"]
    cfg = SceneExpansionCfg(seed=5, max_candidates=2, movable_entity_ids=movable)
    changes = [_change()]
    candidates = [changes]
    metadata = [["source", "fixed"]]
    source = FixedSceneCandidateSource(cfg, "scene", candidates, metadata=metadata)
    original_id = source.variants[0].variant_id
    movable.append("other")
    cfg.seed = 9
    cfg.max_candidates = 3
    cfg.movable_entity_ids = ("other",)
    changes.clear()
    candidates.clear()
    metadata[0][1] = "mutated"

    assert source.variants[0].variant_id == original_id
    assert source.variants[0].pose_changes == (_change(),)
    assert source.variants[0].metadata == (("source", "fixed"),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 0.5},
        {"max_candidates": 0},
        {"max_candidates": -1},
        {"max_candidates": True},
        {"movable_entity_ids": ("cube", "cube")},
        {"movable_entity_ids": ("",)},
        {"movable_entity_ids": "cube"},
        {"movable_entity_ids": None},
    ],
)
def test_config_rejects_invalid_budget_seed_and_permissions(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SceneExpansionCfg(**kwargs)


def test_config_detaches_mutable_permission_list() -> None:
    movable = ["cube"]
    cfg = SceneExpansionCfg(movable_entity_ids=movable)
    movable.append("target")
    assert cfg.movable_entity_ids == ("cube",)


def test_source_revalidates_config_after_in_place_updates() -> None:
    cfg = SceneExpansionCfg()
    cfg.max_candidates = -1
    with pytest.raises(ValueError, match="max_candidates"):
        FixedSceneCandidateSource(cfg, "scene", [()])


def test_source_rejects_unresolved_configuration() -> None:
    cfg = SceneExpansionCfg()
    cfg.seed = MISSING
    with pytest.raises(TypeError, match="seed"):
        FixedSceneCandidateSource(cfg, "scene", [()])


def test_source_rejects_changes_to_entities_not_declared_movable() -> None:
    with pytest.raises(ValueError, match="not declared movable.*target"):
        FixedSceneCandidateSource(
            SceneExpansionCfg(movable_entity_ids=("cube",)),
            "scene",
            [(_change("target"),)],
        )


def test_source_rejects_duplicate_entity_changes() -> None:
    with pytest.raises(ValueError, match="unique entity UIDs"):
        FixedSceneCandidateSource(
            SceneExpansionCfg(movable_entity_ids=("cube",)),
            "scene",
            [(_change(), _change(x=0.1))],
        )


def test_source_supports_empty_and_explicit_nominal_candidates() -> None:
    assert not tuple(FixedSceneCandidateSource(SceneExpansionCfg(), "scene", []))
    nominal = FixedSceneCandidateSource(SceneExpansionCfg(), "scene", [()])
    assert len(nominal) == 1
    assert nominal.variants[0].pose_changes == ()


def test_empty_source_still_rejects_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="metadata values"):
        FixedSceneCandidateSource(
            SceneExpansionCfg(), "scene", [], metadata=(("source", []),)
        )
