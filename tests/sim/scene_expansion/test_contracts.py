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

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from embodichain.lab.sim.scene_expansion import ScenePoseChange, SceneVariant


def _pose(x: float = 0.0) -> list[list[float]]:
    # Non-symmetric yaw detects row/column transposition and reflections.
    angle = math.pi / 3
    return [
        [math.cos(angle), -math.sin(angle), 0.0, x],
        [math.sin(angle), math.cos(angle), 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _variant(**changes: object) -> SceneVariant:
    fields = dict(
        parent_scene_id="scene:revision-1",
        seed=7,
        ordinal=0,
        pose_changes=(ScenePoseChange("cube", _pose()),),
        metadata=(("source", "handwritten"),),
    )
    fields.update(changes)
    return SceneVariant(**fields)


def test_pose_is_copied_into_immutable_arena_frame_matrix() -> None:
    matrix = _pose(0.3)
    change = ScenePoseChange("cube", matrix)
    matrix[0][3] = 9.0

    assert change.arena_pose[0][3] == pytest.approx(0.3)
    assert change.arena_pose[0][1] == pytest.approx(-math.sin(math.pi / 3))
    with pytest.raises(TypeError):
        change.arena_pose[0][3] = 2.0
    with pytest.raises(FrozenInstanceError):
        change.entity_uid = "different"


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0] * 3 for _ in range(3)],
        [[0.0] * 4 for _ in range(4)],
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, math.inf], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, math.nan], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, "0"], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.1, 0, 0, 1]],
    ],
)
def test_pose_rejects_nonrigid_or_nonfinite_matrices(matrix: object) -> None:
    with pytest.raises(ValueError, match="arena_pose"):
        ScenePoseChange("cube", matrix)


@pytest.mark.parametrize("uid", ["", " cube", "cube ", None])
def test_pose_requires_stable_nonempty_entity_uid(uid: object) -> None:
    with pytest.raises(ValueError, match="entity_uid"):
        ScenePoseChange(uid, _pose())


def test_variant_identity_is_canonical_and_metadata_is_detached() -> None:
    changes = [ScenePoseChange("z", _pose()), ScenePoseChange("a", _pose(0.4))]
    metadata = [["revision", "v1"], ["source", "fixed"]]
    first = _variant(pose_changes=changes, metadata=metadata)
    equivalent = _variant(pose_changes=reversed(changes), metadata=reversed(metadata))
    changes.clear()
    metadata[0][1] = "changed"

    assert first.variant_id == equivalent.variant_id
    assert tuple(change.entity_uid for change in first.pose_changes) == ("a", "z")
    assert first.metadata == (("revision", "v1"), ("source", "fixed"))
    with pytest.raises(FrozenInstanceError):
        first.ordinal = 4


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_scene_id": "scene:revision-2"},
        {"seed": 8},
        {"ordinal": 1},
        {"pose_changes": (ScenePoseChange("cube", _pose(0.1)),)},
        {"metadata": (("source", "generated"),)},
    ],
)
def test_variant_identity_captures_proposal_inputs(changes: dict[str, object]) -> None:
    original = _variant()
    assert replace(original, **changes).variant_id != original.variant_id


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_scene_id": ""},
        {"seed": True},
        {"seed": -1},
        {"ordinal": -1},
        {"ordinal": 0.5},
        {"pose_changes": ("cube",)},
        {"pose_changes": (ScenePoseChange("cube", _pose()),) * 2},
        {"metadata": (("source", "a"), ("source", "b"))},
        {"metadata": (("source", []),)},
        {"metadata": ("ab",)},
        {"metadata": (("source",),)},
    ],
)
def test_variant_rejects_ambiguous_or_mutable_inputs(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _variant(**changes)


def test_nominal_variant_has_explicit_proposal_identity() -> None:
    nominal = _variant(pose_changes=())
    assert nominal.pose_changes == ()
    assert nominal.variant_id != _variant().variant_id
