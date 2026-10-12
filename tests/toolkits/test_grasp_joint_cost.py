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

"""Prove joint-specific preferences change ranking without weakening limits."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tools.grasp_assemble._layout import motion_metrics, motion_rejection
from scripts.tools.grasp_assemble._path import select_ik_paths

_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
_WRIST_DISCOUNT = [1, 1, 1, 1, 1, 0.2]


def _competing_postures() -> np.ndarray:
    """Compare 0.1 rad in the shoulder with 0.15 rad in the last wrist joint."""
    postures = np.zeros((2, 6))
    postures[0, 0] = 0.1
    postures[1, 5] = 0.15
    return postures


@pytest.mark.parametrize("include_executed_edge", [True, False])
def test_discount_changes_path_and_initial_seed_rankings(
    include_executed_edge: bool,
) -> None:
    alternatives = _competing_postures()
    layers = [alternatives]
    if include_executed_edge:
        layers.insert(0, np.zeros((1, 6)))
    args = (layers, np.zeros(6), np.tile([-1, 1], (6, 1)), 12)
    uniform = select_ik_paths(*args)
    discounted = select_ik_paths(*args, joint_cost_weights=_WRIST_DISCOUNT)
    np.testing.assert_allclose(uniform[0][-1], alternatives[0], atol=1e-12)
    np.testing.assert_allclose(discounted[0][-1], alternatives[1], atol=1e-12)


def test_discount_changes_layout_score_but_preserves_raw_motion_metrics() -> None:
    shoulder = np.zeros((2, 6))
    shoulder[1, 0] = np.deg2rad(100)
    wrist = np.zeros((2, 6))
    wrist[1, 5] = np.deg2rad(200)
    shoulder_uniform = motion_metrics(shoulder, _JOINT_NAMES)
    wrist_uniform = motion_metrics(wrist, _JOINT_NAMES)
    shoulder_weighted = motion_metrics(
        shoulder, _JOINT_NAMES, joint_cost_weights=_WRIST_DISCOUNT
    )
    wrist_weighted = motion_metrics(
        wrist, _JOINT_NAMES, joint_cost_weights=_WRIST_DISCOUNT
    )
    assert shoulder_uniform["score"] < wrist_uniform["score"]
    assert wrist_weighted["score"] < shoulder_weighted["score"]
    for key in (
        "joint_names",
        "joint_travel_degrees",
        "joint_range_degrees",
        "total_joint_travel_degrees",
        "max_joint_travel_degrees",
        "max_joint_range_degrees",
        "max_joint_step_degrees",
    ):
        assert wrist_weighted[key] == wrist_uniform[key]
    assert wrist_weighted["joint_cost_weights"] == _WRIST_DISCOUNT
    assert wrist_weighted["weighted_total_joint_travel_degrees"] == pytest.approx(40)
    assert wrist_weighted["weighted_max_joint_range_degrees"] == pytest.approx(40)
    assert wrist_weighted["score"] == pytest.approx(120)


def test_zero_wrist_weight_does_not_relax_ik_step_limit() -> None:
    target = np.zeros((1, 6))
    target[0, 5] = np.deg2rad(20)
    assert (
        select_ik_paths(
            [np.zeros((1, 6)), target],
            np.zeros(6),
            np.tile([-1, 1], (6, 1)),
            12,
            joint_cost_weights=[1, 1, 1, 1, 1, 0],
        )
        == []
    )


def test_zero_wrist_weight_does_not_relax_physical_joint_limits() -> None:
    target = np.zeros((1, 6))
    target[0, 5] = 0.15
    assert (
        select_ik_paths(
            [np.zeros((1, 6)), target],
            np.zeros(6),
            np.tile([-0.1, 0.1], (6, 1)),
            12,
            joint_cost_weights=[1, 1, 1, 1, 1, 0],
        )
        == []
    )


def test_zero_wrist_weight_does_not_relax_layout_motion_limits() -> None:
    path = np.zeros((101, 6))
    path[:, 5] = np.deg2rad(np.linspace(0, 300, len(path)))
    metrics = motion_metrics(path, _JOINT_NAMES, joint_cost_weights=[1, 1, 1, 1, 1, 0])
    assert metrics["score"] == 0
    rejection = motion_rejection(metrics, {"layout": {"mode": "optimize"}})
    assert rejection is not None and "max_joint_range_degrees=300" in rejection


def test_zero_weight_is_valid_when_another_joint_has_positive_weight() -> None:
    target = np.zeros((1, 6))
    target[0, 5] = 0.1
    paths = select_ik_paths(
        [np.zeros((1, 6)), target],
        np.zeros(6),
        np.tile([-1, 1], (6, 1)),
        12,
        joint_cost_weights=[1, 1, 1, 1, 1, 0],
    )
    np.testing.assert_allclose(paths[0][-1], target[0], atol=1e-12)


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [1] * 5,
        [[1] * 6],
        [0] * 6,
        [1, 1, 1, 1, 1, -0.2],
        [1, 1, 1, 1, 1, float("nan")],
        [1, 1, 1, 1, 1, float("inf")],
        [1, 1, 1, 1, 1, True],
        [1, 1, 1, 1, 1, "0.2"],
        [1, 1, 1, 1, 1, 0.2 + 0j],
        [1, 1, 1, 1, 1, None],
        "uniform",
        1,
    ],
)
@pytest.mark.parametrize("helper", ["path", "metrics"])
def test_reject_invalid_weights(weights: object, helper: str) -> None:
    with pytest.raises(ValueError, match="joint_cost_weights"):
        if helper == "path":
            select_ik_paths(
                [np.zeros((1, 6))],
                np.zeros(6),
                np.tile([-1, 1], (6, 1)),
                12,
                joint_cost_weights=weights,
            )
        else:
            motion_metrics(np.zeros((2, 6)), _JOINT_NAMES, joint_cost_weights=weights)


def test_general_joint_count_keeps_uniform_default() -> None:
    path = np.array([[0, 0], [1, -2]])
    metrics = motion_metrics(path, ["shoulder", "wrist"])
    assert metrics["joint_cost_weights"] == [1, 1]
    assert metrics["score"] == pytest.approx(np.rad2deg(7))
