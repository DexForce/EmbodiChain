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

"""Pure-numeric tests for manipulability workspace visualization.

No simulation, renderer or robot asset is involved: every test exercises the
score-to-color mapping, the robust/log normalization, the joint-space and
Cartesian alignment contracts, the selected-subset ellipsoid path, and the
point-cloud forwarding contracts against a mocked simulation manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.workspace.configs import VisualizationType
from embodichain.lab.sim.motion.workspace.visualizers.manipulability_visualizer import (
    ManipulabilityColorCfg,
    ManipulabilityVisualizer,
    align_manipulability_scores,
    ellipsoid_surface,
    inspect_points,
    map_manipulability_colors,
    normalize_manipulability,
    select_inspection_indices,
    translational_manipulability_ellipsoid,
)


def _diagonal_jacobian(scales: list[float], dof: int = 6) -> np.ndarray:
    """Build one spatial Jacobian whose translational block is ``diag(scales)``."""
    jac = np.zeros((6, dof))
    for row, value in enumerate(scales):
        jac[row, row] = value
    # Fill the rotational block so the 6-row Jacobian stays full rank.
    for row in range(3, 6):
        jac[row, row] = 1.0
    return jac


class TestNormalization:
    def test_percentile_clip_saturates_outliers_but_keeps_raw_range(self):
        # 98 ones plus two extreme outliers: a min/max scale would push every
        # ordinary point into the bottom 1% of the color range.
        scores = np.concatenate([np.linspace(1.0, 2.0, 98), [0.0, 1000.0]])
        result = normalize_manipulability(scores, percentile_clip=(2.0, 98.0))

        assert result.raw_range == pytest.approx((0.0, 1000.0))
        # Clip bounds sit inside the raw range, so the bulk uses the full ramp.
        low, high = result.clip_range
        assert 0.0 < low < high < 1000.0
        assert result.normalized.min() == pytest.approx(0.0)
        assert result.normalized.max() == pytest.approx(1.0)
        assert result.num_clipped >= 2
        assert result.num_valid == 100

    def test_full_range_clip_is_plain_min_max(self):
        scores = np.array([0.0, 1.0, 4.0])
        result = normalize_manipulability(scores, percentile_clip=(0.0, 100.0))

        assert result.clip_range == pytest.approx((0.0, 4.0))
        assert result.normalized == pytest.approx([0.0, 0.25, 1.0])
        assert result.num_clipped == 0

    def test_log_scale_maps_geometric_midpoint_to_half(self):
        scores = np.array([1e-4, 1e-2, 1.0])
        result = normalize_manipulability(
            scores, percentile_clip=(0.0, 100.0), log_scale=True
        )

        assert result.log_scale is True
        # Raw range is reported untransformed; clip bounds come back in raw
        # units so a color bar can be labelled with real w values.
        assert result.raw_range == pytest.approx((1e-4, 1.0))
        assert result.clip_range == pytest.approx((1e-4, 1.0))
        assert result.normalized == pytest.approx([0.0, 0.5, 1.0])

    def test_linear_scale_compresses_what_log_scale_separates(self):
        scores = np.array([1e-6, 1e-3, 1.0])
        linear = normalize_manipulability(scores, percentile_clip=(0.0, 100.0))
        log = normalize_manipulability(
            scores, percentile_clip=(0.0, 100.0), log_scale=True
        )

        assert linear.normalized[1] < 1e-2  # indistinguishable from the minimum
        assert log.normalized[1] == pytest.approx(0.5)

    def test_log_scale_keeps_exact_zero_finite(self):
        result = normalize_manipulability(
            np.array([0.0, 1.0]), percentile_clip=(0.0, 100.0), log_scale=True
        )
        assert np.isfinite(result.normalized).all()
        assert result.normalized[0] == pytest.approx(0.0)
        assert result.raw_range[0] == 0.0

    def test_empty_input(self):
        result = normalize_manipulability(np.empty(0))
        assert result.normalized.shape == (0,)
        assert np.isnan(result.raw_range).all()
        assert np.isnan(result.clip_range).all()
        assert result.num_valid == 0

    def test_single_point_maps_to_mid_scale(self):
        result = normalize_manipulability(np.array([0.42]))
        assert result.normalized == pytest.approx([0.5])
        assert result.raw_range == pytest.approx((0.42, 0.42))

    def test_all_equal_scores_map_to_mid_scale(self):
        result = normalize_manipulability(np.full(7, 3.0))
        assert result.normalized == pytest.approx(np.full(7, 0.5))
        assert result.clip_range == pytest.approx((3.0, 3.0))

    def test_nan_entries_are_excluded_and_stay_nan(self):
        scores = np.array([1.0, np.nan, 3.0])
        result = normalize_manipulability(scores, percentile_clip=(0.0, 100.0))

        assert np.isnan(result.normalized[1])
        assert result.raw_range == pytest.approx((1.0, 3.0))
        assert result.num_valid == 2

    def test_all_nan_input_yields_nan_ranges(self):
        result = normalize_manipulability(np.full(4, np.nan))
        assert np.isnan(result.normalized).all()
        assert np.isnan(result.raw_range).all()
        assert result.num_valid == 0

    def test_valid_mask_restricts_statistics(self):
        scores = np.array([1.0, 2.0, 100.0])
        mask = np.array([True, True, False])
        result = normalize_manipulability(
            scores, percentile_clip=(0.0, 100.0), valid_mask=mask
        )

        assert result.raw_range == pytest.approx((1.0, 2.0))
        assert np.isnan(result.normalized[2])

    @pytest.mark.parametrize("clip", [(50.0, 50.0), (-1.0, 90.0), (10.0, 101.0)])
    def test_invalid_percentiles_raise(self, clip):
        with pytest.raises(ValueError, match="percentile_clip"):
            normalize_manipulability(np.array([1.0, 2.0]), percentile_clip=clip)

    def test_non_positive_log_epsilon_raises(self):
        with pytest.raises(ValueError, match="log_epsilon"):
            normalize_manipulability(np.array([1.0]), log_epsilon=0.0)

    def test_accepts_torch_input(self):
        result = normalize_manipulability(
            torch.tensor([1.0, 2.0, 3.0]), percentile_clip=(0.0, 100.0)
        )
        assert result.normalized == pytest.approx([0.0, 0.5, 1.0])


class TestColorMapping:
    def test_unreachable_points_are_visually_distinct(self):
        cfg = ManipulabilityColorCfg(percentile_clip=(0.0, 100.0))
        mapping = map_manipulability_colors(
            np.array([0.1, np.nan, 0.9]),
            reachable_mask=np.array([True, False, True]),
            cfg=cfg,
        )

        assert mapping.reachable_mask.tolist() == [True, False, True]
        assert mapping.colors[1] == pytest.approx(np.asarray(cfg.unreachable_color))
        assert mapping.sizes[1] == pytest.approx(cfg.unreachable_point_size)
        assert mapping.sizes[0] == pytest.approx(cfg.point_size)
        # Reachable colors differ from the unreachable color and from each other.
        assert not np.allclose(mapping.colors[0], mapping.colors[1])
        assert not np.allclose(mapping.colors[0], mapping.colors[2])

    def test_nan_score_overrides_a_true_reachable_flag(self):
        # A point flagged reachable but carrying no score cannot be colored by
        # manipulability; it must fall back to the unreachable styling.
        mapping = map_manipulability_colors(
            np.array([1.0, np.nan]), reachable_mask=np.array([True, True])
        )
        assert mapping.reachable_mask.tolist() == [True, False]

    def test_mapping_is_monotonic_in_score(self):
        scores = np.linspace(0.1, 1.0, 12)
        mapping = map_manipulability_colors(
            scores, cfg=ManipulabilityColorCfg(percentile_clip=(0.0, 100.0))
        )
        normalized = mapping.normalization.normalized
        assert np.all(np.diff(normalized) > 0)
        assert normalized[0] == pytest.approx(0.0)
        assert normalized[-1] == pytest.approx(1.0)

    def test_raw_range_is_exposed_through_the_mapping(self):
        mapping = map_manipulability_colors(np.array([0.25, 4.0, np.nan]))
        assert mapping.raw_range == pytest.approx((0.25, 4.0))
        assert mapping.clip_range == mapping.normalization.clip_range

    def test_log_cfg_propagates_to_normalization(self):
        cfg = ManipulabilityColorCfg(log_scale=True, percentile_clip=(0.0, 100.0))
        mapping = map_manipulability_colors(np.array([1e-4, 1e-2, 1.0]), cfg=cfg)
        assert mapping.normalization.log_scale is True
        assert mapping.normalization.normalized == pytest.approx([0.0, 0.5, 1.0])

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="reachable_mask"):
            map_manipulability_colors(
                np.array([1.0, 2.0]), reachable_mask=np.array([True])
            )

    def test_all_unreachable_returns_uniform_styling(self):
        cfg = ManipulabilityColorCfg()
        mapping = map_manipulability_colors(np.full(3, np.nan), cfg=cfg)
        assert not mapping.reachable_mask.any()
        assert np.allclose(mapping.colors, np.asarray(cfg.unreachable_color))
        assert np.isnan(mapping.raw_range).all()


class TestAlignment:
    @staticmethod
    def _joint_space_result(num_points: int = 5) -> dict:
        points = np.arange(num_points * 3, dtype=float).reshape(num_points, 3)
        return {
            "mode": "joint_space",
            "workspace_points": points,
            "joint_configurations": np.zeros((num_points, 6)),
            "manipulability_scores": np.linspace(0.1, 0.5, num_points),
        }

    @staticmethod
    def _cartesian_result(retain_diagnostics: bool = True) -> dict:
        all_points = np.arange(18, dtype=float).reshape(6, 3)
        mask = np.array([False, True, False, True, True, False])
        reachable = all_points[mask]
        scores = np.array([0.7, 0.2, 0.5])
        result = {
            "mode": "cartesian_space",
            "reachable_points": reachable,
            "joint_configurations": np.tile(np.arange(6.0), (3, 1)),
            "manipulability_scores": scores,
        }
        if retain_diagnostics:
            result["all_points"] = all_points
            result["workspace_points"] = all_points
            result["reachability_mask"] = mask
        return result

    def test_joint_space_scores_align_one_to_one(self):
        result = self._joint_space_result()
        aligned = align_manipulability_scores(result)

        assert aligned.points.shape == (5, 3)
        assert aligned.reachable_mask.all()
        assert aligned.scores == pytest.approx(result["manipulability_scores"])
        assert aligned.score_indices.tolist() == [0, 1, 2, 3, 4]
        assert aligned.num_unreachable == 0

    def test_joint_space_length_mismatch_raises(self):
        result = self._joint_space_result()
        result["manipulability_scores"] = np.array([0.1, 0.2])
        result["joint_configurations"] = np.zeros((2, 6))
        with pytest.raises(ValueError, match="workspace_points"):
            align_manipulability_scores(result)

    def test_joint_configuration_length_mismatch_raises(self):
        result = self._joint_space_result()
        result["joint_configurations"] = np.zeros((2, 6))
        with pytest.raises(ValueError, match="joint_configurations"):
            align_manipulability_scores(result)

    def test_cartesian_scores_scatter_onto_reachable_positions(self):
        result = self._cartesian_result()
        aligned = align_manipulability_scores(result)

        assert aligned.points.shape == (6, 3)
        assert aligned.reachable_mask.tolist() == [
            False,
            True,
            False,
            True,
            True,
            False,
        ]
        # Reachable rows keep the original score order; the rest stay NaN.
        assert aligned.scores[[1, 3, 4]] == pytest.approx([0.7, 0.2, 0.5])
        assert np.isnan(aligned.scores[[0, 2, 5]]).all()
        assert aligned.score_indices.tolist() == [-1, 0, -1, 1, 2, -1]
        assert aligned.num_reachable == 3
        assert aligned.num_unreachable == 3

    def test_scattered_scores_stay_attached_to_their_positions(self):
        result = self._cartesian_result()
        aligned = align_manipulability_scores(result)
        for row, point in enumerate(result["reachable_points"]):
            position = int(np.flatnonzero(aligned.score_indices == row)[0])
            assert aligned.points[position] == pytest.approx(point)
            assert aligned.scores[position] == pytest.approx(
                result["manipulability_scores"][row]
            )

    def test_cartesian_without_diagnostics_uses_reachable_points_only(self):
        result = self._cartesian_result(retain_diagnostics=False)
        aligned = align_manipulability_scores(result)

        assert aligned.points.shape == (3, 3)
        assert aligned.reachable_mask.all()
        assert aligned.scores == pytest.approx([0.7, 0.2, 0.5])
        assert aligned.points == pytest.approx(result["reachable_points"])

    def test_include_unreachable_false_drops_the_full_sample_set(self):
        result = self._cartesian_result()
        aligned = align_manipulability_scores(result, include_unreachable=False)
        assert aligned.points.shape == (3, 3)
        assert aligned.reachable_mask.all()

    def test_reordered_all_points_are_rejected(self):
        result = self._cartesian_result()
        # Same mask cardinality, but the reachable rows no longer line up.
        result["all_points"] = result["all_points"][::-1].copy()
        with pytest.raises(ValueError, match="does not match reachable_points"):
            align_manipulability_scores(result)

    def test_mask_cardinality_mismatch_raises(self):
        result = self._cartesian_result()
        result["reachability_mask"] = np.array([True, True, False, True, True, False])
        with pytest.raises(ValueError, match="reachability_mask selects"):
            align_manipulability_scores(result)

    def test_reachable_points_length_mismatch_raises(self):
        result = self._cartesian_result(retain_diagnostics=False)
        result["manipulability_scores"] = np.array([0.1, 0.2])
        result["joint_configurations"] = np.zeros((2, 6))
        with pytest.raises(ValueError, match="reachable_points"):
            align_manipulability_scores(result)

    def test_missing_scores_raise(self):
        result = self._joint_space_result()
        result.pop("manipulability_scores")
        with pytest.raises(ValueError, match="no manipulability scores"):
            align_manipulability_scores(result)

    def test_explicit_scores_override_the_result_entry(self):
        result = self._joint_space_result()
        aligned = align_manipulability_scores(result, scores=np.full(5, 2.0))
        assert aligned.scores == pytest.approx(np.full(5, 2.0))

    def test_torch_results_are_accepted(self):
        result = self._cartesian_result()
        torch_result = {
            key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
            for key, value in result.items()
        }
        aligned = align_manipulability_scores(torch_result)
        assert aligned.scores[[1, 3, 4]] == pytest.approx([0.7, 0.2, 0.5])


class TestEllipsoid:
    def test_semi_axes_equal_translational_singular_values(self):
        jac = torch.as_tensor(
            np.stack([_diagonal_jacobian([3.0, 2.0, 0.5])]), dtype=torch.float64
        )
        radii, axes = translational_manipulability_ellipsoid(jac)

        assert radii.shape == (1, 3)
        assert radii[0] == pytest.approx([3.0, 2.0, 0.5])
        # Principal directions of a diagonal block are the coordinate axes.
        assert np.abs(axes[0]) == pytest.approx(np.eye(3), abs=1e-9)

    def test_rotational_rows_are_ignored(self):
        translational = _diagonal_jacobian([1.0, 1.0, 1.0])
        inflated = translational.copy()
        inflated[3:, :] *= 1000.0  # huge angular rows must not change the radii
        radii_a, _ = translational_manipulability_ellipsoid(translational[None])
        radii_b, _ = translational_manipulability_ellipsoid(inflated[None])
        assert radii_a == pytest.approx(radii_b)

    def test_radii_are_sorted_descending(self):
        rng = np.random.default_rng(0)
        jac = rng.normal(size=(4, 6, 7))
        radii, _ = translational_manipulability_ellipsoid(jac)
        assert np.all(np.diff(radii, axis=1) <= 1e-12)

    def test_axes_match_radii_ordering(self):
        # J_t maps the unit sphere onto the ellipsoid: the leading axis must be
        # the direction of maximal end-effector velocity.
        rng = np.random.default_rng(3)
        jac = rng.normal(size=(1, 6, 6))
        radii, axes = translational_manipulability_ellipsoid(jac)
        translational = jac[0, :3, :]
        gram = translational @ translational.T
        for column in range(3):
            direction = axes[0, :, column]
            assert gram @ direction == pytest.approx(
                radii[0, column] ** 2 * direction, abs=1e-8
            )

    def test_singular_posture_gives_a_flat_ellipsoid_without_nan(self):
        jac = _diagonal_jacobian([1.0, 1.0, 0.0])[None]
        radii, _ = translational_manipulability_ellipsoid(jac)
        assert np.isfinite(radii).all()
        assert radii[0, -1] == pytest.approx(0.0, abs=1e-9)

    def test_surface_points_satisfy_the_ellipsoid_equation(self):
        radii = np.array([3.0, 2.0, 0.5])
        axes = np.eye(3)
        x, y, z = ellipsoid_surface(radii, axes, resolution=9)
        residual = (x / 3.0) ** 2 + (y / 2.0) ** 2 + (z / 0.5) ** 2
        assert residual == pytest.approx(np.ones_like(residual), abs=1e-9)

    def test_surface_is_translated_by_the_center(self):
        center = np.array([1.0, -2.0, 0.5])
        x, y, z = ellipsoid_surface(np.ones(3), np.eye(3), center=center, resolution=9)
        # Every tessellated vertex must lie on the unit sphere around `center`,
        # which holds exactly at any resolution.
        residual = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        assert residual == pytest.approx(np.ones_like(residual), abs=1e-9)


class TestSelectionAndInspection:
    scores = np.array([0.5, np.nan, 0.1, 0.9, 0.3])

    def test_top_and_bottom_selection(self):
        selection = select_inspection_indices(self.scores, top_k=1, bottom_k=1)
        assert selection.indices.tolist() == [3, 2]
        assert selection.labels == ("top", "bottom")

    def test_explicit_selection_takes_priority_and_deduplicates(self):
        selection = select_inspection_indices(
            self.scores, selected=[3], top_k=1, bottom_k=1
        )
        assert selection.indices.tolist() == [3, 2]
        assert selection.labels == ("selected", "bottom")

    def test_unreachable_points_are_never_selected(self):
        selection = select_inspection_indices(self.scores, top_k=5, bottom_k=5)
        assert 1 not in selection.indices.tolist()
        assert len(selection) == 4

    def test_selecting_an_unreachable_point_raises(self):
        with pytest.raises(ValueError, match="no manipulability score"):
            select_inspection_indices(self.scores, selected=[1])

    def test_out_of_range_selection_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            select_inspection_indices(self.scores, selected=[99])

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            select_inspection_indices(self.scores, top_k=-1)

    def test_ties_break_on_ascending_index(self):
        selection = select_inspection_indices(
            np.array([1.0, 1.0, 1.0]), top_k=1, bottom_k=1
        )
        assert selection.indices.tolist() == [2, 0]

    def test_zero_counts_return_an_empty_selection(self):
        selection = select_inspection_indices(self.scores, top_k=0, bottom_k=0)
        assert len(selection) == 0

    def test_ellipsoids_are_computed_for_the_selected_subset_only(self):
        num_points = 400
        rng = np.random.default_rng(7)
        points = rng.normal(size=(num_points, 3))
        scores = rng.uniform(0.05, 1.0, size=num_points)
        qpos = rng.normal(size=(num_points, 6))
        calls: list[int] = []

        def jacobian_fn(selected_qpos: np.ndarray) -> np.ndarray:
            calls.append(len(selected_qpos))
            return np.stack([_diagonal_jacobian([3.0, 2.0, 1.0])] * len(selected_qpos))

        selection = select_inspection_indices(scores, top_k=2, bottom_k=2)
        inspections = inspect_points(
            selection,
            points=points,
            scores=scores,
            jacobian_fn=jacobian_fn,
            joint_configurations=qpos,
        )

        # One batched call, sized by the selection rather than the point set.
        assert calls == [4]
        assert sum(calls) == len(selection) < num_points
        assert len(inspections) == 4
        assert all(item.ellipsoid_radii is not None for item in inspections)

    def test_inspection_carries_score_condition_and_qpos(self):
        points = np.eye(3)
        scores = np.array([0.4, 0.8, 0.2])
        qpos = np.arange(18.0).reshape(3, 6)
        selection = select_inspection_indices(scores, top_k=1, bottom_k=1)
        inspections = inspect_points(
            selection,
            points=points,
            scores=scores,
            jacobian_fn=lambda q: np.stack(
                [_diagonal_jacobian([4.0, 2.0, 1.0])] * len(q)
            ),
            joint_configurations=qpos,
        )

        best = inspections[0]
        assert best.index == 1
        assert best.label == "top"
        assert best.manipulability == pytest.approx(0.8)
        assert best.position == pytest.approx(points[1])
        assert best.joint_configuration == pytest.approx(qpos[1])
        assert best.condition_number == pytest.approx(4.0)
        assert best.ellipsoid_radii == pytest.approx([4.0, 2.0, 1.0])
        assert best.anisotropy == pytest.approx(4.0)
        assert any("cond(J)" in line for line in best.summary_lines())
        assert any("qpos" in line for line in best.summary_lines())

    def test_score_indices_route_to_the_right_joint_configuration(self):
        # Cartesian layout: 4 drawn points, 2 of them scored.
        points = np.arange(12.0).reshape(4, 3)
        scores = np.array([np.nan, 0.9, np.nan, 0.2])
        score_indices = np.array([-1, 0, -1, 1])
        qpos = np.array([[1.0] * 6, [2.0] * 6])
        inspections = inspect_points(
            select_inspection_indices(scores, top_k=1, bottom_k=1),
            points=points,
            scores=scores,
            jacobian_fn=lambda q: np.stack(
                [_diagonal_jacobian([1.0, 1.0, 1.0])] * len(q)
            ),
            joint_configurations=qpos,
            score_indices=score_indices,
        )
        assert [item.index for item in inspections] == [1, 3]
        assert inspections[0].joint_configuration == pytest.approx(qpos[0])
        assert inspections[1].joint_configuration == pytest.approx(qpos[1])

    def test_ellipsoid_can_be_skipped(self):
        scores = np.array([0.4, 0.8])
        inspections = inspect_points(
            select_inspection_indices(scores, top_k=1, bottom_k=0),
            points=np.zeros((2, 3)),
            scores=scores,
            jacobian_fn=lambda q: np.stack(
                [_diagonal_jacobian([1.0, 1.0, 1.0])] * len(q)
            ),
            joint_configurations=np.zeros((2, 6)),
            with_ellipsoid=False,
        )
        assert inspections[0].ellipsoid_radii is None
        assert inspections[0].condition_number is not None

    def test_missing_joint_configurations_raise(self):
        scores = np.array([0.4, 0.8])
        with pytest.raises(ValueError, match="joint_configurations are required"):
            inspect_points(
                select_inspection_indices(scores, top_k=1, bottom_k=0),
                points=np.zeros((2, 3)),
                scores=scores,
                jacobian_fn=lambda q: np.zeros((len(q), 6, 6)),
            )

    def test_wrong_jacobian_batch_size_raises(self):
        scores = np.array([0.4, 0.8])
        with pytest.raises(ValueError, match="jacobian_fn must return"):
            inspect_points(
                select_inspection_indices(scores, top_k=1, bottom_k=0),
                points=np.zeros((2, 3)),
                scores=scores,
                jacobian_fn=lambda q: np.zeros((len(q) + 1, 6, 6)),
                joint_configurations=np.zeros((2, 6)),
            )

    def test_no_jacobian_fn_still_reports_score_and_qpos(self):
        scores = np.array([0.4, 0.8])
        inspections = inspect_points(
            [0],
            points=np.zeros((2, 3)),
            scores=scores,
            joint_configurations=np.arange(12.0).reshape(2, 6),
        )
        assert inspections[0].manipulability == pytest.approx(0.4)
        assert inspections[0].joint_configuration == pytest.approx(np.arange(6.0))
        assert inspections[0].ellipsoid_radii is None
        assert inspections[0].condition_number is None


class TestVisualizerDataBackend:
    def test_type_name_is_registered(self):
        visualizer = ManipulabilityVisualizer(backend="data")
        assert visualizer.get_type_name() == VisualizationType.MANIPULABILITY.value

    def test_data_backend_exposes_ranges_and_mask(self):
        points = np.arange(12.0).reshape(4, 3)
        scores = np.array([0.1, np.nan, 0.5, 1.0])
        visualizer = ManipulabilityVisualizer(
            backend="data",
            color_cfg=ManipulabilityColorCfg(percentile_clip=(0.0, 100.0)),
        )
        data = visualizer.visualize(points, scores=scores)

        assert data["colors"].shape == (4, 4)
        assert data["reachable_mask"].tolist() == [True, False, True, True]
        assert data["raw_range"] == pytest.approx((0.1, 1.0))
        assert data["clip_range"] == pytest.approx((0.1, 1.0))
        assert data["log_scale"] is False

    def test_point_set_argument_supplies_points_and_scores(self):
        result = TestAlignment._cartesian_result()
        aligned = align_manipulability_scores(result)
        visualizer = ManipulabilityVisualizer(backend="data")
        data = visualizer.visualize(aligned.points, point_set=aligned)

        assert data["points"].shape == aligned.points.shape
        assert data["reachable_mask"].tolist() == aligned.reachable_mask.tolist()

    def test_missing_scores_raise(self):
        visualizer = ManipulabilityVisualizer(backend="data")
        with pytest.raises(ValueError, match="scores"):
            visualizer.visualize(np.zeros((3, 3)))

    def test_compact_scores_are_scattered_through_the_reachability_mask(self):
        # The generic analyzer path draws all sampled points but only holds one
        # score per reachable point.
        points = np.arange(15.0).reshape(5, 3)
        mask = np.array([False, True, True, False, True])
        visualizer = ManipulabilityVisualizer(
            backend="data",
            scores=np.array([0.2, 0.4, 0.6]),
            reachable_mask=mask,
            color_cfg=ManipulabilityColorCfg(percentile_clip=(0.0, 100.0)),
        )
        data = visualizer.visualize(points, colors=np.zeros((5, 3)))

        assert data["reachable_mask"].tolist() == mask.tolist()
        assert data["raw_range"] == pytest.approx((0.2, 0.6))
        # Reachability colors passed by the generic caller must not win over the
        # manipulability mapping.
        assert not np.allclose(data["colors"][:, :3], 0.0)

    def test_scores_win_over_supplied_colors(self):
        visualizer = ManipulabilityVisualizer(backend="data")
        data = visualizer.visualize(
            np.zeros((2, 3)),
            colors=np.zeros((2, 3)),
            scores=np.array([0.1, 0.9]),
        )
        assert "raw_range" in data
        assert not np.allclose(data["colors"][:, :3], 0.0)

    def test_unalignable_scores_raise(self):
        visualizer = ManipulabilityVisualizer(backend="data")
        with pytest.raises(ValueError, match="cannot be aligned"):
            visualizer.visualize(np.zeros((5, 3)), scores=np.array([0.1, 0.2]))

    def test_generic_sizes_argument_is_ignored(self):
        # ``WorkspaceAnalyzer.visualize`` passes reachability sizes positionally
        # to every visualizer; the manipulability sizes come from the mapping.
        visualizer = ManipulabilityVisualizer(backend="data")
        data = visualizer.visualize(
            np.zeros((2, 3)), scores=np.array([0.1, 0.9]), sizes=np.array([1.0, 2.0])
        )
        assert data["sizes"].tolist() == [
            visualizer.color_cfg.point_size,
            visualizer.color_cfg.point_size,
        ]

    def test_unknown_backend_raises(self):
        visualizer = ManipulabilityVisualizer(backend="nope")
        with pytest.raises(ValueError, match="Unsupported backend"):
            visualizer.visualize(np.zeros((2, 3)), scores=np.array([1.0, 2.0]))

    def test_inspection_rendering_rejects_non_matplotlib_backends(self):
        visualizer = ManipulabilityVisualizer(backend="open3d")
        with pytest.raises(ValueError, match="matplotlib"):
            visualizer.visualize_inspection([object()])

    def test_empty_inspection_raises(self):
        visualizer = ManipulabilityVisualizer(backend="data")
        with pytest.raises(ValueError, match="no inspections"):
            visualizer.visualize_inspection([])


class TestVisualizerPointCloudBackends:
    """Forwarding contracts for the ``sim_manager`` and ``viser`` backends.

    Both backends receive plain RGB, so the reachability styling that the
    Matplotlib figure expresses with per-point alpha and marker size has to
    survive as color alone. These tests pin that, and pin that a compact
    reachable-only score vector stays attached to the positions it belongs to
    once the mapping is handed to a point-cloud renderer.
    """

    @staticmethod
    def _aligned_point_set():
        """Align a Cartesian result whose middle samples were rejected by IK."""
        return align_manipulability_scores(TestAlignment._cartesian_result())

    def test_sim_manager_backend_forwards_rgb_and_point_size(self):
        sim = MagicMock()
        point_set = self._aligned_point_set()
        visualizer = ManipulabilityVisualizer(
            backend="sim_manager",
            sim_manager=sim,
            control_part_name="arm",
            color_cfg=ManipulabilityColorCfg(point_size=3.0),
        )

        visualizer.visualize(point_set.points, point_set=point_set)

        sim.set_visualization_overlays.assert_not_called()
        forwarded = sim.visualize_point_cloud.call_args.kwargs
        assert forwarded["point_size"] == 3.0
        assert forwarded["name"] == "workspace_pcd_arm"
        # The native renderer ignores alpha, so the RGBA mapping must be cut
        # down to three channels rather than passed through.
        assert forwarded["colors"].shape == (len(point_set.points), 3)
        np.testing.assert_allclose(forwarded["points"], point_set.points)

    def test_viser_backend_publishes_one_uint8_overlay(self):
        sim = MagicMock()
        point_set = self._aligned_point_set()
        visualizer = ManipulabilityVisualizer(
            backend="viser",
            sim_manager=sim,
            control_part_name="arm",
            color_cfg=ManipulabilityColorCfg(point_size=0.006),
        )

        visualizer.visualize(point_set.points, point_set=point_set)

        sim.visualize_point_cloud.assert_not_called()
        overlays = sim.set_visualization_overlays.call_args.args[0]
        assert len(overlays.point_clouds) == 1
        overlay = overlays.point_clouds[0]
        assert overlay.overlay_id == "workspace_arm"
        assert overlay.point_size == 0.006
        assert overlay.colors.dtype == np.uint8
        np.testing.assert_allclose(overlay.points, point_set.points, atol=1e-6)

    def test_forwarding_backends_require_a_sim_manager(self):
        visualizer = ManipulabilityVisualizer(backend="sim_manager")
        with pytest.raises(ValueError, match="sim_manager is required"):
            visualizer.visualize(np.zeros((2, 3)), scores=np.array([0.1, 0.9]))

    @pytest.mark.parametrize("backend", ["sim_manager", "viser"])
    def test_rejected_samples_do_not_shift_scores_onto_other_positions(
        self, backend: str
    ):
        # ``_cartesian_result`` rejects samples 0, 2 and 5 and stores the
        # remaining scores compactly as (0.7, 0.2, 0.5). Scattering them back
        # must land the highest score on point 1 and the lowest on point 3.
        sim = MagicMock()
        point_set = self._aligned_point_set()
        color_cfg = ManipulabilityColorCfg(percentile_clip=(0.0, 100.0))
        visualizer = ManipulabilityVisualizer(
            backend=backend,
            sim_manager=sim,
            control_part_name="arm",
            color_cfg=color_cfg,
        )

        visualizer.visualize(point_set.points, point_set=point_set)

        if backend == "sim_manager":
            forwarded = sim.visualize_point_cloud.call_args.kwargs
            points, rgb = forwarded["points"], np.asarray(forwarded["colors"])
        else:
            overlay = sim.set_visualization_overlays.call_args.args[0].point_clouds[0]
            points, rgb = overlay.points, overlay.colors / 255.0

        np.testing.assert_allclose(points, point_set.points, atol=1e-6)
        gray = np.asarray(color_cfg.unreachable_color[:3])
        for index, reachable in enumerate(point_set.reachable_mask):
            if reachable:
                assert not np.allclose(rgb[index], gray, atol=1 / 255)
            else:
                np.testing.assert_allclose(rgb[index], gray, atol=1 / 255)
        # Low viridis is blue/purple and high viridis is yellow. Reading the
        # channel order back proves the scores kept their own points instead of
        # sliding across the two gray rows that separate them.
        assert rgb[3, 2] > rgb[3, 1]
        assert rgb[1, 1] > rgb[1, 2]
