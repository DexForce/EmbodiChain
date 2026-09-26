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
"""Numerical tests for the shared manipulability compute helpers."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.kinematics import (
    condition_number,
    select_jacobian_rows,
    yoshikawa_manipulability,
)


class TestYoshikawa:
    def test_identity_block_is_unit_volume(self):
        # J = [I3 | 0] -> J J^T = I3 -> sqrt(det) = 1.
        jac = torch.zeros(4, 3, 7)
        jac[:, :3, :3] = torch.eye(3)
        out = yoshikawa_manipulability(jac)
        assert torch.allclose(out, torch.ones(4), atol=1e-6)

    def test_matches_closed_form_scaling(self):
        # J = diag(a, b, c) padded -> sqrt(det(J J^T)) = |a*b*c|.
        jac = torch.zeros(1, 3, 5)
        jac[0, 0, 0], jac[0, 1, 1], jac[0, 2, 2] = 2.0, 3.0, 4.0
        out = yoshikawa_manipulability(jac)
        assert abs(out.item() - 24.0) < 1e-5

    def test_singular_is_zero_not_nan(self):
        # Rank-deficient Jacobian: two identical rows -> det(J J^T) = 0.
        jac = torch.zeros(1, 3, 4)
        jac[0, 0, 0] = 1.0
        jac[0, 1, 0] = 1.0  # duplicate of row 0
        jac[0, 2, 1] = 1.0
        out = yoshikawa_manipulability(jac)
        assert torch.isfinite(out).all()
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_preserves_dtype_and_device(self):
        jac = torch.randn(3, 3, 6, dtype=torch.float64)
        out = yoshikawa_manipulability(jac)
        assert out.shape == (3,)
        assert out.dtype == torch.float64
        assert out.device == jac.device

    def test_deterministic_ranking(self):
        # A well-conditioned posture must outrank a near-singular one, and the
        # ranking must be reproducible across calls.
        good = torch.zeros(1, 3, 6)
        good[0, :3, :3] = torch.eye(3)
        bad = torch.zeros(1, 3, 6)
        bad[0, 0, 0], bad[0, 1, 1], bad[0, 2, 2] = 1.0, 1.0, 1e-4
        batch = torch.cat([bad, good], dim=0)
        first = yoshikawa_manipulability(batch)
        second = yoshikawa_manipulability(batch)
        assert torch.equal(first, second)
        assert int(torch.argmax(first)) == 1

    def test_rejects_unbatched(self):
        with pytest.raises(ValueError):
            yoshikawa_manipulability(torch.eye(3))


class TestConditionNumber:
    def test_identity_is_one(self):
        jac = torch.zeros(2, 3, 5)
        jac[:, :3, :3] = torch.eye(3)
        out = condition_number(jac)
        assert torch.allclose(out, torch.ones(2), atol=1e-6)

    def test_anisotropic_ratio(self):
        jac = torch.zeros(1, 3, 4)
        jac[0, 0, 0], jac[0, 1, 1], jac[0, 2, 2] = 10.0, 1.0, 2.0
        out = condition_number(jac)
        assert abs(out.item() - 10.0) < 1e-4

    def test_singular_is_large_finite(self):
        jac = torch.zeros(1, 3, 4)
        jac[0, 0, 0] = 1.0
        jac[0, 1, 0] = 1.0
        jac[0, 2, 1] = 1.0
        out = condition_number(jac)
        assert torch.isfinite(out).all()
        assert out.item() > 1e6


class TestSelectRows:
    def test_translational_and_rotational(self):
        jac = torch.arange(6 * 7, dtype=torch.float32).reshape(1, 6, 7)
        trans = select_jacobian_rows(jac, "translational")
        rot = select_jacobian_rows(jac, "rotational")
        assert trans.shape == (1, 3, 7)
        assert torch.equal(trans[0], jac[0, :3])
        assert torch.equal(rot[0], jac[0, 3:])

    def test_all_is_identity(self):
        jac = torch.randn(2, 6, 7)
        assert torch.equal(select_jacobian_rows(jac, "all"), jac)

    def test_custom_indices(self):
        jac = torch.randn(2, 6, 7)
        sub = select_jacobian_rows(jac, [0, 2, 4])
        assert sub.shape == (2, 3, 7)
        assert torch.equal(sub[:, 1], jac[:, 2])

    def test_translational_score_differs_from_full(self):
        jac = torch.randn(5, 6, 7, dtype=torch.float64)
        full = yoshikawa_manipulability(jac)
        trans = yoshikawa_manipulability(select_jacobian_rows(jac, "translational"))
        assert full.shape == trans.shape == (5,)
        assert not torch.allclose(full, trans)

    def test_invalid_selection_raises(self):
        jac = torch.randn(1, 6, 7)
        with pytest.raises(ValueError):
            select_jacobian_rows(jac, "bogus")
        with pytest.raises(ValueError):
            select_jacobian_rows(jac, [0, 99])
        with pytest.raises(ValueError):
            select_jacobian_rows(jac, [])
