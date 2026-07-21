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

import numpy as np
import torch

import embodichain.lab.sim
from embodichain.toolkits.graspkit.pg_grasp import ConvexCollisionChecker


def test_collision_cache_isolated_by_decomposition_method(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_plane_equations(vertices, faces, max_hulls, method):
        calls.append(method)
        return [(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))]

    monkeypatch.setattr(embodichain.lab.sim, "CONVEX_DECOMP_DIR", tmp_path)
    monkeypatch.setattr(
        ConvexCollisionChecker,
        "_compute_plane_equations",
        staticmethod(fake_plane_equations),
    )
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    coacd = ConvexCollisionChecker(vertices, faces, convex_decomposition_method="coacd")
    vhacd = ConvexCollisionChecker(
        vertices, faces, convex_decomposition_method="visacd"
    )

    assert calls == ["coacd", "vhacd"]
    assert coacd.cache_path.endswith("_32_coacd.pkl")
    assert vhacd.cache_path.endswith("_32_vhacd.pkl")
