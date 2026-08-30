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

import torch
import warp as wp

from embodichain.utils.warp.kinematics.opw_solver import (
    opw_ik_path_select_kernel,
    wp_vec6f,
)


def test_opw_path_selector_preserves_temporal_branch_continuity() -> None:
    """The path selector must seed each sample from its previous result."""
    wp.init()
    candidates = torch.zeros(1, 3, 8, 6)
    validity = torch.zeros(1, 3, 8, dtype=torch.int32)
    candidates[0, :, 0, 0] = torch.tensor((0.1, 0.2, 0.3))
    candidates[0, :, 1, 0] = torch.tensor((2.0, 1.9, 1.8))
    validity[:, :, :2] = 1
    output = torch.empty(1, 3, 6)
    success = torch.empty(1, 3, dtype=torch.int32)
    lower = wp_vec6f(-3.14, -3.14, -3.14, -3.14, -3.14, -3.14)
    upper = wp_vec6f(3.14, 3.14, 3.14, 3.14, 3.14, 3.14)

    wp.launch(
        kernel=opw_ik_path_select_kernel,
        dim=1,
        inputs=[
            wp.from_torch(candidates),
            wp.from_torch(validity),
            wp.from_torch(torch.zeros(1, 6)),
            wp_vec6f(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            lower,
            upper,
            0.0,
        ],
        outputs=[wp.from_torch(output), wp.from_torch(success)],
        device="cpu",
    )

    assert torch.equal(success, torch.ones_like(success))
    assert torch.allclose(output[0, :, 0], torch.tensor((0.1, 0.2, 0.3)))


def test_opw_path_selector_rejects_limit_blocked_full_turn() -> None:
    """A raw fallback must not turn a blocked equivalent into a full-turn jump."""
    wp.init()
    candidates = torch.zeros(1, 1, 8, 6)
    candidates[0, 0, 0, 0] = -3.0
    validity = torch.zeros(1, 1, 8, dtype=torch.int32)
    validity[0, 0, 0] = 1
    initial_seed = torch.zeros(1, 6)
    initial_seed[0, 0] = 3.0
    output = torch.empty(1, 1, 6)
    success = torch.empty(1, 1, dtype=torch.int32)
    lower = wp_vec6f(-3.1, -3.1, -3.1, -3.1, -3.1, -3.1)
    upper = wp_vec6f(3.1, 3.1, 3.1, 3.1, 3.1, 3.1)

    wp.launch(
        kernel=opw_ik_path_select_kernel,
        dim=1,
        inputs=[
            wp.from_torch(candidates),
            wp.from_torch(validity),
            wp.from_torch(initial_seed),
            wp_vec6f(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            lower,
            upper,
            0.0,
        ],
        outputs=[wp.from_torch(output), wp.from_torch(success)],
        device="cpu",
    )

    assert success.item() == 0
    assert torch.equal(output[:, 0], initial_seed)


def test_opw_path_selector_rejects_equivalent_inside_safety_margin() -> None:
    """A seed-nearest equivalent must remain outside the excluded margin."""
    wp.init()
    candidates = torch.zeros(1, 1, 8, 6)
    candidates[0, 0, 0, 0] = -6.0
    validity = torch.zeros(1, 1, 8, dtype=torch.int32)
    validity[0, 0, 0] = 1
    initial_seed = torch.zeros(1, 6)
    initial_seed[0, 0] = 0.3
    output = torch.empty(1, 1, 6)
    success = torch.empty(1, 1, dtype=torch.int32)
    lower = wp_vec6f(-6.4, -6.4, -6.4, -6.4, -6.4, -6.4)
    upper = wp_vec6f(0.4, 0.4, 0.4, 0.4, 0.4, 0.4)
    safe_margin = 0.2

    wp.launch(
        kernel=opw_ik_path_select_kernel,
        dim=1,
        inputs=[
            wp.from_torch(candidates),
            wp.from_torch(validity),
            wp.from_torch(initial_seed),
            wp_vec6f(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            lower,
            upper,
            safe_margin,
        ],
        outputs=[wp.from_torch(output), wp.from_torch(success)],
        device="cpu",
    )

    assert success.item() == 0
    assert torch.equal(output[:, 0], initial_seed)
