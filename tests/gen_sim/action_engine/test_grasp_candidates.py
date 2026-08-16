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

from copy import deepcopy
from typing import Any

import torch

from embodichain.gen_sim.action_engine.grasp_candidates import (
    SupportCollisionFallbackProvider,
)


class _FakeProvider:
    def __init__(self, result: Any, diagnostics: dict[str, Any]) -> None:
        self.result = result
        self._diagnostics = diagnostics
        self.calls = 0
        self.generator = self
        self.device = torch.device("cpu")

    @property
    def diagnostics(self) -> dict[str, Any]:
        return deepcopy(self._diagnostics)

    @property
    def last_filter_diagnostics(self) -> dict[str, Any]:
        return self.diagnostics

    def get_valid_grasp_poses(self, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.result

    def get_dual_arm_valid_grasp_poses(self, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.result


def _single_result(success: bool) -> tuple[bool, torch.Tensor, float, torch.Tensor]:
    return success, torch.eye(4), 0.05, torch.zeros(1)


def _stage_diagnostics(*, object_collisions: int, support_collisions: int) -> dict:
    return {
        "mode": "single_arm",
        "center": {
            "input_pair_count": 20,
            "angle_valid_pair_count": 10,
            "pose_candidate_count": 10,
            "collision": {
                "candidate_count": 10,
                "object_collision_count": object_collisions,
                "support_collision_count": support_collisions,
                "combined_collision_count": 10,
                "support_filter_enabled": True,
            },
            "collision_free_pose_count": 0,
        },
    }


def test_retries_without_support_heuristic_when_it_alone_exhausts_candidates() -> None:
    strict = _FakeProvider(
        _single_result(False),
        _stage_diagnostics(object_collisions=0, support_collisions=10),
    )
    relaxed = _FakeProvider(_single_result(True), {"mode": "single_arm"})
    provider = SupportCollisionFallbackProvider(strict, relaxed)

    result = provider.get_valid_grasp_poses(
        object_pose=torch.eye(4),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        object_part="center",
    )

    assert result[0]
    assert strict.calls == 1
    assert relaxed.calls == 1
    assert provider.diagnostics["support_collision_fallback"] == {
        "attempted": True,
        "accepted": True,
        "reason": "support_heuristic_exhausted",
        "relaxed": {"mode": "single_arm"},
    }


def test_does_not_relax_when_object_collision_exhausts_candidates() -> None:
    strict = _FakeProvider(
        _single_result(False),
        _stage_diagnostics(object_collisions=10, support_collisions=10),
    )
    relaxed = _FakeProvider(_single_result(True), {"mode": "single_arm"})
    provider = SupportCollisionFallbackProvider(strict, relaxed)

    result = provider.get_valid_grasp_poses(
        object_pose=torch.eye(4),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        object_part="center",
    )

    assert not result[0]
    assert strict.calls == 1
    assert relaxed.calls == 0
    assert "support_collision_fallback" not in provider.diagnostics


def test_dual_arm_retry_requires_every_failed_side_to_be_support_exhausted() -> None:
    strict_result = {
        "left": {"is_success": True},
        "right": {"is_success": False},
    }
    relaxed_result = {
        "left": {"is_success": True},
        "right": {"is_success": True},
    }
    strict_diagnostics = {
        "mode": "dual_arm",
        "right": _stage_diagnostics(
            object_collisions=0,
            support_collisions=10,
        )["center"],
    }
    strict = _FakeProvider(strict_result, strict_diagnostics)
    relaxed = _FakeProvider(relaxed_result, {"mode": "dual_arm"})
    provider = SupportCollisionFallbackProvider(strict, relaxed)

    result = provider.get_dual_arm_valid_grasp_poses(
        object_pose=torch.eye(4),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        left_to_right_arm_direction=torch.tensor([0.0, 1.0, 0.0]),
        middle_empty_ratio=0.4,
    )

    assert result == relaxed_result
    assert relaxed.calls == 1
    assert provider.diagnostics["support_collision_fallback"]["accepted"]
