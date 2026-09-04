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

import pytest
import torch

from embodichain.lab.sim.planners.se3 import (
    plan_se3_line,
    se3_line_evaluate,
    se3_line_state,
)


def test_se3_line_uses_geodesic_rotation_and_linear_translation() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    result = se3_line_evaluate(start, end, torch.tensor([-1.0, 0.5, 2.0]))
    root = 2.0**-0.5
    expected_mid = torch.tensor(
        [
            [root, -root, 0.0, 0.9142135623730951],
            [root, root, 0.0, 0.7928932188134524],
            [0.0, 0.0, 1.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result[0], start)
    assert torch.allclose(result[1], expected_mid, atol=1e-12)
    assert torch.allclose(result[2], end, atol=1e-12)


def test_se3_line_handles_exact_half_turn() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
    end[:3, 3] = torch.tensor([0.2, 0.1, -0.3], dtype=torch.float64)
    poses = se3_line_evaluate(
        start, end, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    )
    assert torch.isfinite(poses).all()
    assert torch.allclose(poses[0], start, atol=1e-12)
    assert torch.allclose(poses[-1], end, atol=1e-12)
    assert torch.allclose(
        poses[:, :3, :3].transpose(-1, -2) @ poses[:, :3, :3],
        torch.eye(3, dtype=torch.float64).expand(3, -1, -1),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ),
            "homogeneous last row",
        ),
        (torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0])), "SO\\(3\\)"),
        (torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])), "SO\\(3\\)"),
    ],
)
def test_se3_line_rejects_non_rigid_transforms(
    transform: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        se3_line_evaluate(torch.eye(4), transform, torch.tensor(0.5))


def test_se3_line_composes_twist_acceleration_and_jerk() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    pose, velocity, acceleration, jerk = se3_line_state(
        start,
        end,
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.4], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([-0.1], dtype=torch.float64),
    )
    tangent = torch.tensor(
        [1.5 * torch.pi / 2, torch.pi / 2 / 2, 3.0, 0.0, 0.0, torch.pi / 2],
        dtype=torch.float64,
    )
    assert pose.shape == (1, 4, 4)
    assert torch.allclose(velocity[0], tangent * 0.4, atol=1e-12)
    assert torch.allclose(acceleration[0], tangent * 0.2, atol=1e-12)
    assert torch.allclose(jerk[0], tangent * -0.1, atol=1e-12)


def test_plan_se3_line_respects_six_dimensional_limits() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.eye(4, dtype=torch.float64)
    end[:3, 3] = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    velocity_limit = torch.tensor([0.4, 0.3, 0.2, 1.0, 1.0, 1.0], dtype=torch.float64)
    acceleration_limit = torch.full((6,), 0.8, dtype=torch.float64)
    jerk_limit = torch.full((6,), 2.0, dtype=torch.float64)
    result = plan_se3_line(
        start,
        end,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        sample_count=501,
        minimum_duration=3.0,
    )
    assert result.duration.item() == 3.0
    assert torch.allclose(result.poses[0], start)
    assert torch.allclose(result.poses[-1], end)
    assert torch.all(result.velocities.abs().amax(dim=0) <= velocity_limit + 1e-12)
    assert torch.all(
        result.accelerations.abs().amax(dim=0) <= acceleration_limit + 1e-12
    )
    assert torch.all(result.jerks.abs().amax(dim=0) <= jerk_limit + 1e-12)


def test_se3_screw_midpoint_matches_holistic_motion_binding() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize(
    ("profile", "duration", "mid_velocity", "mid_jerk"),
    [
        (
            "trapezoidal",
            2.8490042900628456,
            [
                0.04950495049504951,
                -0.24752475247524752,
                0.06303166063045361,
                0.0,
                0.0,
                0.9900990099009901,
            ],
            [0.0] * 6,
        ),
        (
            "double_s",
            3.2632066495476266,
            [
                0.04813658758058283,
                -0.24068293790291415,
                0.061289406856205576,
                0.0,
                0.0,
                0.9627317516116566,
            ],
            [
                -0.09705901479276445,
                0.4852950739638222,
                -0.12357937580718285,
                0.0,
                0.0,
                -1.9411802958552888,
            ],
        ),
    ],
)
def test_se3_time_law_matches_binding_golden(
    profile: str,
    duration: float,
    mid_velocity: list[float],
    mid_jerk: list[float],
) -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    result = plan_se3_line(
        start,
        end,
        torch.tensor([0.4, 0.3, 0.2, 1.0, 1.0, 1.0], dtype=torch.float64),
        torch.full((6,), 0.8, dtype=torch.float64),
        torch.full((6,), 2.0, dtype=torch.float64),
        profile=profile,
        sample_count=3,
    )
    assert result.duration.item() == pytest.approx(duration, abs=1e-12)
    assert torch.allclose(
        result.velocities[1],
        torch.tensor(mid_velocity, dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        result.accelerations[1], torch.zeros(6, dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(
        result.jerks[1], torch.tensor(mid_jerk, dtype=torch.float64), atol=1e-12
    )
    midpoint = se3_line_evaluate(start, end, torch.tensor(0.5, dtype=torch.float64))
    assert torch.allclose(
        midpoint[:3, 3],
        torch.tensor(
            [0.10857864376269052, -0.16213203435596427, 0.05], dtype=torch.float64
        ),
        atol=1e-12,
    )


def test_se3_minimum_duration_report_matches_binding() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    result = plan_se3_line(
        start,
        end,
        torch.tensor([0.4, 0.3, 0.2, 1.0, 1.0, 1.0], dtype=torch.float64),
        torch.full((6,), 0.8, dtype=torch.float64),
        torch.full((6,), 2.0, dtype=torch.float64),
        minimum_duration=5.0,
        sample_count=1001,
    )
    expected_velocity = torch.tensor(
        [0.03141593, 0.15707963, 0.04, 0.0, 0.0, 0.62831853], dtype=torch.float64
    )
    assert result.duration.item() == pytest.approx(5.0, abs=1e-12)
    assert torch.allclose(
        result.constraint_report["peak_velocity"], expected_velocity, atol=5e-9
    )
    assert result.constraint_report["maximum_utilization"].item() == pytest.approx(
        0.6283185307179583, abs=1e-12
    )
    assert result.constraint_report["within_limits"].item()


def test_plan_se3_line_validates_transforms_before_planning() -> None:
    invalid_end = torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    with pytest.raises(ValueError, match=r"SO\(3\)"):
        plan_se3_line(
            torch.eye(4, dtype=torch.float64),
            invalid_end,
            torch.ones(6, dtype=torch.float64),
            torch.ones(6, dtype=torch.float64),
            torch.ones(6, dtype=torch.float64),
        )
