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

"""Manipulability profiles, bands, and guided residual selection."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.motion.expansion.contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.motion.expansion.manipulability import (
    ManipulabilityBands,
    describe_manipulability,
    manipulability_guided_residual,
)

LIMITS = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])


def template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="reference",
        source_revision="one",
        template_id="pick",
        joint_names=("arm", "gripper"),
        positions=torch.tensor(
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.1]]
        ),
        dt=torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1]),
        phases=(
            TrajectoryPhase("transit", 0, 3, allowed_operators=("joint_residual",)),
            TrajectoryPhase("grasp", 3, 5, kind="contact"),
        ),
        controlled_joint_indices=(0,),
        allowed_operators=("joint_residual",),
    )


def diagonal_jacobians(scales: torch.Tensor) -> torch.Tensor:
    """Build (T, 3, 3) diagonal Jacobians whose Yoshikawa score is ``scale ** 3``."""
    return torch.eye(3, dtype=scales.dtype) * scales[:, None, None]


def bottleneck_jacobian(positions: torch.Tensor) -> torch.Tensor:
    """Model a reference whose interior waypoint passes nearest a singularity.

    The bottleneck score is ``(0.5 + |q - 0.25|) ** 3`` at that waypoint, so a
    residual steering the arm away from ``0.25`` raises the whole trajectory's
    manipulability and a residual leaving it in place keeps the tight posture.
    """
    scale = 0.5 + (positions[:, 0].to(torch.float64) - 0.25).abs()
    return diagonal_jacobians(scale)


# Bottleneck of the unmodified reference, used to normalize the bands below.
REFERENCE_SCORE = 0.125


def test_profile_reports_per_sample_scores_and_phase_bottlenecks() -> None:
    jacobians = diagonal_jacobians(torch.tensor([2.0, 1.0, 3.0, 4.0]))
    profile = describe_manipulability(
        jacobians,
        phases=(TrajectoryPhase("early", 0, 2), TrajectoryPhase("late", 2, 4)),
    )
    torch.testing.assert_close(
        profile.scores, torch.tensor([8.0, 1.0, 27.0, 64.0], dtype=torch.float64)
    )
    assert profile.phase_ids == ("early", "late")
    torch.testing.assert_close(
        profile.phase_bottlenecks, torch.tensor([1.0, 27.0], dtype=torch.float64)
    )
    assert profile.bottleneck == pytest.approx(1.0)
    assert profile.mean == pytest.approx(25.0)


def test_row_selection_scores_only_the_requested_task_directions() -> None:
    jacobian = torch.zeros(1, 6, 6)
    jacobian[0].fill_diagonal_(1.0)
    jacobian[0, 3:, 3:] *= 0.5
    full = describe_manipulability(jacobian).bottleneck
    rotational = describe_manipulability(jacobian, rows="rotational").bottleneck
    assert full == pytest.approx(0.125)
    assert rotational == pytest.approx(0.125)
    assert describe_manipulability(jacobian, rows="translational").bottleneck == (
        pytest.approx(1.0)
    )


def test_low_precision_jacobians_are_scored_without_determinant_underflow() -> None:
    # float32 det() of a 1e-4-scaled Jacobian underflows to exactly zero and
    # would report a healthy posture as singular.
    jacobians = diagonal_jacobians(torch.full((2,), 1e-4)).to(torch.float32)
    assert describe_manipulability(jacobians).bottleneck == pytest.approx(1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 3),
        torch.zeros(0, 3, 3),
        torch.full((2, 3, 3), float("nan")),
    ],
)
def test_profile_rejects_malformed_jacobians(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        describe_manipulability(bad)


def test_profile_rejects_phases_outside_the_scored_samples() -> None:
    with pytest.raises(ValueError, match="within the scored samples"):
        describe_manipulability(
            diagonal_jacobians(torch.ones(2)),
            phases=(TrajectoryPhase("late", 0, 3),),
        )


def test_bands_split_ratios_against_the_reference_manipulability() -> None:
    bands = ManipulabilityBands((0.5, 0.9), reference=4.0)
    assert bands.count == 3
    assert [bands.band_of(value) for value in (0.0, 1.9, 2.0, 3.5, 3.6, 40.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert bands.ratio(2.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edges": (), "reference": 1.0},
        {"edges": (0.5, 0.5), "reference": 1.0},
        {"edges": (0.9, 0.5), "reference": 1.0},
        {"edges": (0.0,), "reference": 1.0},
        {"edges": (0.5,), "reference": 0.0},
        {"edges": (0.5,), "reference": float("inf")},
    ],
)
def test_bands_reject_degenerate_edges_or_reference(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ManipulabilityBands(kwargs["edges"], reference=kwargs["reference"])


def test_bands_reject_negative_scores() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ManipulabilityBands((0.5,), reference=1.0).band_of(-1.0)


def test_guided_sampling_keeps_the_first_proposal_reaching_the_target_band() -> None:
    bands = ManipulabilityBands((1.2, 1.5), reference=REFERENCE_SCORE)
    scored: list[float] = []

    def jacobian_fn(positions: torch.Tensor) -> torch.Tensor:
        scored.append(float(positions[1, 0]))
        return bottleneck_jacobian(positions)

    # Band 0 keeps the tight reference posture, which uniform residual sampling
    # around a comfortable waypoint reaches only part of the time.
    result = manipulability_guided_residual(
        template(),
        joint_limits=LIMITS,
        normalized_scale=0.05,
        generator=torch.Generator().manual_seed(0),
        jacobian_fn=jacobian_fn,
        bands=bands,
        target_band=0,
        proposals=8,
    )
    assert result.matched and result.band == 0
    assert bands.ratio(result.profile.bottleneck) < 1.2
    assert result.proposals_scored == len(scored) < 8
    assert float(result.template.positions[1, 0]) == pytest.approx(scored[-1])
    assert torch.equal(
        result.template.positions[[0, 2, 3, 4]], template().positions[[0, 2, 3, 4]]
    )


def test_guided_sampling_reaches_every_configured_band() -> None:
    bands = ManipulabilityBands((1.2, 1.5), reference=REFERENCE_SCORE)
    reached = {
        target: manipulability_guided_residual(
            template(),
            joint_limits=LIMITS,
            normalized_scale=0.05,
            generator=torch.Generator().manual_seed(11),
            jacobian_fn=bottleneck_jacobian,
            bands=bands,
            target_band=target,
            proposals=12,
        )
        for target in range(bands.count)
    }
    assert all(result.matched for result in reached.values())
    assert [reached[target].band for target in range(bands.count)] == [0, 1, 2]
    # One reference and one random stream still produce distinct postures.
    postures = {float(result.template.positions[1, 0]) for result in reached.values()}
    assert len(postures) == bands.count


def test_guided_sampling_falls_back_to_the_nearest_reachable_band() -> None:
    # No residual within the allowed scale can reach this far-away band.
    bands = ManipulabilityBands((100.0, 200.0), reference=REFERENCE_SCORE)
    result = manipulability_guided_residual(
        template(),
        joint_limits=LIMITS,
        normalized_scale=0.05,
        generator=torch.Generator().manual_seed(1),
        jacobian_fn=bottleneck_jacobian,
        bands=bands,
        target_band=2,
        proposals=3,
    )
    assert not result.matched
    assert result.band == 0
    assert result.proposals_scored == 3


def test_guided_sampling_is_reproducible_without_consuming_global_rng() -> None:
    before = torch.random.get_rng_state().clone()
    bands = ManipulabilityBands((1.2, 1.5), reference=REFERENCE_SCORE)
    results = [
        manipulability_guided_residual(
            template(),
            joint_limits=LIMITS,
            normalized_scale=0.05,
            generator=torch.Generator().manual_seed(7),
            jacobian_fn=bottleneck_jacobian,
            bands=bands,
            target_band=2,
            proposals=4,
        )
        for _ in range(2)
    ]
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(results[0].template.positions, results[1].template.positions)
    assert results[0].band == results[1].band


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (torch.tensor([[-1.0, 0.4], [-1.0, 1.0]]), "violates joint limits"),
        (torch.zeros(2, 2), "finite increasing intervals"),
    ],
)
def test_guided_sampling_reraises_when_every_proposal_fails(
    limits: torch.Tensor, message: str
) -> None:
    # A systematically invalid draw must surface instead of being swallowed as
    # an exhausted proposal budget.
    with pytest.raises(ValueError, match=message):
        manipulability_guided_residual(
            template(),
            joint_limits=limits,
            normalized_scale=0.05,
            generator=torch.Generator().manual_seed(3),
            jacobian_fn=bottleneck_jacobian,
            bands=ManipulabilityBands((0.5,), reference=REFERENCE_SCORE),
            target_band=0,
            proposals=4,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"target_band": 2},
        {"target_band": -1},
        {"target_band": True},
        {"proposals": 0},
        {"jacobian_fn": "not_callable"},
        {"bands": (0.5,)},
    ],
)
def test_guided_sampling_rejects_invalid_selection_arguments(override: dict) -> None:
    kwargs = {
        "joint_limits": LIMITS,
        "normalized_scale": 0.05,
        "generator": torch.Generator().manual_seed(0),
        "jacobian_fn": bottleneck_jacobian,
        "bands": ManipulabilityBands((0.5,), reference=1.0),
        "target_band": 0,
        "proposals": 2,
        **override,
    }
    with pytest.raises(ValueError):
        manipulability_guided_residual(template(), **kwargs)


def test_guided_sampling_requires_an_annotated_free_phase() -> None:
    replay_only = TrajectoryTemplate(
        source_id="reference",
        source_revision="one",
        template_id="pick",
        joint_names=("arm",),
        positions=torch.zeros(3, 1),
        dt=torch.tensor([0.0, 0.1, 0.1]),
        controlled_joint_indices=(0,),
    )
    with pytest.raises(ValueError, match="does not allow"):
        manipulability_guided_residual(
            replay_only,
            joint_limits=torch.tensor([[-1.0, 1.0]]),
            normalized_scale=0.05,
            generator=torch.Generator().manual_seed(0),
            jacobian_fn=bottleneck_jacobian,
            bands=ManipulabilityBands((0.5,), reference=1.0),
            target_band=0,
        )


def test_guided_sampling_rejects_a_jacobian_set_that_misses_samples() -> None:
    with pytest.raises(ValueError, match="one .* Jacobian per candidate sample"):
        manipulability_guided_residual(
            template(),
            joint_limits=LIMITS,
            normalized_scale=0.05,
            generator=torch.Generator().manual_seed(0),
            jacobian_fn=lambda positions: diagonal_jacobians(torch.ones(2)),
            bands=ManipulabilityBands((0.5,), reference=1.0),
            target_band=0,
        )
