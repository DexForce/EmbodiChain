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

import math

import pytest
import torch

from embodichain.lab.sim.motion.expansion.cfg import TrajectoryAugmentationCfg
from embodichain.lab.sim.motion.expansion.contracts import (
    SceneCase,
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.motion.expansion.variants import (
    NOMINAL_OPERATOR,
    apply_trajectory_variant,
    expand_row_variants,
    expand_trajectory_variants,
    plan_trajectory_variants,
    sample_approach_cone,
)

JOINTS = 6
SAMPLES = 41
FREE_STOP = 30
OPERATORS = ("joint_residual", "via_points", "nullspace_residual", "retime")


def case() -> SceneCase:
    return SceneCase("scene_0", "state_0", "signature", "task_0", "robot_0")


def template() -> TrajectoryTemplate:
    positions = torch.zeros(SAMPLES, JOINTS)
    positions[:, 0] = torch.linspace(0.0, 1.0, SAMPLES)
    positions[:, 3] = torch.linspace(0.0, -0.4, SAMPLES)
    dt = torch.full((SAMPLES,), 0.1)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id="handwritten_qpos",
        source_revision="one",
        template_id="reference_0",
        joint_names=tuple(f"joint_{index}" for index in range(JOINTS)),
        positions=positions,
        dt=dt,
        phases=(
            TrajectoryPhase("reach", 0, FREE_STOP, "free", OPERATORS),
            TrajectoryPhase("grasp", FREE_STOP, SAMPLES, "contact"),
        ),
        allowed_operators=OPERATORS,
        controlled_joint_indices=tuple(range(JOINTS)),
    )


def joint_limits() -> torch.Tensor:
    return torch.stack((torch.full((JOINTS,), -3.0), torch.full((JOINTS,), 3.0)), dim=1)


def task_jacobians(rows: int = 5) -> torch.Tensor:
    jacobians = torch.zeros(SAMPLES, rows, JOINTS)
    for row in range(rows):
        jacobians[:, row, row] = 1.0
    return jacobians


def augmentation(**factors: object) -> TrajectoryAugmentationCfg:
    return TrajectoryAugmentationCfg.from_mapping(
        {
            "seed": 11,
            "factors": factors,
            "coverage": {"target_per_cell": 8, "joint_dedup_normalized_tol": 0.005},
        }
    )


def test_enumeration_starts_from_the_reference_and_spreads_across_factors() -> None:
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 2},
        ik={"enabled": True},
        timing={"enabled": True, "duration_scales": [1.0, 1.4]},
    )
    variants = plan_trajectory_variants(cfg, 6)
    assert variants[0].is_nominal
    assert variants[0].spatial_operator == NOMINAL_OPERATOR
    assert [variant.spatial_operator for variant in variants[1:5]] == [
        "via_points",
        "nullspace_residual",
        "via_points",
        "nullspace_residual",
    ]
    # The duration scale only advances once both spatial operators were tried.
    assert [variant.duration_scale for variant in variants[1:5]] == [1.0, 1.0, 1.4, 1.4]
    assert plan_trajectory_variants(cfg, 6) == variants


def test_enumeration_is_reference_only_without_enabled_factors() -> None:
    variants = plan_trajectory_variants(TrajectoryAugmentationCfg(), 3)
    assert all(variant.is_nominal for variant in variants)
    with pytest.raises(ValueError, match="positive integer"):
        plan_trajectory_variants(TrajectoryAugmentationCfg(), 0)


def test_expansion_preserves_the_reference_row_and_every_fixed_waypoint() -> None:
    source = template()
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 3},
        ik={"enabled": True},
        timing={"enabled": True, "duration_scales": [1.0, 1.3]},
    )
    result = expand_trajectory_variants(
        source,
        cfg=cfg,
        case=case(),
        count=6,
        joint_limits=joint_limits(),
        control_dt=0.1,
        task_jacobians=task_jacobians(),
    )
    batch = result.candidates
    assert len(batch.identities) == 6
    assert result.variants[0].is_nominal
    torch.testing.assert_close(batch.positions[0, :SAMPLES], source.positions)
    for row in range(len(batch.identities)):
        phases = batch.phases[row]
        free, contact = phases[0], phases[1]
        # Both endpoints of the augmented free phase remain the planned waypoints.
        torch.testing.assert_close(
            batch.positions[row, free.start_index], source.positions[0]
        )
        torch.testing.assert_close(
            batch.positions[row, free.stop_index - 1],
            source.positions[FREE_STOP - 1],
        )
        torch.testing.assert_close(
            batch.positions[row, contact.start_index : contact.stop_index],
            source.positions[FREE_STOP:],
        )


def test_expansion_reports_rejections_instead_of_dropping_them_silently() -> None:
    # Timing variants share one geometry, so the per-geometry quota binds.
    cfg = TrajectoryAugmentationCfg.from_mapping(
        {
            "factors": {
                "timing": {
                    "enabled": True,
                    "duration_scales": [1.0, 1.2, 1.4],
                    "profiles": ["uniform", "ease_in", "ease_out"],
                }
            },
            "coverage": {"target_per_cell": 3},
        }
    )
    result = expand_trajectory_variants(
        template(),
        cfg=cfg,
        case=case(),
        count=9,
        joint_limits=joint_limits(),
        control_dt=0.1,
        max_attempts=30,
    )
    assert len(result.variants) == 3
    assert result.rejected["duplicate"] == result.attempted - 3
    assert (
        len({identity.geometry_family_id for identity in result.candidates.identities})
        == 1
    )


def test_motion_limit_failures_are_rejected_with_their_own_reason() -> None:
    cfg = augmentation(spatial={"enabled": True, "joint_offset_scale": 0.2})
    result = expand_trajectory_variants(
        template(),
        cfg=cfg,
        case=case(),
        count=4,
        joint_limits=joint_limits(),
        control_dt=0.1,
        velocity_limits=torch.full((JOINTS,), 0.4),
        acceleration_limits=torch.full((JOINTS,), 100.0),
        max_attempts=6,
    )
    assert result.rejected["motion_limits"] >= 1
    assert all(variant.is_nominal for variant in result.variants)


def test_redundancy_expansion_requires_reference_jacobians() -> None:
    cfg = augmentation(ik={"enabled": True})
    result = expand_trajectory_variants(
        template(),
        cfg=cfg,
        case=case(),
        count=3,
        joint_limits=joint_limits(),
        control_dt=0.1,
        max_attempts=5,
    )
    assert len(result.variants) == 1
    assert result.rejected["operator_rejected"] == 4


def test_shorter_rows_hold_their_final_command_after_padding() -> None:
    cfg = augmentation(timing={"enabled": True, "duration_scales": [1.0, 2.0]})
    result = expand_trajectory_variants(
        template(),
        cfg=cfg,
        case=case(),
        count=2,
        joint_limits=joint_limits(),
        control_dt=0.1,
    )
    batch = result.candidates
    lengths = batch.valid_length.tolist()
    assert lengths[0] < lengths[1]
    horizon = batch.positions.shape[1]
    torch.testing.assert_close(
        batch.positions[0, lengths[0] :],
        batch.positions[0, lengths[0] - 1].expand(horizon - lengths[0], JOINTS),
    )
    assert not bool(batch.dt[0, lengths[0] :].any())
    assert bool(batch.valid_mask[0, lengths[0] - 1])
    assert not bool(batch.valid_mask[0, lengths[0]])


def test_expansion_validates_its_arguments() -> None:
    cfg = TrajectoryAugmentationCfg()
    with pytest.raises(ValueError, match="count must be"):
        expand_trajectory_variants(
            template(),
            cfg=cfg,
            case=case(),
            count=0,
            joint_limits=joint_limits(),
            control_dt=0.1,
        )
    with pytest.raises(ValueError, match="supplied together"):
        expand_trajectory_variants(
            template(),
            cfg=cfg,
            case=case(),
            count=1,
            joint_limits=joint_limits(),
            control_dt=0.1,
            velocity_limits=torch.ones(JOINTS),
        )
    with pytest.raises(ValueError, match="at least count"):
        expand_trajectory_variants(
            template(),
            cfg=cfg,
            case=case(),
            count=4,
            joint_limits=joint_limits(),
            control_dt=0.1,
            max_attempts=2,
        )


def test_approach_cone_keeps_the_nominal_direction_and_stays_inside_the_cap() -> None:
    cfg = TrajectoryAugmentationCfg.from_mapping(
        {"factors": {"approach": {"enabled": True, "cone_half_angle_rad": 0.3}}}
    )
    generator = torch.Generator(device="cpu").manual_seed(5)
    polar, azimuth = sample_approach_cone(cfg, count=16, generator=generator)
    assert polar.shape == azimuth.shape == (16,)
    assert float(polar[0]) == 0.0 and float(azimuth[0]) == 0.0
    assert float(polar.max()) <= 0.3
    assert float(azimuth.abs().max()) <= math.pi
    # Area-uniform sampling puts roughly half the directions past the median
    # polar angle of the cap, unlike uniform sampling of the angle itself.
    median = math.acos((1 + math.cos(0.3)) / 2)
    assert 4 <= int((polar[1:] > median).sum()) <= 11


def test_approach_cone_refuses_a_disabled_factor() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(ValueError, match="disabled"):
        sample_approach_cone(TrajectoryAugmentationCfg(), count=4, generator=generator)


def row_templates(count: int) -> list[TrajectoryTemplate]:
    rows = []
    for index in range(count):
        source = template()
        positions = source.positions.clone()
        positions[:, 0] *= 1.0 + 0.05 * index
        rows.append(
            TrajectoryTemplate(
                source_id=source.source_id,
                source_revision=source.source_revision,
                template_id=f"row_{index}",
                joint_names=source.joint_names,
                positions=positions,
                dt=source.dt,
                phases=source.phases,
                allowed_operators=source.allowed_operators,
                controlled_joint_indices=source.controlled_joint_indices,
            )
        )
    return rows


def row_cases(count: int) -> list[SceneCase]:
    return [
        SceneCase("scene_0", f"state_{index}", "signature", "task_0", "robot_0")
        for index in range(count)
    ]


def test_row_expansion_gives_each_row_its_own_variant_and_keeps_row_zero() -> None:
    rows = row_templates(5)
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 3},
        ik={"enabled": True},
        timing={"enabled": True, "duration_scales": [1.0, 1.25]},
    )
    result = expand_row_variants(
        rows,
        cfg=cfg,
        cases=row_cases(5),
        joint_limits=joint_limits(),
        control_dt=0.1,
        task_jacobians=[task_jacobians()] * 5,
    )
    batch = result.candidates
    assert len(batch.identities) == 5
    assert result.rejected == {}
    assert result.variants[0].is_nominal
    assert len({variant.spatial_operator for variant in result.variants[1:]}) == 2
    assert batch.source_row_indices.tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(batch.positions[0, :SAMPLES], rows[0].positions)
    for index, row in enumerate(rows):
        contact = batch.phases[index][1]
        torch.testing.assert_close(
            batch.positions[index, contact.start_index : contact.stop_index],
            row.positions[FREE_STOP:],
        )


def test_row_expansion_falls_back_to_the_reference_instead_of_dropping_a_row() -> None:
    rows = row_templates(4)
    # The redundancy operator needs Jacobians; without them its rows must fall
    # back rather than leave a parallel environment without a command stream.
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 3},
        ik={"enabled": True},
    )
    result = expand_row_variants(
        rows, cfg=cfg, cases=row_cases(4), joint_limits=joint_limits(), control_dt=0.1
    )
    assert len(result.candidates.identities) == 4
    # Only the single redundancy row of this enumeration can fail.
    assert result.rejected["operator_rejected"] == 1
    assert [variant.spatial_operator for variant in result.variants] == [
        NOMINAL_OPERATOR,
        "via_points",
        NOMINAL_OPERATOR,
        "via_points",
    ]
    for index, variant in enumerate(result.variants):
        assert dict(result.candidates.factors[index])["spatial_operator"] == (
            variant.spatial_operator
        )


def test_row_expansion_validates_its_sequences() -> None:
    cfg = TrajectoryAugmentationCfg()
    with pytest.raises(ValueError, match="at least one row"):
        expand_row_variants(
            [], cfg=cfg, cases=[], joint_limits=joint_limits(), control_dt=0.1
        )
    with pytest.raises(ValueError, match="one SceneCase per template row"):
        expand_row_variants(
            row_templates(2),
            cfg=cfg,
            cases=row_cases(1),
            joint_limits=joint_limits(),
            control_dt=0.1,
        )
    with pytest.raises(ValueError, match="one entry per template row"):
        expand_row_variants(
            row_templates(2),
            cfg=cfg,
            cases=row_cases(2),
            joint_limits=joint_limits(),
            control_dt=0.1,
            task_jacobians=[task_jacobians()],
        )


def test_variant_application_is_reproducible_and_row_specific() -> None:
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 2}
    )
    variant = plan_trajectory_variants(cfg, 2)[1]
    rows = row_templates(2)
    first = apply_trajectory_variant(
        rows[0], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    repeat = apply_trajectory_variant(
        rows[0], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    torch.testing.assert_close(first.positions, repeat.positions)
    # A different template identity draws independently for the same variant.
    other = apply_trajectory_variant(
        rows[1], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    assert not torch.allclose(
        first.positions - rows[0].positions, other.positions - rows[1].positions
    )
    nominal = apply_trajectory_variant(
        rows[0],
        plan_trajectory_variants(cfg, 1)[0],
        cfg=cfg,
        joint_limits=joint_limits(),
        control_dt=0.1,
    )
    torch.testing.assert_close(nominal.positions, rows[0].positions)


def test_row_expansion_walks_the_factor_grid_across_rollouts() -> None:
    rows = row_templates(4)
    cases = row_cases(4)
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 2},
        timing={
            "enabled": True,
            "duration_scales": [1.0, 1.2],
            "profiles": ["uniform", "ease_in"],
        },
    )
    seen = []
    for rollout in range(3):
        result = expand_row_variants(
            rows,
            cfg=cfg,
            cases=cases,
            joint_limits=joint_limits(),
            control_dt=0.1,
            ordinal_offset=rollout * (len(rows) - 1),
        )
        # The reference branch survives every rollout, ahead of fresh ordinals.
        assert result.variants[0].is_nominal
        seen.append(tuple(variant.ordinal for variant in result.variants))
    assert seen == [(0, 1, 2, 3), (0, 4, 5, 6), (0, 7, 8, 9)]
    combinations = {
        (variant.duration_scale, variant.timing_profile)
        for ordinals in seen[:2]
        for variant in plan_trajectory_variants(cfg, 7)
        if variant.ordinal in ordinals and not variant.is_nominal
    }
    assert len(combinations) == 4


def test_row_expansion_rejects_a_negative_offset() -> None:
    with pytest.raises(ValueError, match="ordinal_offset"):
        expand_row_variants(
            row_templates(2),
            cfg=TrajectoryAugmentationCfg(),
            cases=row_cases(2),
            joint_limits=joint_limits(),
            control_dt=0.1,
            ordinal_offset=-1,
        )
