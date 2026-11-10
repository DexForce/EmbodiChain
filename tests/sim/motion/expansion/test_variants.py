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
from dataclasses import replace

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
    default_variant_factors,
    apply_trajectory_variant,
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
    variants = plan_trajectory_variants(6, cfg)
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
    assert plan_trajectory_variants(6, cfg) == variants


def test_enumeration_is_reference_only_without_enabled_factors() -> None:
    variants = plan_trajectory_variants(3, TrajectoryAugmentationCfg())
    assert all(variant.is_nominal for variant in variants)
    with pytest.raises(ValueError, match="positive integer"):
        plan_trajectory_variants(0, TrajectoryAugmentationCfg())


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


def test_explicitly_enabled_redundancy_without_jacobians_fails_fast() -> None:
    # Spending the whole attempt budget on proposals that cannot succeed would
    # bury the real problem in a rejection log.
    with pytest.raises(ValueError, match="task_jacobians"):
        expand_trajectory_variants(
            template(),
            cfg=augmentation(ik={"enabled": True}),
            case=case(),
            count=3,
            joint_limits=joint_limits(),
            control_dt=0.1,
            max_attempts=5,
        )


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


def distinct_templates(count: int) -> list[TrajectoryTemplate]:
    rows = []
    for index in range(count):
        source = template()
        positions = source.positions.clone()
        positions[:, 0] *= 1.0 + 0.05 * index
        rows.append(
            TrajectoryTemplate(
                source_id=source.source_id,
                source_revision=source.source_revision,
                template_id=f"reference_{index}",
                joint_names=source.joint_names,
                positions=positions,
                dt=source.dt,
                phases=source.phases,
                allowed_operators=source.allowed_operators,
                controlled_joint_indices=source.controlled_joint_indices,
            )
        )
    return rows


def test_variant_application_is_reproducible_and_reference_specific() -> None:
    cfg = augmentation(
        spatial={"enabled": True, "method": "via_points", "via_count": 2}
    )
    variant = plan_trajectory_variants(2, cfg)[1]
    references = distinct_templates(2)
    first = apply_trajectory_variant(
        references[0], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    repeat = apply_trajectory_variant(
        references[0], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    torch.testing.assert_close(first.positions, repeat.positions)
    # A different template identity draws independently for the same variant.
    other = apply_trajectory_variant(
        references[1], variant, cfg=cfg, joint_limits=joint_limits(), control_dt=0.1
    )
    assert not torch.allclose(
        first.positions - references[0].positions,
        other.positions - references[1].positions,
    )
    nominal = apply_trajectory_variant(
        references[0],
        plan_trajectory_variants(1, cfg)[0],
        cfg=cfg,
        joint_limits=joint_limits(),
        control_dt=0.1,
    )
    torch.testing.assert_close(nominal.positions, references[0].positions)


def test_a_reference_artifact_outside_limits_does_not_reject_every_proposal() -> None:
    # An observed reference can hold an untouched joint a few microradians
    # outside its range. That must not be blamed on the sampled residual.
    source = template()
    positions = source.positions.clone()
    positions[:, -1] = -1e-6
    drifted = TrajectoryTemplate(
        source_id=source.source_id,
        source_revision=source.source_revision,
        template_id=source.template_id,
        joint_names=source.joint_names,
        positions=positions,
        dt=source.dt,
        phases=source.phases,
        allowed_operators=source.allowed_operators,
        controlled_joint_indices=tuple(range(JOINTS - 1)),
    )
    limits = joint_limits().clone()
    limits[-1] = torch.tensor([0.0, 0.04])
    result = expand_trajectory_variants(
        drifted,
        cfg=augmentation(
            spatial={"enabled": True, "method": "via_points", "via_count": 2}
        ),
        case=case(),
        count=3,
        joint_limits=limits,
        control_dt=0.1,
    )
    assert len(result.variants) == 3
    assert not any("joint limits" in reason for reason in result.rejected)


def test_omitting_the_configuration_enables_every_usable_factor() -> None:
    policy = default_variant_factors()
    assert policy.factors.spatial.enabled and policy.factors.timing.enabled
    assert policy.factors.ik.enabled
    # The deduplication tolerance has to stay below the offset scale, or
    # variants that really differ are discarded as duplicates.
    assert (
        policy.coverage.joint_dedup_normalized_tol
        < policy.factors.spatial.joint_offset_scale
    )
    assert not default_variant_factors(redundancy=False).factors.ik.enabled

    result = expand_trajectory_variants(
        template(),
        case=case(),
        count=5,
        joint_limits=joint_limits(),
        control_dt=0.1,
        task_jacobians=task_jacobians(),
    )
    assert len(result.variants) == 5
    assert result.variants[0].is_nominal
    assert result.cfg == default_variant_factors(redundancy=True)
    operators = {variant.spatial_operator for variant in result.variants}
    assert {"via_points", "nullspace_residual"} <= operators


def test_omitting_both_configuration_and_jacobians_drops_the_ik_factor() -> None:
    result = expand_trajectory_variants(
        template(),
        case=case(),
        count=4,
        joint_limits=joint_limits(),
        control_dt=0.1,
    )
    assert not result.cfg.factors.ik.enabled
    assert "nullspace_residual" not in {
        variant.spatial_operator for variant in result.variants
    }
    assert not any("task_jacobians" in reason for reason in result.rejected)


def test_an_explicit_configuration_is_never_replaced_by_the_default() -> None:
    explicit = TrajectoryAugmentationCfg()
    result = expand_trajectory_variants(
        template(),
        cfg=explicit,
        case=case(),
        count=4,
        joint_limits=joint_limits(),
        control_dt=0.1,
    )
    assert result.cfg == explicit
    assert len(result.variants) == 1 and result.variants[0].is_nominal


def test_several_spatial_methods_each_get_their_own_variant() -> None:
    cfg = augmentation(
        spatial={
            "enabled": True,
            "method": ["joint_residual", "via_points"],
            "via_count": 2,
        },
        ik={"enabled": True},
    )
    operators = [
        variant.spatial_operator for variant in plan_trajectory_variants(4, cfg)
    ]
    assert operators == [
        NOMINAL_OPERATOR,
        "joint_residual",
        "via_points",
        "nullspace_residual",
    ]
    result = expand_trajectory_variants(
        template(),
        cfg=cfg,
        case=case(),
        count=4,
        joint_limits=joint_limits(),
        control_dt=0.1,
        task_jacobians=task_jacobians(),
    )
    assert {variant.spatial_operator for variant in result.variants} == {
        NOMINAL_OPERATOR,
        "joint_residual",
        "via_points",
        "nullspace_residual",
    }


def test_the_default_policy_offers_every_joint_path_operator() -> None:
    policy = default_variant_factors()
    assert set(policy.factors.spatial.method) == {"joint_residual", "via_points"}
    operators = {
        variant.spatial_operator
        for variant in plan_trajectory_variants(4, cfg=policy)
        if not variant.is_nominal
    }
    assert operators == {"joint_residual", "via_points", "nullspace_residual"}


def test_the_approach_factor_cannot_be_dropped_silently() -> None:
    # It moves a Cartesian standoff pose, so a qpos enumeration can never
    # honour it. Accepting the configuration and ignoring the factor would
    # leave a caller waiting for variation that never arrives.
    cfg = augmentation(
        spatial={"enabled": True},
        approach={"enabled": True, "cone_half_angle_rad": 0.3},
    )
    with pytest.raises(ValueError, match="perturb_approach_direction"):
        plan_trajectory_variants(4, cfg)
    with pytest.raises(ValueError, match="perturb_approach_direction"):
        expand_trajectory_variants(
            template(),
            cfg=cfg,
            case=case(),
            count=4,
            joint_limits=joint_limits(),
            control_dt=0.1,
        )
    # The configuration schema still accepts it, so an upstream planning stage
    # can declare the factor in a shared job configuration.
    assert cfg.factors.approach.enabled
    assert not default_variant_factors().factors.approach.enabled


def test_configured_task_rows_select_from_a_full_spatial_jacobian() -> None:
    # Six rows on six joints leave no redundancy, so a configuration that keeps
    # every row must fail while one that drops a row must succeed. That is only
    # true if the variant path actually reads ik.task_rows.
    source = template()
    full = torch.zeros(SAMPLES, 6, JOINTS)
    for row in range(6):
        full[:, row, row] = 1.0

    def expand(rows: list[int]):
        return expand_trajectory_variants(
            source,
            cfg=augmentation(ik={"enabled": True, "task_rows": rows}),
            case=case(),
            count=2,
            joint_limits=joint_limits(),
            control_dt=0.1,
            task_jacobians=full,
            max_attempts=2,
        )

    with pytest.raises(ValueError, match="redundancy"):
        expand([0, 1, 2, 3, 4, 5])
    result = expand([0, 1, 2, 3, 4])
    assert result.variants[1].spatial_operator == "nullspace_residual"
    moved = result.candidates.positions[1] - source.positions
    # Only the sixth joint is unconstrained, so the residual lives there.
    assert float(moved[:, 5].abs().max()) > 0
    assert float(moved[:, :5].abs().max()) == 0.0


def test_a_malformed_jacobian_is_not_counted_as_a_rejected_proposal() -> None:
    # Swallowing this as bookkeeping would return a partial result built from
    # the nominal variant alone, hiding the caller's mistake.
    with pytest.raises(ValueError, match="matching the template samples"):
        expand_trajectory_variants(
            template(),
            cfg=augmentation(ik={"enabled": True}),
            case=case(),
            count=3,
            joint_limits=joint_limits(),
            control_dt=0.1,
            task_jacobians=torch.zeros(3, 5, JOINTS),
        )


def test_the_unconfigured_plan_then_apply_workflow_runs_without_jacobians() -> None:
    reference = template()
    limits = joint_limits()
    # Planning cannot see whether the caller has Jacobians, so the variants it
    # offers unconfigured must all be applicable without them.
    variants = plan_trajectory_variants(8)
    assert not any(
        variant.spatial_operator == "nullspace_residual" for variant in variants
    )
    for variant in variants:
        applied = apply_trajectory_variant(
            reference, variant, joint_limits=limits, control_dt=0.1
        )
        assert applied.positions.shape[1] == JOINTS


def test_planning_still_offers_posture_variants_when_asked_for_redundancy() -> None:
    policy = default_variant_factors(redundancy=True)
    variants = plan_trajectory_variants(8, cfg=policy)
    assert any(variant.spatial_operator == "nullspace_residual" for variant in variants)
    jacobians = task_jacobians()
    for variant in variants:
        apply_trajectory_variant(
            template(),
            variant,
            cfg=policy,
            joint_limits=joint_limits(),
            control_dt=0.1,
            task_jacobians=jacobians,
        )


def test_application_enables_redundancy_only_when_jacobians_are_supplied() -> None:
    posture = next(
        variant
        for variant in plan_trajectory_variants(
            8, cfg=default_variant_factors(redundancy=True)
        )
        if variant.spatial_operator == "nullspace_residual"
    )
    # The same variant applied without a configuration follows the Jacobians:
    # present means the posture factor runs, absent means it is not reachable.
    applied = apply_trajectory_variant(
        template(),
        posture,
        joint_limits=joint_limits(),
        control_dt=0.1,
        task_jacobians=task_jacobians(),
    )
    assert applied.positions.shape[0] == SAMPLES
    with pytest.raises(ValueError, match="nullspace_residual"):
        apply_trajectory_variant(
            template(), posture, joint_limits=joint_limits(), control_dt=0.1
        )


def _analytic_template() -> TrajectoryTemplate:
    """Use three task coordinates, one redundant joint and an untouched hand."""
    q = torch.zeros(13, 5, dtype=torch.float64)
    q[:9, 2] = 0.5 - 0.4 * torch.linspace(0, math.pi, 9).sin().square()
    q[9:, 2] = 0.5
    q[:9, 0] = torch.linspace(0, 0.2, 9)
    q[9:, 0] = 0.2
    q[:, 3], q[:, 4] = 0.2, 0.15
    dt = torch.full((13,), 0.1, dtype=torch.float64)
    dt[0] = 0
    return TrajectoryTemplate(
        "analytic",
        "one",
        "redundant",
        ("y", "redundant", "x", "z", "hand"),
        q,
        dt,
        phases=(
            TrajectoryPhase(
                "move",
                0,
                9,
                "free",
                ("joint_residual", "task_space_residual", "retime"),
            ),
            TrajectoryPhase("grasp", 9, 13, "contact"),
        ),
        allowed_operators=("joint_residual", "task_space_residual", "retime"),
        controlled_joint_indices=(2, 0, 3, 1),
    )


def _analytic_kinematics(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic FK with translational manipulability exactly ``1 + q[:, 0]``."""
    poses = torch.eye(4, dtype=q.dtype, device=q.device).repeat(len(q), 1, 1)
    poses[:, 0, 3], poses[:, 1, 3] = q[:, 0], q[:, 1]
    poses[:, 2, 3] = (1 + q[:, 0]) * q[:, 2]
    jac = torch.zeros(len(q), 6, 4, dtype=q.dtype, device=q.device)
    jac[:, 0, 0], jac[:, 1, 1] = 1, 1
    jac[:, 2, 0], jac[:, 2, 2] = q[:, 2], 1 + q[:, 0]
    return poses, jac


def _analytic_cfg(joint: bool, task: bool, guided: bool) -> TrajectoryAugmentationCfg:
    return augmentation(
        spatial={"enabled": joint, "joint_offset_scale": 0.08},
        task_space={
            "enabled": task,
            "translation_axes": [False, True, False],
            "max_translation_m": 0.04,
            "max_rotation_rad": 0.0,
            "ik": {
                "damping": 1e-4,
                "step_size": 1.0,
                "max_iterations": 12,
                "pos_eps": 1e-7,
                "rot_eps": 1e-7,
            },
        },
        manipulability={
            "enabled": guided,
            "guided_proposals": 4,
            "jacobian_rows": "translational",
            "band_edges": [0.85, 1.05],
        },
    )


@pytest.mark.parametrize(
    "joint,task,guided",
    [
        (True, False, False),
        (False, False, True),
        (False, True, False),
        (True, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_all_generation_routes_share_constraints_budgets_and_provenance(
    joint: bool, task: bool, guided: bool
) -> None:
    source = _analytic_template()
    cfg = _analytic_cfg(joint, task, guided)
    state = torch.random.get_rng_state().clone()
    kwargs = dict(
        cfg=cfg,
        case=case(),
        count=4,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
        max_attempts=24,
    )
    result = expand_trajectory_variants(source, **kwargs)
    repeated = expand_trajectory_variants(source, **kwargs)
    assert len(result.variants) == 4
    assert result.attempted <= 24
    assert result.attempted == len(result.variants) + sum(result.rejected.values())
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(result.candidates.positions, repeated.candidates.positions)
    assert result.rejected == repeated.rejected
    assert result.variants[0].is_nominal
    assert torch.equal(result.candidates.positions[0], source.positions)
    reference_poses, _ = _analytic_kinematics(source.positions[:, [2, 0, 3, 1]])
    for row in range(1, 4):
        q = result.candidates.positions[row]
        assert torch.equal(q[[0, 8]], source.positions[[0, 8]])
        assert torch.equal(q[9:], source.positions[9:])
        assert torch.equal(q[:, 4], source.positions[:, 4])
        factors = result.candidates.factors[row]
        assert ("task_space_operator" in factors) == task
        assert ("manipulability_band" in factors) == guided
        if task:
            poses, _ = _analytic_kinematics(q[:, [2, 0, 3, 1]])
            # The deformation authorizes Y only. X and Z have independent
            # analytic targets, including after a redundant joint edit.
            torch.testing.assert_close(
                poses[:, [0, 2], 3], reference_poses[:, [0, 2], 3], atol=1e-6, rtol=0
            )
            assert (
                float((poses[:, 1, 3] - reference_poses[:, 1, 3]).abs().max())
                <= 0.040001
            )
        if guided:
            assert factors["manipulability"] == pytest.approx(
                float((1 + q[:, 2]).min()), abs=1e-8
            )


def test_guidance_inner_draws_cannot_exceed_the_shared_proposal_budget() -> None:
    cfg = _analytic_cfg(False, False, True).to_dict()
    cfg["factors"]["manipulability"]["band_edges"] = [0.1, 0.2]
    result = expand_trajectory_variants(
        _analytic_template(),
        cfg=TrajectoryAugmentationCfg.from_mapping(cfg),
        case=case(),
        count=2,
        max_attempts=3,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
    )
    assert result.attempted == 3
    assert len(result.variants) == 2
    assert result.candidates.factors[1]["manipulability_proposals_scored"] == 2
    assert result.rejected == {"guidance_unselected": 1}


def test_unreachable_task_space_draws_are_rejected_without_reusing_rows() -> None:
    cfg = _analytic_cfg(False, True, False).to_dict()
    cfg["factors"]["task_space"].update(
        translation_axes=[False] * 3,
        max_translation_m=0.0,
        rotation_axes=[False, False, True],
        max_rotation_rad=0.2,
    )
    result = expand_trajectory_variants(
        _analytic_template(),
        cfg=TrajectoryAugmentationCfg.from_mapping(cfg),
        case=case(),
        count=3,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
        max_attempts=5,
    )
    assert len(result.variants) == 1
    assert result.variants[0].is_nominal
    assert result.attempted == 5
    assert result.rejected == {"trajectory_ik": 4}


@pytest.mark.parametrize("task,guided", [(True, False), (False, True)])
def test_generation_requires_live_kinematics_and_propagates_bad_callbacks(
    task: bool, guided: bool
) -> None:
    kwargs = dict(
        cfg=_analytic_cfg(False, task, guided),
        case=case(),
        count=2,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
    )
    with pytest.raises(ValueError, match="requires kinematics"):
        expand_trajectory_variants(_analytic_template(), **kwargs)

    def malformed(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        poses, jac = _analytic_kinematics(q)
        return poses, jac[:-1]

    with pytest.raises(ValueError, match="Jacobians must have shape"):
        expand_trajectory_variants(_analytic_template(), kinematics=malformed, **kwargs)


def test_combination_respects_separate_joint_and_task_phase_permissions() -> None:
    source = replace(
        _analytic_template(),
        phases=(
            TrajectoryPhase("cartesian", 0, 7, "free", ("task_space_residual",)),
            TrajectoryPhase("joint", 7, 13, "free", ("joint_residual",)),
        ),
    )
    cfg = _analytic_cfg(True, True, False)
    result = apply_trajectory_variant(
        source,
        plan_trajectory_variants(2, cfg)[1],
        cfg=cfg,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
    )
    assert not torch.equal(result.positions[8:12], source.positions[8:12])
    assert torch.equal(result.positions[[0, 6, 7, 12]], source.positions[[0, 6, 7, 12]])
    before, _ = _analytic_kinematics(source.positions[:, [2, 0, 3, 1]])
    after, _ = _analytic_kinematics(result.positions[:, [2, 0, 3, 1]])
    torch.testing.assert_close(
        after[:7, [0, 2], 3], before[:7, [0, 2], 3], atol=1e-6, rtol=0
    )


def test_task_space_combines_with_live_nullspace_and_retiming() -> None:
    source = _analytic_template()
    source = replace(
        source,
        phases=(
            replace(
                source.phases[0],
                allowed_operators=source.phases[0].allowed_operators
                + ("nullspace_residual",),
            ),
            source.phases[1],
        ),
        allowed_operators=source.allowed_operators + ("nullspace_residual",),
    )
    payload = _analytic_cfg(False, True, False).to_dict()
    payload["factors"]["task_space"]["translation_axes"] = [True, False, False]
    payload["factors"]["ik"]["enabled"] = True
    payload["factors"]["timing"].update(enabled=True, duration_scales=[1.0, 1.5])
    result = expand_trajectory_variants(
        source,
        cfg=TrajectoryAugmentationCfg.from_mapping(payload),
        case=case(),
        count=3,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
    )
    assert len(result.variants) == 3
    assert result.variants[1].spatial_operator == "nullspace_residual"
    lengths = result.candidates.valid_length.tolist()
    assert lengths[2] > lengths[1]
    for row in (1, 2):
        contact = result.candidates.phases[row][1]
        assert torch.equal(
            result.candidates.positions[row, contact.start_index : contact.stop_index],
            source.positions[9:],
        )
        assert torch.equal(
            result.candidates.positions[row, lengths[row] - 1], source.positions[-1]
        )
        assert not torch.equal(
            result.candidates.positions[row, : lengths[row], 1],
            torch.zeros(lengths[row], dtype=torch.float64),
        )


def test_apply_rejects_a_guidance_band_outside_the_configured_partition() -> None:
    cfg = _analytic_cfg(False, False, True)
    variant = replace(plan_trajectory_variants(2, cfg)[1], manipulability_band=3)
    with pytest.raises(ValueError, match="index the configured bands"):
        apply_trajectory_variant(
            _analytic_template(),
            variant,
            cfg=cfg,
            joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
            control_dt=0.1,
            kinematics=_analytic_kinematics,
        )


def test_unconfigured_posture_application_can_derive_jacobians_from_kinematics() -> (
    None
):
    source = _analytic_template()
    source = replace(
        source,
        phases=(
            replace(
                source.phases[0],
                allowed_operators=source.phases[0].allowed_operators
                + ("nullspace_residual",),
            ),
            source.phases[1],
        ),
        allowed_operators=source.allowed_operators + ("nullspace_residual",),
    )
    posture = next(
        v
        for v in plan_trajectory_variants(4, default_variant_factors(redundancy=True))
        if v.spatial_operator == "nullspace_residual"
    )
    result = apply_trajectory_variant(
        source,
        posture,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 5),
        control_dt=0.1,
        kinematics=_analytic_kinematics,
    )
    assert not torch.equal(result.positions[:, 1], source.positions[:, 1])
    assert torch.equal(result.positions[[0, 8, 9, 12]], source.positions[[0, 8, 9, 12]])
