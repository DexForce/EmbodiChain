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

from dataclasses import replace
import math

import pytest
import torch

from embodichain.lab.sim.motion.expansion.contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.motion.expansion.operators import (
    ProposalRejected,
    joint_residual,
    nullspace_residual,
    perturb_approach_direction,
    retime,
    rotate_grasp_about_object_axis,
    validate_motion_limits,
    via_points,
)


def test_grasp_rotation_uses_object_origin_and_preserves_inputs() -> None:
    object_pose = torch.eye(4)
    object_pose[:3, 3] = torch.tensor([2.0, 3.0, 4.0])
    grasp = object_pose.clone()
    grasp[0, 3] += 0.2
    grasp[2, 3] += 0.1
    before = grasp.clone()
    candidates = rotate_grasp_about_object_axis(
        object_pose,
        grasp,
        axis=torch.tensor([0.0, 0.0, 2.0]),
        angles=torch.tensor([0.0, math.pi / 2]),
    )
    torch.testing.assert_close(candidates[0], grasp)
    torch.testing.assert_close(candidates[1, :3, 3], torch.tensor([2.0, 3.2, 4.1]))
    torch.testing.assert_close(
        candidates[1, :3, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(grasp, before)
    candidates[0, 0, 3] = 100
    torch.testing.assert_close(grasp, before)


def test_grasp_variants_follow_rotated_object_frame() -> None:
    # The object's local Z points along scene X. Quarter-turn augmentation
    # must orbit that axis, even after translating the entire scene.
    object_pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 1.0, 0.0, -2.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    grasp = object_pose.clone()
    grasp[2, 3] -= 0.2
    candidate = rotate_grasp_about_object_axis(
        object_pose,
        grasp,
        axis=torch.tensor([0.0, 0.0, 1.0]),
        angles=torch.tensor([math.pi / 2]),
    )[0]
    torch.testing.assert_close(candidate[:3, 3], torch.tensor([3.0, -1.8, 1.0]))
    torch.testing.assert_close(candidate[:3, :3].T @ candidate[:3, :3], torch.eye(3))
    assert torch.linalg.det(candidate[:3, :3]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad", ["reflection", "zero_axis", "axis_shape", "nan_angle", "empty_angles"]
)
def test_grasp_rotation_rejects_invalid_geometry(bad: str) -> None:
    pose = torch.eye(4)
    axis = torch.tensor([0.0, 0.0, 1.0])
    angles = torch.zeros(1)
    if bad == "reflection":
        pose[0, 0] = -1
    elif bad == "zero_axis":
        axis.zero_()
    elif bad == "axis_shape":
        axis = torch.ones(2)
    elif bad == "nan_angle":
        angles[0] = torch.nan
    elif bad == "empty_angles":
        angles = torch.zeros(0)
    with pytest.raises(ValueError):
        rotate_grasp_about_object_axis(pose, torch.eye(4), axis=axis, angles=angles)


def template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="reference",
        source_revision="one",
        template_id="pick",
        joint_names=("arm", "gripper"),
        positions=torch.tensor(
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.1], [0.5, 0.1]]
        ),
        dt=torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        phases=(
            TrajectoryPhase(
                "transit", 0, 3, allowed_operators=("retime", "joint_residual")
            ),
            TrajectoryPhase("grasp", 3, 6, kind="contact"),
        ),
        controlled_joint_indices=(0,),
        allowed_operators=("retime", "joint_residual"),
    )


def test_retime_changes_length_preserves_contact_duration_and_endpoints() -> None:
    source = template()
    result = retime(source, duration_scale=2.0, control_dt=0.1)
    assert result.positions.shape == (8, 2)
    assert result.phases[0].stop_index == 5
    assert result.phases[1].start_index == 5
    torch.testing.assert_close(result.positions[4], source.positions[2])
    torch.testing.assert_close(result.positions[5:], source.positions[3:])
    torch.testing.assert_close(result.dt[6:].sum(), source.dt[4:].sum())
    torch.testing.assert_close(source.positions, template().positions)


def test_spatial_residual_preserves_locked_samples_and_uncontrolled_joints() -> None:
    source = template()
    generator = torch.Generator().manual_seed(5)
    result = joint_residual(
        source,
        joint_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        normalized_scale=0.05,
        generator=generator,
    )
    assert result.positions[1, 0] != source.positions[1, 0]
    assert torch.equal(
        result.positions[[0, 2, 3, 4, 5]], source.positions[[0, 2, 3, 4, 5]]
    )
    assert torch.equal(result.positions[:, 1], source.positions[:, 1])


def test_explicit_generator_is_reproducible_without_consuming_global_rng() -> None:
    before = torch.random.get_rng_state().clone()
    kwargs = dict(
        joint_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]), normalized_scale=0.05
    )
    first = joint_residual(
        template(), generator=torch.Generator().manual_seed(2), **kwargs
    )
    second = joint_residual(
        template(), generator=torch.Generator().manual_seed(2), **kwargs
    )
    assert torch.equal(first.positions, second.positions)
    assert torch.equal(before, torch.random.get_rng_state())


def test_unlabelled_qpos_cannot_be_augmented() -> None:
    with pytest.raises(ValueError, match="annotated"):
        retime(replace(template(), phases=()), duration_scale=2, control_dt=0.1)


def test_retime_does_not_silently_move_uncontrolled_schedule() -> None:
    source = template()
    source.positions[1, 1] = 0.01
    with pytest.raises(ValueError, match="uncontrolled"):
        retime(source, duration_scale=2, control_dt=0.1)


def test_retime_rejects_clock_mismatch_and_unbounded_allocation() -> None:
    with pytest.raises(ValueError, match="clock"):
        retime(template(), duration_scale=2, control_dt=0.07)
    with pytest.raises(ValueError, match="budget"):
        retime(template(), duration_scale=1000, control_dt=0.1, max_samples=10)


def test_dynamic_limits_reject_overfast_motion_but_do_not_claim_task_success() -> None:
    source = template()
    failed = validate_motion_limits(
        source,
        velocity_limits=torch.tensor([1.0, 2.0]),
        acceleration_limits=torch.full((2,), 100.0),
    )
    assert not failed.accepted
    slow = retime(source, duration_scale=4, control_dt=0.1)
    passed = validate_motion_limits(
        slow,
        velocity_limits=torch.tensor([1.0, 2.0]),
        acceleration_limits=torch.full((2,), 100.0),
    )
    assert passed.accepted
    assert passed.checks[0].check_id == "motion_limits"


def test_contact_internal_event_cannot_be_removed_by_host_clock() -> None:
    source = replace(template(), dt=torch.tensor([0.0, 0.1, 0.1, 0.1, 0.05, 0.15]))
    with pytest.raises(ValueError, match="clock"):
        retime(source, duration_scale=2, control_dt=0.1)


def test_large_derivatives_return_a_failed_check() -> None:
    source = replace(
        template(), positions=template().positions * 1e38, dt=template().dt * 1e-3
    )
    result = validate_motion_limits(
        source, velocity_limits=torch.ones(2), acceleration_limits=torch.ones(2)
    )
    assert not result.accepted
    assert result.checks[0].metrics["speed_ratio"] > 1e30


def test_half_precision_derivatives_are_accumulated_in_high_precision() -> None:
    source = replace(
        template(), positions=template().positions.half(), dt=template().dt.half()
    )
    result = validate_motion_limits(
        source,
        velocity_limits=torch.full((2,), 20.0),
        acceleration_limits=torch.full((2,), 0.0001),
    )
    assert not result.accepted
    assert result.checks[0].metrics["acceleration_ratio"] > 65504


def path_template(samples: int = 21, joints: int = 4) -> TrajectoryTemplate:
    """A longer free phase followed by a locked contact phase."""
    positions = torch.zeros(samples, joints)
    positions[:, 0] = torch.linspace(0.0, 1.0, samples)
    dt = torch.full((samples,), 0.1)
    dt[0] = 0
    free_stop = samples - 4
    operators = ("via_points", "nullspace_residual", "joint_residual")
    return TrajectoryTemplate(
        source_id="reference",
        source_revision="one",
        template_id="reach",
        joint_names=tuple(f"joint_{index}" for index in range(joints)),
        positions=positions,
        dt=dt,
        phases=(
            TrajectoryPhase("transit", 0, free_stop, "free", operators),
            TrajectoryPhase("grasp", free_stop, samples, kind="contact"),
        ),
        allowed_operators=operators,
        controlled_joint_indices=tuple(range(joints)),
    )


def wide_limits(joints: int = 4) -> torch.Tensor:
    return torch.stack((torch.full((joints,), -3.0), torch.full((joints,), 3.0)), dim=1)


def test_via_points_preserve_phase_endpoints_and_the_contact_phase() -> None:
    source = path_template()
    result = via_points(
        source,
        joint_limits=wide_limits(),
        via_count=3,
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    free, contact = result.phases
    torch.testing.assert_close(
        result.positions[free.start_index], source.positions[free.start_index]
    )
    torch.testing.assert_close(
        result.positions[free.stop_index - 1], source.positions[free.stop_index - 1]
    )
    torch.testing.assert_close(
        result.positions[contact.start_index :], source.positions[contact.start_index :]
    )
    assert not torch.allclose(result.positions, source.positions)


def sign_changes_along_phase(
    template: TrajectoryTemplate, via_count: int, seed: int
) -> int:
    """Count sign changes of one joint's residual inside the free phase."""
    source = path_template(samples=25)
    result = via_points(
        template,
        joint_limits=wide_limits(),
        via_count=via_count,
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    free = source.phases[0]
    window = (result.positions - source.positions)[
        free.start_index : free.stop_index, 0
    ]
    return int((window[1:].sign() * window[:-1].sign() < 0).sum())


def test_only_several_knots_can_reverse_the_residual_along_a_phase() -> None:
    source = path_template(samples=25)
    seeds = range(12)
    # One knot is a single signed lobe, so it can never reverse. Several knots
    # can, because their magnitudes are sampled independently; whether a given
    # draw does is a property of the sample, not of every seed.
    assert all(sign_changes_along_phase(source, 1, seed) == 0 for seed in seeds)
    assert any(sign_changes_along_phase(source, 4, seed) >= 1 for seed in seeds)
    single = via_points(
        source,
        joint_limits=wide_limits(),
        via_count=1,
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(3),
    )
    many = via_points(
        source,
        joint_limits=wide_limits(),
        via_count=4,
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(3),
    )
    assert not torch.allclose(single.positions, many.positions)


def test_via_point_knots_share_one_joint_direction() -> None:
    # Independent per-joint sampling decorrelates the joints and makes the tool
    # path wander; every knot must displace along one direction.
    source = path_template(samples=25)
    result = via_points(
        source,
        joint_limits=wide_limits(),
        via_count=4,
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    free = source.phases[0]
    residual = (result.positions - source.positions)[free.start_index : free.stop_index]
    controlled = residual[:, list(source.controlled_joint_indices)]
    # A rank-one residual has one nonzero singular value; the rest sit at the
    # float32 rounding of the stored positions.
    spectrum = torch.linalg.svdvals(controlled.to(torch.float64))
    assert float(spectrum[1] / spectrum[0]) < 1e-6


def test_via_points_reject_unusable_counts_and_short_phases() -> None:
    source = path_template()
    generator = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(ValueError, match="via_count"):
        via_points(
            source,
            joint_limits=wide_limits(),
            via_count=0,
            normalized_scale=0.05,
            generator=generator,
        )
    with pytest.raises(ValueError, match="interior sample per knot"):
        via_points(
            source,
            joint_limits=wide_limits(),
            via_count=40,
            normalized_scale=0.05,
            generator=generator,
        )
    with pytest.raises(ValueError, match="joint limits"):
        via_points(
            source,
            joint_limits=torch.stack((torch.zeros(4), torch.full((4,), 1e-3)), dim=1),
            via_count=2,
            normalized_scale=1.0,
            generator=generator,
        )


def test_nullspace_residual_holds_the_declared_task_rows() -> None:
    source = path_template()
    samples, joints = source.positions.shape
    jacobians = torch.zeros(samples, 2, joints)
    jacobians[:, 0, 0] = 1.0
    jacobians[:, 1, 1] = 1.0
    result = nullspace_residual(
        source,
        task_jacobians=jacobians,
        joint_limits=wide_limits(),
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    torch.testing.assert_close(result.positions[:, :2], source.positions[:, :2])
    assert not torch.allclose(result.positions[:, 2:], source.positions[:, 2:])
    free = source.phases[0]
    torch.testing.assert_close(
        result.positions[free.stop_index - 1], source.positions[free.stop_index - 1]
    )


@pytest.mark.parametrize("kind", ["identity", "random", "ill_conditioned"])
def test_nullspace_residual_refuses_any_fully_constrained_task(kind: str) -> None:
    # A generic full-rank Jacobian leaves floating-point residue in
    # ``I - pinv(J) @ J``, so a magnitude probe would pass it through and emit a
    # near-unchanged variant. Only a rank test rejects all three.
    source = path_template()
    samples, joints = source.positions.shape
    generator = torch.Generator(device="cpu").manual_seed(0)
    if kind == "identity":
        jacobians = torch.eye(joints).expand(samples, joints, joints).contiguous()
    else:
        square = torch.randn(samples, joints, joints, generator=generator)
        jacobians = square + (3.0 if kind == "ill_conditioned" else 0.0)
    assert int(torch.linalg.matrix_rank(jacobians.double())[0]) == joints
    with pytest.raises(ValueError, match="redundancy"):
        nullspace_residual(
            source,
            task_jacobians=jacobians,
            joint_limits=wide_limits(),
            normalized_scale=0.05,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )


def test_nullspace_residual_accepts_a_rank_deficient_task() -> None:
    source = path_template()
    samples, joints = source.positions.shape
    # Five constrained rows on four joints still leave one redundant direction
    # once the rows are linearly dependent.
    rows = torch.randn(
        samples, joints - 1, joints, generator=torch.Generator().manual_seed(2)
    )
    result = nullspace_residual(
        source,
        task_jacobians=rows,
        joint_limits=wide_limits(),
        normalized_scale=0.05,
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    assert not torch.allclose(result.positions, source.positions)


def test_a_limit_violation_is_a_rejection_not_a_malformed_input() -> None:
    # A generation loop retries on a rejected draw; it must not retry past a
    # caller error, so the two carry different exception types.
    source = path_template()
    tight = torch.stack((torch.zeros(4), torch.full((4,), 1e-3)), dim=1)
    with pytest.raises(ProposalRejected):
        via_points(
            source,
            joint_limits=tight,
            via_count=2,
            normalized_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
    with pytest.raises(ValueError) as malformed:
        via_points(
            source,
            joint_limits=wide_limits(),
            via_count=40,
            normalized_scale=0.05,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
    assert not isinstance(malformed.value, ProposalRejected)


def test_nullspace_residual_requires_aligned_jacobians() -> None:
    source = path_template()
    with pytest.raises(ValueError, match="matching the template samples"):
        nullspace_residual(
            source,
            task_jacobians=torch.zeros(3, 2, 4),
            joint_limits=wide_limits(),
            normalized_scale=0.05,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )


def test_timing_profiles_keep_the_path_and_total_duration() -> None:
    source = template()
    uniform = retime(source, duration_scale=2.0, control_dt=0.1)
    eased = retime(source, duration_scale=2.0, control_dt=0.1, profile="ease_in")
    assert uniform.positions.shape == eased.positions.shape
    torch.testing.assert_close(uniform.dt.sum(), eased.dt.sum())
    assert not torch.allclose(uniform.positions, eased.positions)
    # Phase boundaries stay on the same control steps, so waypoints never move.
    assert [phase.start_index for phase in uniform.phases] == [
        phase.start_index for phase in eased.phases
    ]
    torch.testing.assert_close(
        eased.positions[eased.phases[1].start_index :],
        uniform.positions[uniform.phases[1].start_index :],
    )


def test_ease_in_and_ease_out_are_opposite_and_bounded() -> None:
    source = template()
    free = source.phases[0]
    reference = retime(source, duration_scale=3.0, control_dt=0.1)
    slow_start = retime(source, duration_scale=3.0, control_dt=0.1, profile="ease_in")
    fast_start = retime(source, duration_scale=3.0, control_dt=0.1, profile="ease_out")
    stop = reference.phases[0].stop_index
    midpoint = (free.start_index + stop) // 2
    assert (
        slow_start.positions[midpoint, 0]
        < reference.positions[midpoint, 0]
        < fast_start.positions[midpoint, 0]
    )


def test_unknown_timing_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="profile must be one of"):
        retime(template(), duration_scale=2.0, control_dt=0.1, profile="bounce")


def test_approach_cone_keeps_the_contact_pose_and_standoff_distance() -> None:
    contact = torch.eye(4)
    contact[:3, 3] = torch.tensor([0.5, -0.1, 0.3])
    before = contact.clone()
    standoffs = perturb_approach_direction(
        contact,
        approach_axis=torch.tensor([0.0, 0.0, 2.0]),
        standoff_distance=0.12,
        polar_angles=torch.tensor([0.0, 0.4, 0.4]),
        azimuth_angles=torch.tensor([0.0, 0.0, math.pi / 2]),
    )
    assert standoffs.shape == (3, 4, 4)
    torch.testing.assert_close(contact, before)
    torch.testing.assert_close(
        standoffs[0, :3, 3], torch.tensor([0.5, -0.1, 0.18]), atol=1e-6, rtol=0
    )
    for index in range(3):
        offset = standoffs[index, :3, 3] - contact[:3, 3]
        torch.testing.assert_close(
            torch.linalg.vector_norm(offset), torch.tensor(0.12), atol=1e-6, rtol=0
        )
    # Different azimuths tilt in different planes around the nominal axis.
    assert not torch.allclose(standoffs[1], standoffs[2])


def test_aligned_standoff_travels_along_its_own_approach_axis() -> None:
    contact = torch.eye(4)
    contact[:3, 3] = torch.tensor([0.4, 0.0, 0.25])
    axis = torch.tensor([0.0, 0.0, 1.0])
    aligned = perturb_approach_direction(
        contact,
        approach_axis=axis,
        standoff_distance=0.1,
        polar_angles=torch.tensor([0.35]),
        azimuth_angles=torch.tensor([0.8]),
    )[0]
    travelled = aligned[:3, 3] + 0.1 * (aligned[:3, :3] @ axis)
    torch.testing.assert_close(travelled, contact[:3, 3], atol=1e-6, rtol=0)
    unaligned = perturb_approach_direction(
        contact,
        approach_axis=axis,
        standoff_distance=0.1,
        polar_angles=torch.tensor([0.35]),
        azimuth_angles=torch.tensor([0.8]),
        align_tool=False,
    )[0]
    # Both variants stand off in the same tilted direction; only the tool
    # orientation differs, so an unaligned approach is a pure translation.
    torch.testing.assert_close(unaligned[:3, :3], contact[:3, :3])
    torch.testing.assert_close(unaligned[:3, 3], aligned[:3, 3])
    assert not torch.allclose(unaligned[:3, :3], aligned[:3, :3])


def test_approach_cone_rejects_malformed_inputs() -> None:
    contact = torch.eye(4)
    with pytest.raises(ValueError, match="same length"):
        perturb_approach_direction(
            contact,
            approach_axis=torch.tensor([0.0, 0.0, 1.0]),
            standoff_distance=0.1,
            polar_angles=torch.tensor([0.0, 0.1]),
            azimuth_angles=torch.tensor([0.0]),
        )
    with pytest.raises(ValueError, match="standoff_distance"):
        perturb_approach_direction(
            contact,
            approach_axis=torch.tensor([0.0, 0.0, 1.0]),
            standoff_distance=0.0,
            polar_angles=torch.tensor([0.0]),
            azimuth_angles=torch.tensor([0.0]),
        )
    with pytest.raises(ValueError, match="nonzero norm"):
        perturb_approach_direction(
            contact,
            approach_axis=torch.zeros(3),
            standoff_distance=0.1,
            polar_angles=torch.tensor([0.0]),
            azimuth_angles=torch.tensor([0.0]),
        )
