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

"""Real2sim templates use the public candidate contracts and local RNG only."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from embodichain.data_pipeline.real2sim import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
    MimicJoint,
)
from embodichain.data_pipeline.real2sim.normalization import (
    Real2SimNormalizationCfg,
    normalize_real2sim,
)
from embodichain.compute.trajectory import differentiate_positions
from embodichain.lab.sim.motion.expansion import (
    Real2SimSourceAdapter,
    SourceContext,
    SourceAdapter,
    SceneCase,
)
from embodichain.lab.sim.motion.expansion.operators import joint_residual


def _unit():
    q = np.c_[
        np.linspace(0.0, 0.3, 20),
        np.linspace(0.0, 0.1, 20) ** 2,
        np.r_[np.zeros(10), np.linspace(0.0, 0.5, 5), np.full(5, 0.5)],
    ]
    raw = Real2SimTrajectory(
        "fixture",
        "r1",
        "pick-0",
        "robot",
        "v1",
        ("arm", "elbow", "finger"),
        ("rad", "rad", "rad"),
        q,
        np.arange(20) * 0.01,
        "s",
        "cal1",
        "ph1",
        "verified",
        "verified",
        (
            SourcePhase("approach", 0, 10, "free", ("joint_residual",)),
            SourcePhase("close", 10, 15, "contact"),
            SourcePhase("hold", 15, 20, "hold"),
        ),
    )
    layout = RobotJointLayout(
        "robot",
        "v1",
        ("arm", "elbow", "finger", "mimic"),
        ("rad", "rad", "rad", "rad"),
        np.zeros(4),
        ("arm", "elbow"),
        (MimicJoint("mimic", "finger", -1.0),),
    )
    return normalize_real2sim(raw, layout, Real2SimNormalizationCfg(control_dt=0.01))


def _context(unit):
    return SourceContext(
        unit.source.source_id,
        unit.revision,
        unit.source.unit_id,
        SceneCase("case", "initial", "scene1", "pick", "profile1"),
        0.01,
    )


def test_two_variants_preserve_contacts_hand_mimics_and_boundary_velocities():
    unit = _unit()
    adapter = Real2SimSourceAdapter()
    assert isinstance(adapter, SourceAdapter)
    template = adapter.export_template(unit, context=_context(unit))
    limits = torch.tensor(
        [[-2.0, 2.0], [-2.0, 2.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=template.positions.dtype,
    )
    variants = []
    for seed in (7, 11):
        variant = joint_residual(
            template,
            joint_limits=limits,
            normalized_scale=0.002,
            generator=torch.Generator().manual_seed(seed),
        )
        repeated = joint_residual(
            template,
            joint_limits=limits,
            normalized_scale=0.002,
            generator=torch.Generator().manual_seed(seed),
        )
        assert torch.equal(variant.positions, repeated.positions)
        assert torch.equal(variant.positions[10:], template.positions[10:])
        assert torch.equal(variant.positions[:, 2:], template.positions[:, 2:])
        assert torch.equal(
            variant.positions[[0, 1, 8, 9]], template.positions[[0, 1, 8, 9]]
        )
        v = differentiate_positions(variant.positions[None], variant.dt[None])[0]
        original = differentiate_positions(template.positions[None], template.dt[None])[
            0
        ]
        assert torch.equal(v[[0, 9, 10, 14, 15, 19]], original[[0, 9, 10, 14, 15, 19]])
        variants.append(variant)
    assert not torch.equal(variants[0].positions, variants[1].positions)
    np.testing.assert_array_equal(unit.positions, template.positions.numpy())
    assert unit.phases[0].start == 0 and unit.phases[0].stop == 10
    assert template.phases[0].start_index == 1 and template.phases[0].stop_index == 9


def test_unverified_clock_mismatch_and_unsupported_retime_fail_closed():
    unit = _unit()
    context = _context(unit)
    adapter = Real2SimSourceAdapter()
    with pytest.raises(ValueError, match="control clocks"):
        adapter.export_template(unit, context=replace(context, control_dt=0.02))
    with pytest.raises(ValueError, match="identity"):
        adapter.export_template(unit, context=replace(context, source_revision="wrong"))
    unverified = replace(
        unit, report={**unit.report_dict(), "input_prerequisites_verified": False}
    )
    with pytest.raises(ValueError, match="Unverified"):
        adapter.export_template(unverified, context=context)
    phase = replace(unit.phases[0], allowed_operators=("retime",))
    with pytest.raises(ValueError, match="same-grid"):
        adapter.export_template(
            replace(unit, phases=(phase, *unit.phases[1:])), context=context
        )


def test_coupled_driver_expansion_is_not_silently_inconsistent():
    unit = _unit()
    with pytest.raises(ValueError, match="mimic-driver"):
        Real2SimSourceAdapter().export_template(
            replace(unit, expandable_joint_indices=(2,)), context=_context(unit)
        )


def test_existing_coordinator_generates_nominal_and_two_deterministic_variants():
    from embodichain.lab.sim.motion.expansion import (
        CandidateCoordinator,
        ExpansionSession,
        TrajectoryExpansionJobCfg,
    )

    unit = _unit()
    context = _context(unit)
    cfg = TrajectoryExpansionJobCfg.from_mapping(
        {
            "augmentation": {
                "seed": 7,
                "max_variants_per_reference": 3,
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual"],
                        "joint_offset_scale": 0.002,
                    }
                },
                "coverage": {"joint_dedup_normalized_tol": 1e-5},
            },
            "scheduling": {"candidate_budget": 8},
        }
    )
    limits = torch.tensor(
        [[-2.0, 2.0], [-2.0, 2.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64
    )

    def generate():
        session = ExpansionSession(cfg)
        session.register_case(context.scene_case, limits, joint_names=unit.joint_names)
        coordinator = CandidateCoordinator(
            cfg,
            session=session,
            source_adapter=Real2SimSourceAdapter(),
            source_context=context,
            joint_limits=limits,
        )
        return coordinator.enqueue_source(unit, count=3), session

    first, session = generate()
    second, _ = generate()
    assert len(first) == len(second) == 3
    for left, right in zip(first, second):
        assert left.spec.identity == right.spec.identity
        assert torch.equal(left.template.positions, right.template.positions)
    assert session.snapshot()["counts"]["proposed"] == 3
