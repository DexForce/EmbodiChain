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

"""Public Cartesian segment recovery against the frozen source control flow."""

from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from embodichain.lab.sim.motion import _cartesian as cartesian
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenOptions,
    MotionGenerator,
)
from ._cartesian_segment_fixture import segment_fixture, real_segment_fixture
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.motion._budget import _planning_budget_scope

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/cartesian_segment_reference.json").read_text()
)["cases"]


@pytest.mark.parametrize("reference", _REFERENCE, ids=lambda item: item["case"]["name"])
def test_segment_sampling_retries_preserve_source_counts_paths_and_rng(
    reference, monkeypatch
):
    case = reference["case"]
    generator, targets, calls, continuous, recover = segment_fixture(case)
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", recover)
    options = MotionGenOptions(
        strategy="ik_interp",
        is_linear=True,
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        sample_count=case["count"],
        start_qpos=torch.zeros(1, 1),
        control_part="arm",
        interpolation_dt=0.04,
        segment_sample_count_retries=(260, 320),
        segment_continuous_ik_recovery=True,
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    result = generator.generate(targets, options, defer_recovery=case["defer"])
    assert isinstance(result, PlanResult)
    value = result.recovery
    assert value is not None
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    actual = {
        "case": case,
        "calls": calls,
        "continuous": continuous,
        "used_continuity": value.used_continuity,
        "success": result.success.tolist(),
        "positions_sha256": digest(result.positions),
        "dt_sha256": digest(result.dt),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    assert actual == reference
    assert value.attempted_counts == tuple(call["count"] for call in calls)
    assert (
        options.segment_sample_count_retries == (260, 320)
        and not options.preserve_cartesian_samples
    )


def _options(**changes):
    return MotionGenOptions(
        **dict(
            strategy="ik_interp",
            is_linear=True,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            sample_count=4,
            start_qpos=torch.zeros(1, 1),
            control_part="arm",
            interpolation_dt=0.04,
            segment_sample_count_retries=(260, 320),
            segment_continuous_ik_recovery=True,
        )
        | changes
    )


@pytest.mark.parametrize("outcome", ["nominal", "dense", "continuous", "failed"])
def test_unified_generate_keeps_normalized_success_timing_and_failed_holds(
    tmp_path, monkeypatch, outcome
):
    generator, targets, calls, _, recover = real_segment_fixture(
        tmp_path, {"outcome": outcome}
    )
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", recover)
    options = _options()
    before = options.start_qpos.clone()
    result = generator.generate(targets, options)
    assert isinstance(result, PlanResult) and result.recovery is not None
    assert result.recovery.attempted_counts == tuple(call["count"] for call in calls)
    assert result.recovery.used_continuity == (outcome == "continuous")
    assert result.positions.shape[1] == (260 if outcome == "dense" else 4)
    assert result.dt[0, 0] == 0 and torch.all(result.dt[0, 1:] == 0.04)
    assert result.success.tolist() == [outcome != "failed"]
    assert result.velocities is not None
    assert torch.equal(result.velocities[:, (0, -1)], torch.zeros(1, 2, 1))
    if outcome == "failed":
        assert torch.equal(result.positions, torch.zeros_like(result.positions))
        assert torch.equal(result.velocities, torch.zeros_like(result.velocities))
    else:
        assert result.positions[0, -1, 0] == pytest.approx(0.2)
    assert torch.equal(options.start_qpos, before)
    assert options.sample_count == 4 and options.segment_sample_count_retries == (
        260,
        320,
    )
    assert not options.preserve_cartesian_samples


def test_linear_keyframes_work_without_spending_recovery():
    generator, targets, calls, _, _ = segment_fixture({"outcome": "nominal"})
    result = generator.generate(
        targets,
        _options(segment_sample_count_retries=(), segment_continuous_ik_recovery=False),
    )
    assert isinstance(result, PlanResult)
    assert result.recovery.attempted_counts == (4,)
    assert len(calls) == 1 and calls[0]["preserve"]


def test_default_generate_uses_original_single_attempt_without_linear_sampling():
    generator, targets, calls, _, _ = segment_fixture({"outcome": "nominal"})

    def forbidden(**kwargs):
        raise AssertionError("Default generation must not acquire a new FK observation")

    generator.robot.compute_fk = forbidden
    result = generator.generate(targets, MotionGenOptions(sample_count=4))
    assert isinstance(result, PlanResult) and result.recovery is None
    assert len(calls) == 1 and not calls[0]["preserve"]


class _Budget:
    def __init__(self, limit=10, cancelled=False):
        self.limit, self.cancelled, self.charges = limit, cancelled, []

    def check(self):
        if self.cancelled:
            raise RuntimeError("cancelled")

    def charge(self, kind, amount=1):
        self.check()
        if len(self.charges) + amount > self.limit:
            raise RuntimeError("quota exhausted")
        self.charges.extend([kind] * amount)


@pytest.mark.parametrize("defer", [False, True])
def test_unified_generation_charges_actual_attempts_once(tmp_path, defer):
    generator, targets, calls, _, _ = real_segment_fixture(
        tmp_path, {"outcome": "dense"}
    )
    budget = _Budget()
    with _planning_budget_scope(budget):
        result = generator.generate(targets, _options(), defer_recovery=defer)
    expected = (4,) if defer else (4, 260)
    assert result.recovery.attempted_counts == expected
    assert len(calls) == len(expected)
    assert budget.charges == ["planning"] * len(expected)


def test_quota_exhaustion_prevents_the_next_actual_planning_attempt(tmp_path):
    generator, targets, calls, _, _ = real_segment_fixture(
        tmp_path, {"outcome": "dense"}
    )
    budget = _Budget(limit=1)
    with (
        pytest.raises(RuntimeError, match="quota exhausted"),
        _planning_budget_scope(budget),
    ):
        generator.generate(targets, _options())
    assert [call["count"] for call in calls] == [4]
    assert budget.charges == ["planning"]


def test_cancelled_generation_rejects_before_fk_or_ik(tmp_path):
    generator, targets, calls, _, _ = real_segment_fixture(
        tmp_path, {"outcome": "dense"}
    )

    def forbidden(**kwargs):
        raise AssertionError("Cancelled generation cannot acquire FK")

    generator.robot.compute_fk = forbidden
    budget = _Budget()
    with _planning_budget_scope(budget):
        budget.cancelled = True
        with pytest.raises(RuntimeError, match="cancelled"):
            generator.generate(targets, _options())
        budget.cancelled = False
    assert not calls


def test_recovery_rejects_unsupported_batch_before_reading_fk():
    generator, targets, calls, _, _ = segment_fixture({"outcome": "nominal"})

    def forbidden(**kwargs):
        raise AssertionError("Unsupported batch must reject before FK")

    generator.robot.compute_fk = forbidden
    with pytest.raises(ValueError, match="B1 only"):
        generator.generate(targets, _options(start_qpos=torch.zeros(2, 1)))
    assert not calls


@pytest.mark.parametrize(
    "changes", [{"is_linear": False}, {"preserve_cartesian_samples": True}]
)
def test_recovery_requires_explicit_unsampled_linear_keyframes(changes):
    with pytest.raises(ValueError, match="unsampled linear keyframes"):
        _options(**changes)


@pytest.mark.parametrize("defer", [False, True])
def test_linear_recovery_keeps_fixed_grid_speed_gate_and_original_rejection(
    tmp_path, defer
):
    generator, targets, _, _, _ = real_segment_fixture(tmp_path, {"outcome": "nominal"})
    generator.robot.cfg.fpath = str(tmp_path / "slow.urdf")
    Path(generator.robot.cfg.fpath).write_text(
        '<robot><joint name="joint"><limit velocity="0.1"/></joint></robot>'
    )
    result = generator.generate(
        targets, _options(segment_continuous_ik_recovery=False), defer_recovery=defer
    )
    assert result.success.tolist() == [not defer]
    assert result.recovery.attempted_counts == ((4,) if defer else (4, 260))
    if defer:
        assert torch.equal(result.positions, torch.zeros(1, 4, 1))
        assert torch.equal(
            result.rejected_candidate.positions,
            torch.linspace(0, 0.2, 4).reshape(1, 4, 1),
        )
        assert torch.equal(result.dt, result.rejected_candidate.dt)
    else:
        assert result.positions.shape == (1, 260, 1)
        assert result.positions[0, -1, 0] == pytest.approx(0.2)


def test_exhausted_recovery_retains_original_rejected_grid_not_last_dense_hold(
    tmp_path,
):
    generator, targets, _, _, _ = real_segment_fixture(tmp_path, {"outcome": "nominal"})
    generator.robot.cfg.fpath = str(tmp_path / "slow.urdf")
    Path(generator.robot.cfg.fpath).write_text(
        '<robot><joint name="joint"><limit velocity="0.0001"/></joint></robot>'
    )
    result = generator.generate(targets, _options(segment_continuous_ik_recovery=False))
    assert result.success.tolist() == [False]
    assert result.recovery.attempted_counts == (4, 260, 320)
    assert result.positions.shape == (1, 4, 1)
    assert result.rejected_candidate.positions.shape == (1, 4, 1)
    assert torch.equal(
        result.rejected_candidate.positions, torch.linspace(0, 0.2, 4).reshape(1, 4, 1)
    )
    assert torch.equal(result.positions, torch.zeros_like(result.positions))


def test_unified_generate_uses_real_continuous_solver_and_restores_rng(tmp_path):
    generator, targets, _, _, _ = real_segment_fixture(tmp_path, {"outcome": "failed"})
    del generator._generate_ik_interpolation
    generator.robot.get_qpos_limits = lambda **kwargs: torch.tensor([[[-1.0, 1.0]]])

    def solve(*, pose, joint_seed, **kwargs):
        # Nominal zero seeds fail; the continuous solver's bounded alternate
        # seed can reach the same Cartesian target without moving the robot.
        valid = (joint_seed != 0).any(dim=1)
        return valid, pose[:, 0, 3:4].clone()

    generator.robot.compute_ik = solve
    torch.manual_seed(23)
    before = torch.get_rng_state().clone()
    result = generator.generate(targets, _options(segment_sample_count_retries=(5, 6)))
    assert result.success.tolist() == [True]
    assert result.recovery.attempted_counts == (4, 5, 6)
    assert result.recovery.used_continuity
    torch.testing.assert_close(
        result.positions, torch.linspace(0, 0.2, 4).reshape(1, 4, 1)
    )
    assert torch.equal(result.dt, torch.tensor([[0.0, 0.04, 0.04, 0.04]]))
    assert result.velocities is not None
    assert torch.equal(torch.get_rng_state(), before)
