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
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenerator,
    MotionGenOptions,
)
from embodichain.lab.sim.motion.ik import (
    resolve_continuous_ik_path,
    sample_cartesian_targets,
)
from embodichain.lab.sim.motion.search import search_cost_scope
from embodichain.lab.sim.motion.planners.utils import PlanState
from embodichain.lab.task_program.integrations.configured import _decode_motion_policy

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/fixed_grid_reference.json").read_text()
)


def _generator(tmp_path, *, batch=1, speed=1.0, failure=None):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test"><joint name="joint" type="revolute">'
        '<limit velocity="10"/></joint></robot>'
    )
    seeds = []

    def solve(*, pose, joint_seed, name):
        assert name == "arm"
        index = len(seeds)
        seeds.append(joint_seed.clone())
        torch.rand(1)
        qpos = joint_seed + pose[:, :1, 3]
        valid = torch.ones(batch, dtype=torch.bool)
        if index == 1 and failure == "invalid":
            valid[1] = False
        if index == 1 and failure == "nonfinite":
            qpos[1] = float("nan")
        return valid, qpos

    generator = object.__new__(MotionGenerator)
    generator.device = torch.device("cpu")
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(planner_type="toppra"))
    generator.robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(urdf)),
        joint_names=["joint"],
        control_parts={"arm": ["joint"]},
        num_instances=batch,
        get_joint_ids=lambda name=None: [0],
        get_qvel_limits=lambda **kwargs: torch.full((batch, 1), speed),
        get_qpos_limits=lambda **kwargs: torch.tensor([[[-1.0, 1.0]]]).expand(
            batch, -1, -1
        ),
        compute_ik=solve,
    )
    return generator, seeds


@pytest.mark.parametrize("cap_owner", ["runtime", "urdf"])
def test_fixed_grid_holds_failed_commands_and_preserves_candidate_for_retry(
    tmp_path, cap_owner
):
    generator, _ = _generator(tmp_path, speed=1.0 if cap_owner == "runtime" else 10.0)
    if cap_owner == "urdf":
        path = Path(generator.robot.cfg.fpath)
        path.write_text(path.read_text().replace('velocity="10"', 'velocity="1"'))
    options = MotionGenOptions(
        strategy="ik_interp",
        sample_count=3,
        interpolation_dt=0.1,
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        interpolation_mode="fixed_grid",
    )
    targets = [PlanState.from_qpos(torch.tensor([[0.6]]))]
    attempts = []
    for reference in _REFERENCE["fixed_grid"]:
        result = generator.generate(
            targets, replace(options, sample_count=reference["sample_count"])
        )
        attempts.append(result)
        assert result.success.tolist() == reference["success"]
        torch.testing.assert_close(
            result.dt, torch.tensor(reference["dt"]), rtol=0, atol=0
        )
        if result.is_all_success():
            assert result.rejected_candidate is None
            torch.testing.assert_close(
                result.positions, torch.tensor(reference["positions"]), rtol=0, atol=0
            )
            break
        assert torch.count_nonzero(result.positions) == 0
        rejected = result.rejected_candidate
        assert rejected.env_mask.tolist() == [True]
        torch.testing.assert_close(
            rejected.positions, torch.tensor(reference["positions"]), rtol=0, atol=0
        )
        assert not joint_velocity_validity(
            rejected.positions, rejected.dt, torch.ones(1, 1)
        ).any()
        # A retry decision must inspect the candidate, not the safe zero-speed hold.
        assert joint_velocity_validity(
            result.positions, result.dt, torch.ones(1, 1)
        ).all()
        result.positions.fill_(9.0)
        torch.testing.assert_close(
            rejected.positions, torch.tensor(reference["positions"]), rtol=0, atol=0
        )
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "case", _REFERENCE["sequential"], ids=lambda case: str(case["failure"])
)
def test_sequential_public_generation_preserves_frozen_seeds_rng_and_failure_rows(
    tmp_path, case
):
    generator, seeds = _generator(
        tmp_path, batch=2, speed=10.0, failure=case["failure"]
    )
    poses = torch.eye(4).repeat(2, 3, 1, 1)
    poses[:, :, 0, 3] = torch.tensor([0.1, 0.2, 0.3])
    options = MotionGenOptions(
        strategy="ik_interp",
        sample_count=4,
        interpolation_dt=0.1,
        control_part="arm",
        start_qpos=torch.tensor([[0.0], [1.0]]),
        preserve_cartesian_samples=True,
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        result = generator.generate(
            [PlanState.from_xpos(poses[:, i]) for i in range(3)], options
        )
        assert float(torch.rand(1)) == case["next_random"]
    assert result.success.tolist() == case["success"]
    for actual, expected in zip(seeds, case["seeds"], strict=True):
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
    torch.testing.assert_close(
        result.positions, torch.tensor(case["positions"]), rtol=0, atol=0
    )


def test_explicit_policy_round_trip_and_closed_decoder():
    policy = _decode_motion_policy(
        {
            "sample_count": 8,
            "interpolation_mode": "fixed_grid",
            "ik_seed_mode": "sequential",
            "acceleration_limit": None,
        },
        path="motion",
    )
    options = policy.to_motion_gen_options(
        start_qpos=torch.zeros(1, 1), control_part="arm", interpolation_dt=0.1
    )
    assert (
        options.interpolation_mode == "fixed_grid"
        and options.ik_seed_mode == "sequential"
    )
    assert MotionPolicy().interpolation_mode == "adaptive"
    assert MotionPolicy().acceleration_limit == 0.5
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_motion_policy(
            {"sample_count": 8, "seed_policy": "guess"}, path="motion"
        )


def test_continuous_path_adapter_restores_rng_and_keeps_requested_timing(tmp_path):
    generator, seeds = _generator(tmp_path, speed=10.0)
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 0.1
    targets = sample_cartesian_targets(
        torch.eye(4)[None], [PlanState.from_xpos(pose)], 3
    )
    options = MotionGenOptions(
        strategy="ik_interp",
        sample_count=3,
        interpolation_dt=0.1,
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
    )
    before = torch.get_rng_state().clone()
    with search_cost_scope() as cost:
        result = resolve_continuous_ik_path(generator, targets, options)
    assert torch.equal(torch.get_rng_state(), before)
    assert result.is_all_success()
    torch.testing.assert_close(
        result.positions, torch.tensor([[[0.0], [0.05], [0.15]]])
    )
    torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.1, 0.1]]))
    assert len(seeds) == 2 and cost.counters == {"ik_solve": 2}
    with pytest.raises(ValueError, match="one explicit timed"):
        resolve_continuous_ik_path(
            generator, targets, replace(options, start_qpos=torch.zeros(2, 1))
        )


@pytest.mark.parametrize("owner", [MotionPolicy, MotionGenOptions])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"interpolation_mode": "fixed_grid", "acceleration_limit": 0.5},
        {
            "interpolation_mode": "fixed_grid",
            "velocity_limit": 0.2,
            "acceleration_limit": None,
        },
        {"strategy": "motion_gen", "ik_seed_mode": "sequential"},
        {
            "strategy": "motion_gen",
            "interpolation_mode": "fixed_grid",
            "acceleration_limit": None,
        },
        {"ik_seed_mode": "unknown"},
    ],
)
def test_explicit_interpolation_modes_reject_incompatible_contracts(owner, kwargs):
    with pytest.raises(ValueError):
        owner(**{"strategy": "ik_interp", **kwargs})
