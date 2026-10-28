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

"""Frozen paired requests, independent scoring and isolated dispatch timings."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners import PlanResult, PlanState
from embodichain_benchmarks.motion_generation import _paired_arm_performance as paired
from embodichain_benchmarks.motion_generation.config import (
    PlannerSpecCfg,
    ProtocolCfg,
    SuiteCfg,
    TrackCfg,
)
from embodichain_benchmarks.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    PlannerMetadata,
)


class _Robot:
    device = torch.device("cpu")
    num_instances = 1

    def get_qpos_limits(self, *, name):
        return torch.tensor([[[-2.0, 2.0]]])

    def get_qvel_limits(self, *, name):
        return torch.tensor([[2.0]])

    def compute_batch_fk(self, *, qpos, name, to_matrix):
        pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        pose[..., 0, 3] = qpos[..., 0]
        return pose


def _case():
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    pose[..., 0, 3] = 0.2
    return BenchmarkCase(
        suite_version="paired_test",
        track="atomic",
        scenario_id="hand_over",
        case_id="source_a",
        seed=11,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="nominal",
        start_qpos=torch.zeros(1, 1),
        target_waypoints=pose,
        reference_qpos=torch.tensor([[[0.2]]]),
        full_start_qpos=torch.zeros(1, 2),
        skill_id="hand_over",
    )


class _Generator:
    def __init__(self, packed, wrong_right=False):
        self.supports_multi_part = packed
        self.wrong_right = wrong_right
        self.calls = []

    def generate(self, states, options):
        self.calls.append(options.control_part)
        start = options.start_qpos.clone()
        goal = states[-1].xpos[:, :1, 3]
        if self.wrong_right and options.control_part == "right_arm":
            goal = goal - 0.1
        result = PlanResult(
            success=torch.ones(1, dtype=torch.bool),
            positions=torch.stack([start, goal], dim=1),
            dt=torch.tensor([[0.0, 0.25]]),
        )
        # Backends may normalize caller-owned inputs. This must not corrupt
        # cached targets, another mode or the next repeated call.
        states[-1].xpos.add_(10)
        options.start_qpos.add_(10)
        return result

    def generate_multi_part(self, targets, options):
        return {
            part: self.generate(states, options[part])
            for part, states in targets.items()
        }


class _Adapter:
    separate_prepare = False

    def __init__(self, spec, context):
        self.spec, self.context = spec, context
        self.timing_is_solved = spec.id != "nmg"
        self.generator = _Generator(spec.adapter == "nmg_onnx", spec.id == "bad")
        self.closed = False
        self.metadata = PlannerMetadata(
            algorithm_id=spec.id,
            algorithm_role=AlgorithmRole.CANDIDATE,
            adapter=spec.adapter,
            config_hash="test",
            capabilities=frozenset(),
        )

    def availability(self):
        return True, None

    def build(self):
        pass

    def close(self):
        self.closed = True

    def require_motion_generator(self):
        return self.generator

    def atomic_plan_options(self, *, control_dt=None):
        return None


class _Provider:
    def prepare_planner(self, adapter, case):
        self.adapter = adapter

    def close_planner(self, adapter):
        self.adapter = None

    def reset_case(self, *args):
        pass

    def plan_contract_error(self, result):
        return None

    def plan_case(self, adapter, case):
        targets = {
            part: [PlanState.from_xpos(case.target_waypoints[:, 0].clone())]
            for part in ("left_arm", "right_arm")
        }
        options = {
            part: MotionGenOptions(
                control_part=part,
                start_qpos=case.start_qpos.clone(),
                interpolation_dt=0.16,
            )
            for part in targets
        }
        return adapter.generator.generate_multi_part(targets, options)


def _runner(monkeypatch):
    specs = [
        PlannerSpecCfg(id="toppra", adapter="toppra"),
        PlannerSpecCfg(id="nmg", adapter="nmg_onnx"),
        PlannerSpecCfg(id="bad", adapter="other"),
    ]
    case = _case()
    robot, provider = _Robot(), _Provider()
    suite = SuiteCfg(
        name="dual_test",
        planners=specs,
        protocol=ProtocolCfg(validation_samples=16),
        tracks=[TrackCfg(id="atomic", scenario="atomic_skill")],
    )
    runner = SimpleNamespace(
        suite=suite,
        planner_specs=specs,
        replay_cases=[case],
        device=torch.device("cpu"),
        control_part="left_arm",
        adapters={},
        execution={"timing_comparable": False},
        embodiment_resolution=SimpleNamespace(
            to_metadata=lambda: {"embodiment_id": "test", "config_hash": "fixed"}
        ),
    )

    def batches():
        yield object(), robot, provider, deepcopy(runner.replay_cases)

    runner._case_batches = batches
    adapters = []

    def create(spec, context, **kwargs):
        adapter = _Adapter(spec, context)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(paired, "create_planner_adapter", create)
    return runner, adapters


def test_paired_replay_freezes_inputs_scores_both_arms_and_keeps_modes_separate(
    tmp_path, monkeypatch
):
    runner, adapters = _runner(monkeypatch)
    request_path = tmp_path / "frozen.json"
    result = paired._run(
        runner,
        tmp_path / "result",
        request_manifest=request_path,
        warmup_trials=1,
        measured_trials=2,
        source_cases_per_suite=1,
    )
    assert result["status"] == "completed" and result["case_count"] == 1
    assert result["execution"]["timing_comparable"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "result/trials.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 8  # 2 classical modes + 2 NMG modes, twice each.
    nmg = [row for row in rows if row["algorithm"] == "nmg"]
    assert [row["dispatch"] for row in nmg] == [
        "sequential",
        "packed",
        "packed",
        "sequential",
    ]
    assert all(row["dispatch_parity"]["equivalent"] for row in nmg)
    assert all(row["outcomes"]["left_arm"][0]["geometry_valid"] for row in rows)
    assert all(
        not row["outcomes"]["right_arm"][0]["geometry_valid"]
        for row in rows
        if row["algorithm"] == "bad"
    )
    quality = json.loads((tmp_path / "result/aggregates.json").read_text())[
        "paired_arm_success"
    ]
    assert (
        next(row for row in quality if row["algorithm"] == "bad")["success_rate"] == 0
    )
    assert len(quality) == 4
    assert (
        sum(
            line.startswith("| ---")
            for line in (tmp_path / "result/report.md").read_text().splitlines()
        )
        == 3
    )
    assert all(adapter.closed for adapter in adapters)
    assert (tmp_path / "result/resolved_suite.yaml").is_file()
    assert (
        len(json.loads((tmp_path / "result/planners.json").read_text())["actual"]) == 3
    )
    frozen = json.loads(request_path.read_text())
    assert frozen["requests"][0]["parts"]["left_arm"]["start_qpos"]["values"] == [[0.0]]


def test_paired_cache_reuse_never_recaptures_and_rejects_changed_source(
    tmp_path, monkeypatch
):
    runner, _ = _runner(monkeypatch)
    request_path = tmp_path / "frozen.json"
    kwargs = dict(
        request_manifest=request_path,
        warmup_trials=1,
        measured_trials=1,
        source_cases_per_suite=1,
    )
    first = paired._run(runner, tmp_path / "first", **kwargs)
    contents = request_path.read_bytes()
    monkeypatch.setattr(
        paired, "_capture", lambda *args: pytest.fail("immutable cache was regenerated")
    )
    second = paired._run(runner, tmp_path / "second", **kwargs)
    assert second["case_sha256"] == first["case_sha256"]
    assert request_path.read_bytes() == contents
    original_limit = runner.suite.protocol.joint_acceleration_limit_rad_s2
    runner.suite.protocol.joint_acceleration_limit_rad_s2 = 1.7
    with pytest.raises(ValueError, match="source identity"):
        paired._run(runner, tmp_path / "different_budget", **kwargs)
    runner.suite.protocol.joint_acceleration_limit_rad_s2 = original_limit
    runner.replay_cases[0].start_qpos.add_(0.1)
    with pytest.raises(ValueError, match="source identity"):
        paired._run(runner, tmp_path / "third", **kwargs)


def test_paired_cache_rejects_corruption_before_execution(tmp_path, monkeypatch):
    runner, _ = _runner(monkeypatch)
    request_path = tmp_path / "frozen.json"
    kwargs = dict(
        request_manifest=request_path,
        warmup_trials=1,
        measured_trials=1,
        source_cases_per_suite=1,
    )
    paired._run(runner, tmp_path / "first", **kwargs)
    payload = json.loads(request_path.read_text())
    payload["requests"][0]["parts"]["right_arm"]["start_qpos"]["values"][0][0] = 1.0
    request_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        paired._run(runner, tmp_path / "second", **kwargs)
