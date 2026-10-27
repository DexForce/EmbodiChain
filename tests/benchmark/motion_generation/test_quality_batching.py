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

"""Original cases, grids and failures survive execution-only NMG batching."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.planners.neural_planner import NeuralPlanner
from embodichain.lab.sim.motion.planners.toppra_planner import retime_joint_paths
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain_benchmarks.motion_generation import run_benchmark
from embodichain_benchmarks.motion_generation._quality_batching import (
    case_groups,
    pack_cases,
    split_result,
)
from embodichain_benchmarks.motion_generation.artifacts import write_case_manifest
from embodichain_benchmarks.motion_generation.config import (
    PlannerSpecCfg,
    SuiteCfg,
    TrackCfg,
)
from embodichain_benchmarks.motion_generation.models import BenchmarkCase
from embodichain_benchmarks.motion_generation.embodiments import resolve_embodiment
from embodichain_benchmarks.motion_generation.planners.nmg_onnx import NmgOnnxAdapter
from embodichain_benchmarks.motion_generation.planners import nmg_onnx as nmg_module
from embodichain_benchmarks.motion_generation import runner as runner_module

pytestmark = pytest.mark.no_sim


def _case(index: int) -> BenchmarkCase:
    """Use two completed paths of different lengths and one valid failure."""
    start = torch.zeros(1, 7)
    start[0, 1] = index * 0.05
    target = start[:, None].clone()
    target[0, 0, 0] = [0.2, 0.4, 1.0][index]
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    pose[0, 0, :3, 3] = target[0, 0, :3]
    return BenchmarkCase(
        suite_version="batch-test",
        track="geometry",
        scenario_id="free_space",
        case_id=f"original-{index}",
        seed=101 + index,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="frozen",
        start_qpos=start,
        target_waypoints=pose,
        reference_qpos=target,
        case_parameters={
            "constraint_modalities": ["position"],
            "joint_threshold_rad": 0.01,
        },
    )


def test_raw_split_restores_endpoint_stencils_and_owns_its_output() -> None:
    """An interior padded state becomes the original one-sided endpoint."""
    trace = torch.tensor(
        [[[0.0], [0.1], [0.3], [0.3], [0.3]], [[0.0], [0.1], [0.2], [0.3], [0.4]]]
    )
    dt = torch.full((2, 5), 0.1)
    dt[:, 0] = 0
    velocities, accelerations = NeuralPlanner._compute_vel_acc_via_finite_diff(
        trace, dt
    )
    result = PlanResult(
        success=torch.tensor([True, False]),
        positions=trace,
        dt=dt,
        velocities=velocities,
        accelerations=accelerations,
        policy_rollout_positions=trace,
        policy_rollout_lengths=torch.tensor([3, 5]),
    )
    row = split_result(result, 0, retimed=False)
    torch.testing.assert_close(row.velocities.flatten(), torch.tensor([1.0, 1.5, 2.0]))
    torch.testing.assert_close(
        row.accelerations.flatten(), torch.tensor([10.0, 10.0, 10.0])
    )
    assert row.positions.shape == (1, 3, 1) and row.policy_rollout_lengths.tolist() == [
        3
    ]
    result.positions.zero_()
    assert row.positions[0, -1, 0].item() == pytest.approx(0.3)
    assert row.policy_rollout_positions[0, -1, 0].item() == pytest.approx(0.3)


def test_retimed_split_matches_independent_singleton_time_grids() -> None:
    """Real retiming preserves a short row, a long row and stationary output."""
    trace = torch.tensor(
        [
            [[0.0], [0.1], [0.1], [0.1]],
            [[0.0], [0.1], [0.3], [0.5]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )
    lengths = torch.tensor([2, 4, 2], dtype=torch.int32)
    kwargs = dict(
        constraints={"velocity": 1.0, "acceleration": 2.0},
        sample_interval=0.02,
        device="cpu",
    )
    batched = retime_joint_paths(trace, path_lengths=lengths, **kwargs)
    batched = replace(
        batched, policy_rollout_positions=trace, policy_rollout_lengths=lengths
    )
    for index, n in enumerate(lengths.tolist()):
        original = retime_joint_paths(trace[index : index + 1, :n], **kwargs)
        split = split_result(batched, index, retimed=True)
        for name in ("positions", "dt", "velocities", "accelerations"):
            torch.testing.assert_close(
                getattr(split, name), getattr(original, name), rtol=1e-5, atol=1e-6
            )


def test_grouping_preserves_semantics_without_changing_original_inputs() -> None:
    cases = [_case(i) for i in range(3)]
    different = replace(
        cases[0],
        case_id="orientation",
        case_parameters={"constraint_modalities": ["orientation"]},
    )
    groups = list(case_groups([cases[0], different, *cases[1:]], 2))
    assert [[case.case_id for case in group] for group in groups] == [
        ["original-0", "original-1"],
        ["original-2"],
        ["orientation"],
    ]
    full_start = replace(
        cases[0], case_id="full-state", full_start_qpos=torch.zeros(1, 9)
    )
    assert len(list(case_groups([cases[0], full_start], 4))) == 2
    packed = pack_cases(groups[1], 4)
    assert packed.batch_size == 4 and cases[2].batch_size == 1
    assert packed.start_qpos.shape == (4, 7)
    torch.testing.assert_close(packed.start_qpos, cases[2].start_qpos.expand(4, -1))
    packed.start_qpos.zero_()
    assert cases[2].start_qpos[0, 1].item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Performance"):
        list(
            case_groups(
                [
                    replace(
                        cases[0],
                        case_parameters={"measurement_purpose": "performance_replay"},
                    )
                ],
                2,
            )
        )


@pytest.mark.parametrize("fixed_batch_size", [None, 1])
@pytest.mark.parametrize("retimed", [False, True])
def test_public_batch_replay_preserves_original_records_and_scores(
    tmp_path, monkeypatch, retimed, fixed_batch_size
) -> None:
    """Exercise real runner/scoring with explicit per-row FK and padding."""
    mapped = []
    planned_batches = []
    closed_adapters = []
    destroyed_simulations = []

    class Robot:
        device = torch.device("cpu")
        uid = "quality-test"

        def __init__(self, n):
            self.num_instances = n

        def set_qpos(self, *args, **kwargs):
            pass

        def clear_dynamics(self):
            pass

        def get_qpos_limits(self, name):
            return torch.tensor([[[-2.0, 2.0]] * 7])

        def get_qvel_limits(self, name):
            return torch.ones(1, 7) * 2

        def compute_fk(self, qpos, name, to_matrix, env_ids=None):
            assert qpos.shape[0] == (
                self.num_instances if env_ids is None else len(env_ids)
            )
            if env_ids is not None:
                mapped.append(tuple(env_ids))
            pose = torch.eye(4).repeat(qpos.shape[0], 1, 1)
            pose[:, :3, 3] = qpos[:, :3]
            return pose

        def compute_batch_fk(self, qpos, name, to_matrix, env_ids=None):
            assert qpos.shape[0] == (
                self.num_instances if env_ids is None else len(env_ids)
            )
            if env_ids is not None:
                mapped.append(tuple(env_ids))
            pose = torch.eye(4).repeat(qpos.shape[0], qpos.shape[1], 1, 1)
            pose[..., :3, 3] = qpos[..., :3]
            return pose

    def simulation(cfg):
        return SimpleNamespace(
            robot=Robot(cfg.num_envs),
            update=lambda **kw: None,
            sync_render_state=lambda: None,
            destroy=lambda **kw: destroyed_simulations.append(cfg.num_envs),
        )

    simulation.flush_cleanup_queue = lambda: None
    monkeypatch.setattr(runner_module, "SimulationManager", simulation)
    monkeypatch.setattr(
        runner_module,
        "create_robot_provider",
        lambda spec: SimpleNamespace(
            control_part="arm", add_robot=lambda sim: sim.robot
        ),
    )

    monkeypatch.setattr(
        nmg_module,
        "MotionGenerator",
        lambda cfg: SimpleNamespace(
            planner=SimpleNamespace(
                _policy=SimpleNamespace(fixed_batch_size=fixed_batch_size),
                close=lambda: closed_adapters.append(True),
            )
        ),
    )

    class Adapter(NmgOnnxAdapter):
        capabilities = frozenset(
            {
                "eef_waypoint",
                "joint_waypoint",
                "masked_constraints",
                "batched",
                "empty_world",
            }
        )

        def availability(self):
            return True, None

        def plan(self, case):
            planned_batches.append(case.batch_size)
            if fixed_batch_size is not None and case.batch_size != fixed_batch_size:
                raise ValueError("ONNX policy has incompatible fixed batch size")
            indices = (case.start_qpos[:, 1] * 20).round().long().tolist()
            lengths = torch.tensor([[2, 4, 5][i] for i in indices], dtype=torch.int32)
            n = int(lengths.max())
            paths = [
                [0.0, 0.2, 0.2, 0.2, 0.2],
                [0.0, 0.1, 0.3, 0.4, 0.4],
                [0.0, 0.1, 0.2, 0.3, 0.4],
            ]
            q = case.start_qpos[:, None].repeat(1, n, 1)
            q[:, :, 0] = torch.tensor([paths[i][:n] for i in indices])
            dt = torch.full(q.shape[:2], 0.1)
            dt[:, 0] = 0
            v, a = NeuralPlanner._compute_vel_acc_via_finite_diff(q, dt)
            if retimed:
                for row, length in enumerate(lengths.tolist()):
                    dt[row, length:] = 0
            return PlanResult(
                success=torch.tensor([i != 2 for i in indices]),
                positions=q,
                dt=dt,
                velocities=v,
                accelerations=a,
                policy_rollout_positions=q,
                policy_rollout_lengths=lengths,
            )

    suite = SuiteCfg(
        suite_version="batch-test",
        planners=[
            PlannerSpecCfg(
                id="nmg",
                adapter="nmg_onnx",
                enabled=True,
                config={
                    "model_revision": "fixture",
                    "onnx_model_path": "fixture.onnx",
                    "retime": retimed,
                },
            )
        ],
        tracks=[
            TrackCfg(id="geometry", scenario="free_space", config={"batch_sizes": [1]})
        ],
    )
    suite.protocol.warmup_trials = 0
    suite.protocol.measured_trials = 1
    suite.protocol.validation_samples = 32
    suite.protocol.joint_acceleration_limit_rad_s2 = 0.5
    cases = [_case(i) for i in range(3)]
    source = write_case_manifest(
        tmp_path / "originals.json",
        cases,
        suite=suite,
        embodiment_resolution=resolve_embodiment(
            suite.embodiment, suite.robot, base_dir=tmp_path
        ),
    )

    def run(capacity):
        return run_benchmark(
            suite,
            adapters={"nmg_onnx": Adapter},
            device="cpu",
            case_manifest=source,
            quality_batch_size=capacity,
            output_root=tmp_path / f"batch-{capacity}",
        )

    singleton = run(1)
    assert all(record.status == "ok" for record in singleton.records)
    if fixed_batch_size == 1:
        calls_before = len(planned_batches)
        with pytest.raises(
            ValueError, match="quality_batch_size=4.*fixed batch size 1"
        ):
            run(4)
        assert len(planned_batches) == calls_before
        assert len(closed_adapters) == len(destroyed_simulations) == 2
        trials = next((tmp_path / "batch-4").rglob("trials.jsonl"))
        records = [json.loads(line) for line in trials.read_text().splitlines()]
        assert all(record["status"] == "ok" for record in records)
        assert all(record["phase"] == "construct" for record in records)
        fixed_batch_size = 4
    results = [singleton, run(4)]
    measured = [
        {r.case_id: r for r in result.records if r.phase.value == "measured"}
        for result in results
    ]
    assert set(measured[0]) == set(measured[1]) == {case.case_id for case in cases}
    for case in cases:
        before, after = [records[case.case_id] for records in measured]
        assert before.status == after.status == "ok", after.failure_message
        assert asdict(before.outcomes[0]) == asdict(after.outcomes[0])
        assert after.batch_size == 1 and after.seed == case.seed
        assert after.cost_time_ms is None and after.execution_time_ms is None
        assert after.metadata["benchmark_execution"]["timing_comparable"] is False
    assert sum("batch_measurement" in r.metadata for r in measured[1].values()) == 1
    manifests = [
        json.loads((result.run_dir / "case_manifest.json").read_text())
        for result in results
    ]
    assert manifests[0]["cases"] == manifests[1]["cases"]
    assert len(manifests[1]["cases"]) == 3
    assert all(case["batch_size"] == 1 for case in manifests[1]["cases"])
    assert {0, 1, 2} <= {row[0] for row in mapped}
