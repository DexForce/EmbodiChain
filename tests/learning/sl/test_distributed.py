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

"""Bounded CPU Gloo coverage for shared-loss DDP and rank-local recovery."""

from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Requires CPU Gloo",
    ),
]


class _Transition(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, dtype=torch.float64)
        self.step_size = 0.25

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).tanh()


class _MaskedObjective:
    def __init__(self, multistep: bool = False) -> None:
        self.multistep = multistep

    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        if self.multistep:
            prediction = batch.inputs["state"]
            for action in batch.inputs["actions"].unbind(1):
                prediction = prediction + model.step_size * model(
                    torch.cat((prediction, action), -1)
                )
        else:
            prediction = model(batch.inputs)
        errors = (prediction - batch.targets).square()[batch.masks["valid"]]
        return LossResult(errors.mean(), {"mse": errors.mean()}, weight=errors.numel())


def _weighted_worker(rank: int) -> None:
    for multistep in (False, True):
        torch.manual_seed(73)
        model = _Transition()
        reference = copy.deepcopy(model)
        objective = _MaskedObjective(multistep)
        values = torch.arange(12, dtype=torch.float64).view(6, 2) / 10
        targets = torch.linspace(-0.4, 0.6, 6, dtype=torch.float64).view(6, 1)
        mask = torch.tensor([[True], [False], [False], [True], [True], [True]])
        if multistep:
            inputs = {
                "state": values[:, :1],
                "actions": torch.arange(18, dtype=torch.float64).view(6, 3, 1) / 20,
            }
            local_inputs = {
                key: value[rank * 3 : rank * 3 + 3] for key, value in inputs.items()
            }
        else:
            inputs = values
            local_inputs = values[rank * 3 : rank * 3 + 3]
        global_batch = SupervisedBatch(inputs, targets, {"valid": mask})
        # The reference differentiates the independent concatenated-batch loss.
        expected = objective.compute_loss(reference, global_batch)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.07)
        expected.loss.backward()
        expected_optimizer.step()
        learner = SupervisedLearner(
            model, objective, torch.optim.SGD(model.parameters(), lr=0.07)
        )
        assert learner.model is model
        report = learner.update(
            SupervisedBatch(
                local_inputs,
                targets[rank * 3 : rank * 3 + 3],
                {"valid": mask[rank * 3 : rank * 3 + 3]},
            )
        )
        assert report["weight"] == 4
        assert report["loss"] == pytest.approx(expected.loss.item(), abs=1e-14)
        assert report["mse"] == pytest.approx(expected.loss.item(), abs=1e-14)
        for actual, target in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, target, rtol=1e-12, atol=1e-14)
            copies = [torch.empty_like(actual) for _ in range(2)]
            dist.all_gather(copies, actual.detach())
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)


class _DropoutModel(torch.nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1)
        )
        self.register_buffer("seen", torch.zeros((), dtype=torch.int64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.seen.add_(self.rank + 1)
        return self.network(value)


class _Regression:
    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        error = (model(batch.inputs) - batch.targets).square()
        return LossResult(error.mean(), weight=error.numel())


class _StochasticSource:
    def __init__(self) -> None:
        values = torch.arange(40, dtype=torch.float32).view(20, 2) / 40
        self.source = TensorBatchSource(
            SupervisedBatch(values, values.sum(-1, keepdim=True))
        )
        self.source_id = self.source.source_id + ":python-numpy-noise-v1"

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        batch = self.source.sample(batch_size, generator=generator)
        offset = 0.01 * (random.random() + float(np.random.random()))
        return SupervisedBatch(batch.inputs + offset, batch.targets)


class _RankUpdateState:
    def __init__(self, rank: int) -> None:
        self.rank, self.updates, self.total = rank, 0, torch.zeros(())

    def on_update(self, learner: SupervisedLearner) -> None:
        self.updates += 1
        self.total += next(learner.model.parameters()).detach().sum() + self.rank

    def state_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "updates": self.updates, "total": self.total.clone()}

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state["rank"] != self.rank
            or type(state["updates"]) is not int
            or state["updates"] < 0
        ):
            raise ValueError("Rank-local update state differs")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.updates = state["updates"]
        self.total.copy_(state["total"])


def _build_recovery(path: Path, rank: int) -> SupervisedTrainer:
    random.seed(101 + rank)
    np.random.seed(201 + rank)
    torch.manual_seed(301 + rank)
    model = _DropoutModel(rank)
    learner = SupervisedLearner(
        model, _Regression(), torch.optim.Adam(model.parameters(), lr=0.01)
    )
    cfg = SupervisedTrainerCfg(
        total_updates=6,
        batch_size=4,
        seed=17,
        output_dir=str(path),
        checkpoint_interval=2,
        validation_interval=2,
    )
    trainer = SupervisedTrainer(
        learner,
        _StochasticSource(),
        cfg,
        signature={"regression": "rank-local recovery"},
        training_states={"rank_update": _RankUpdateState(rank)},
    )

    def evaluate(current: SupervisedLearner) -> dict[str, float]:
        assert rank == 0, "Validation must run only on rank zero"
        assert not current.model.training
        current.model(torch.ones(1, 2))
        random.random()
        np.random.random(3)
        torch.rand(5)
        torch.randint(10, (3,), generator=trainer.sampler)
        trainer.training_states["rank_update"].updates += 100
        # BC-style checkpoint publication inside a rank-zero callback must not hang.
        trainer.save(path / "selected.pt")
        with (path.parent / f"{path.name}-validation-rank{rank}.jsonl").open(
            "a"
        ) as handle:
            handle.write(json.dumps({"step": current.num_updates}) + "\n")
        return {"probe": float(current.num_updates)}

    trainer.evaluate = evaluate
    return trainer


def _snapshot(trainer: SupervisedTrainer) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "learner": trainer.learner.state_dict(),
            "training_states": {
                name: value.state_dict()
                for name, value in trainer.training_states.items()
            },
            "sampler": trainer.sampler.get_state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
    )


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            _assert_tree_equal(value, target)
    else:
        assert actual == expected


def _recovery_worker(rank: int, directory: Path, stage: str) -> None:
    trainer = _build_recovery(directory / stage, rank)
    expected_sampler = torch.Generator().manual_seed(trainer.cfg.seed + rank)
    torch.testing.assert_close(
        trainer.sampler.get_state(), expected_sampler.get_state()
    )
    if stage == "resumed":
        trainer.resume(directory / "partial/selected.pt")
        # Check rank-local buffers/RNG before a later forward could hide an error.
        partial = torch.load(directory / f"partial-rank{rank}.pt", weights_only=False)
        _assert_tree_equal(_snapshot(trainer), partial)
    trainer.fit(stop_after=2 if stage == "partial" else None)
    torch.save(_snapshot(trainer), directory / f"{stage}-rank{rank}.pt")


class _FailingSource:
    source_id = "collaborative-failure-source-v1"

    def __init__(self, rank: int, fails: bool) -> None:
        self.rank, self.fails = rank, fails

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        if self.fails and self.rank == 1:
            raise ValueError("injected-source-failure")
        return SupervisedBatch(torch.ones(batch_size, 2), torch.zeros(batch_size, 1))


class _FailingObjective(_Regression):
    def __init__(self, rank: int, failure: str) -> None:
        self.rank, self.failure = rank, failure

    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        if self.failure == "objective" and self.rank == 1:
            raise ValueError("injected-objective-failure")
        result = super().compute_loss(model, batch)
        if self.rank == 1 and self.failure == "loss":
            return LossResult(result.loss * float("nan"))
        if self.rank == 1 and self.failure == "weight":
            return LossResult(result.loss, weight=0)
        return result


def _failure_worker(rank: int, directory: Path) -> None:
    for phase in ("source", "objective", "loss", "weight", "validation"):
        torch.manual_seed(11)
        model = torch.nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        learner = SupervisedLearner(
            model,
            _FailingObjective(rank, phase),
            torch.optim.Adam(model.parameters(), lr=0.01),
        )

        def evaluate(current: SupervisedLearner) -> dict[str, float]:
            assert rank == 0
            raise ValueError("injected-validation-failure")

        trainer = SupervisedTrainer(
            learner,
            _FailingSource(rank, phase == "source"),
            SupervisedTrainerCfg(
                total_updates=2,
                batch_size=2,
                output_dir=str(directory / phase),
                checkpoint_interval=0,
                validation_interval=1 if phase == "validation" else 0,
            ),
            evaluate=evaluate if phase == "validation" else None,
        )
        sampler_before = trainer.sampler.get_state().clone()
        try:
            trainer.fit()
        except Exception as error:
            message = {
                "loss": "Non-finite supervised loss",
                "weight": "Objective weight",
            }.get(phase, f"injected-{phase}-failure")
            assert message in str(error)
        else:
            raise AssertionError(f"Rank {rank} did not observe the {phase} failure")
        assert learner.num_updates == (1 if phase == "validation" else 0)
        if phase != "validation":
            _assert_tree_equal(model.state_dict(), before)
            torch.testing.assert_close(trainer.sampler.get_state(), sampler_before)
            assert not learner.optimizer.state
        dist.barrier()


def _artifact_worker(rank: int, directory: Path) -> None:
    def build(output: Path) -> SupervisedTrainer:
        trainer = _build_recovery(output, rank)
        trainer.evaluate = None
        trainer.cfg.validation_interval = 0
        return trainer

    source = build(directory / "source")
    collision = build(source.output)
    source.fit(stop_after=2)
    original = {
        name: (source.output / name).read_bytes()
        for name in ("metrics.jsonl", "last.pt", "run.json")
    }
    with pytest.raises(RuntimeError, match="another run"):
        collision.fit(stop_after=1)
    for name, content in original.items():
        assert (source.output / name).read_bytes() == content

    foreign = build(directory / "foreign")
    foreign.fit(stop_after=1)
    candidate = build(foreign.output)
    before = _snapshot(candidate)
    with pytest.raises(RuntimeError, match="another run"):
        candidate.resume(source.output / "last.pt")
    _assert_tree_equal(_snapshot(candidate), before)

    for history in ("newer", "incomplete"):
        if rank == 0:
            (source.output / "metrics.jsonl").write_bytes(
                original["metrics.jsonl"]
                + (
                    b'{"num_updates": 3}\n'
                    if history == "newer"
                    else b'{"num_updates":'
                )
            )
        dist.barrier()
        log = (source.output / "metrics.jsonl").read_bytes()
        candidate = build(source.output)
        before = _snapshot(candidate)
        with pytest.raises(RuntimeError, match="history"):
            candidate.resume(source.output / "last.pt")
        _assert_tree_equal(_snapshot(candidate), before)
        assert (source.output / "metrics.jsonl").read_bytes() == log
        dist.barrier()

    # Previously published DDP checkpoints have no run ID. They still recover
    # exactly, but must acquire their new identity in a fresh destination.
    if rank == 0:
        state = torch.load(source.output / "last.pt", weights_only=False)
        state.pop("run_id")
        torch.save(state, directory / "legacy.pt")
    dist.barrier()
    recovered = build(directory / "legacy-recovered")
    recovered.resume(directory / "legacy.pt")
    recovered.fit(stop_after=3)
    saved = torch.load(recovered.output / "last.pt", weights_only=False)
    assert (
        saved["run_id"]
        == json.loads((recovered.output / "run.json").read_text())["run_id"]
    )
    assert saved["learner"]["num_updates"] == 3


def _source_resume_worker(rank: int, directory: Path) -> None:
    from .test_sequence import _build_trainer

    original = _build_trainer(directory / "original")
    original.fit(stop_after=2)
    for boundary in ("unverified", "quota"):
        candidate = _build_trainer(
            directory / boundary, immutable=not (rank == 1 and boundary == "unverified")
        )
        if rank == 1 and boundary == "quota":
            candidate.source.source.configure_sampling_budget(5, 5)
        before = _snapshot(candidate)
        run_id = candidate._artifacts.run_id
        with pytest.raises(RuntimeError, match="immutable|unbudgeted"):
            candidate.resume(original.output / "last.pt")
        _assert_tree_equal(_snapshot(candidate), before)
        assert not candidate.output.exists()
        assert not candidate._distributed_prepared
        assert candidate._artifacts.run_id == run_id
        assert not candidate._artifacts._claimed
        dist.barrier()


def _chunked_checkpoint_worker(rank: int) -> None:
    from embodichain.learning.sl.distributed import _DistributedContext

    context = _DistributedContext("cpu")
    # Slightly over 16 MiB, so this exercises multiple transfer chunks.
    value = {"rank": rank, "tensor": torch.arange(4_194_307, dtype=torch.float32)}
    states = context.gather(value)
    if rank == 0:
        assert states is not None
        assert [state["rank"] for state in states] == list(range(dist.get_world_size()))
        for state in states[1:]:
            torch.testing.assert_close(state["tensor"], value["tensor"], rtol=0, atol=0)
    else:
        assert states is None

    with pytest.raises(RuntimeError, match="checkpoint state serialization"):
        context.gather((lambda: None) if rank == 1 else value)


def _distributed_worker(
    rank: int, group_file: str, directory: str, scenario: str, world_size: int
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{group_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        if scenario == "weighted":
            _weighted_worker(rank)
        elif scenario == "errors":
            _failure_worker(rank, Path(directory))
        elif scenario == "artifacts":
            _artifact_worker(rank, Path(directory))
        elif scenario == "source-resume":
            _source_resume_worker(rank, Path(directory))
        elif scenario == "chunked-checkpoint":
            _chunked_checkpoint_worker(rank)
        else:
            _recovery_worker(rank, Path(directory), scenario)
    finally:
        dist.destroy_process_group()


def _spawn(scenario: str, directory: Path, world_size: int = 2) -> None:
    context = mp.spawn(
        _distributed_worker,
        args=(
            str(directory / f"group-{scenario}"),
            str(directory),
            scenario,
            world_size,
        ),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 100
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(f"Gloo {scenario} exceeded 100 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def test_unequal_weight_single_and_multistep_match_global_batch(tmp_path: Path) -> None:
    _spawn("weighted", tmp_path)


def test_chunked_checkpoint_gathers_and_propagates_failures(tmp_path: Path) -> None:
    _spawn("chunked-checkpoint", tmp_path, world_size=3)


def test_restart_preserves_rank_rng_sampling_buffers_and_single_writer(
    tmp_path: Path,
) -> None:
    for stage in ("full", "partial", "resumed"):
        _spawn(stage, tmp_path)
    for rank in range(2):
        full = torch.load(tmp_path / f"full-rank{rank}.pt", weights_only=False)
        resumed = torch.load(tmp_path / f"resumed-rank{rank}.pt", weights_only=False)
        _assert_tree_equal(resumed, full)
    full_lines = (tmp_path / "full/metrics.jsonl").read_text()
    combined = (tmp_path / "partial/metrics.jsonl").read_text() + (
        tmp_path / "resumed/metrics.jsonl"
    ).read_text()
    assert combined == full_lines
    records = [json.loads(line) for line in full_lines.splitlines()]
    assert [record["num_updates"] for record in records] == list(range(1, 7))
    assert all(record["weight"] == 8 for record in records)
    assert (tmp_path / "full/last.pt").is_file()
    assert len((tmp_path / "full-validation-rank0.jsonl").read_text().splitlines()) == 3
    assert not list(tmp_path.glob("*-validation-rank1.jsonl"))
    # A real two-rank checkpoint cannot silently become a one-rank run.
    candidate = _build_recovery(tmp_path / "different-world-size", rank=0)
    before = _snapshot(candidate)
    with pytest.raises((ValueError, RuntimeError), match="[Ww]orld.?size"):
        candidate.resume(tmp_path / "partial/selected.pt")
    _assert_tree_equal(_snapshot(candidate), before)


def test_rank_failures_reach_all_workers_without_partial_update(tmp_path: Path) -> None:
    _spawn("errors", tmp_path)


def test_output_identity_and_history_admission_reaches_all_ranks(
    tmp_path: Path,
) -> None:
    _spawn("artifacts", tmp_path)


def test_source_resume_preflight_rejection_reaches_all_ranks_without_mutation(
    tmp_path: Path,
) -> None:
    _spawn("source-resume", tmp_path)


def test_torchrun_cli_initializes_cleans_up_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = {
        "factory": "_distributed_recipe:build",
        "trainer": {
            "total_updates": 4,
            "batch_size": 3,
            "seed": 23,
            "device": "cpu",
            "output_dir": str(output),
            "checkpoint_interval": 0,
        },
        "test_audit_dir": str(tmp_path),
        "test_phase": "partial",
        "test_stop_after": 2,
    }
    config_path = tmp_path / "recipe.json"
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None, (str(Path(__file__).resolve().parent), child_env.get("PYTHONPATH"))
        )
    )
    for phase in ("partial", "resumed"):
        config["test_phase"] = phase
        config["test_stop_after"] = 2 if phase == "partial" else None
        config_path.write_text(json.dumps(config))
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "embodichain",
            "train-sl",
            "--config",
            str(config_path),
        ]
        if phase == "resumed":
            command.extend(["--resume", str(output / "last.pt")])
        with subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=100)
            except subprocess.TimeoutExpired:
                # The new process group contains only this test's launcher/workers.
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                pytest.fail(f"torchrun {phase} exceeded 100 seconds:\n{stdout}{stderr}")
            assert process.returncode == 0, stdout + stderr
        for rank in range(2):
            audit = json.loads((tmp_path / f"{phase}-rank{rank}.json").read_text())
            assert audit == {
                "initialized_in_factory": True,
                "initialized_at_exit": False,
            }
    metrics = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["num_updates"] for row in metrics] == [1, 2, 3, 4]
    assert all(row["weight"] == 6 for row in metrics)
