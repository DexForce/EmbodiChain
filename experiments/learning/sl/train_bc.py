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

"""Fixed-corpus visual BC acceptance using the shared supervised trainer.

Training/development/test budgets follow PR828's declared convergence recipe.
This offline recipe has no ODS engine, background producer or source quotas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import torch

from embodichain.learning.models.chunk_bc import ChunkBCModel, ChunkBCModelCfg
from embodichain.learning.sl import LossResult, SupervisedBatch, SupervisedLearner
from embodichain.learning.sl.bc import ChunkBCObjective, HDF5BCSource
from embodichain.learning.sl.trainer import SupervisedTrainer, SupervisedTrainerCfg
from .pick_place import CAMERAS, TARGET_SEMANTICS, frozen_statistics, source


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _rate(step: int, updates: int = 20000) -> float:
    if step <= 500:
        return 3e-4 * max(1, step) / 500
    progress = min(1.0, (step - 500) / (updates - 500))
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress)) / 2


class _DeadlineObjective(ChunkBCObjective):
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline

    def compute_loss(self, model: ChunkBCModel, batch: SupervisedBatch) -> LossResult:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Declared BC run deadline exhausted")
        return super().compute_loss(model, batch)


def _evaluate(
    checkpoint: Path,
    output: Path,
    *,
    scenes: int,
    seed: int,
    config: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    command = [
        sys.executable,
        "-m",
        "experiments.learning.sl.evaluate_pick_place",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--scenes",
        str(scenes),
        "--seed",
        str(seed),
        "--sim-device",
        evaluation["sim_device"],
        "--policy-device",
        evaluation["policy_device"],
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = evaluation["visible_devices"]
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = min(3600.0, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("No BC worker budget remains for evaluation")
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(
            f"Independent BC evaluator failed with code {code}; see {output.with_suffix('.log')}"
        )
    report = json.loads(output.read_text())
    if (
        report.get("status") != "complete"
        or report.get("scenes") != scenes
        or not report.get("closed")
        or report.get("expert_initialized")
    ):
        raise RuntimeError(
            "Independent evaluator did not complete the declared student panel"
        )
    return report


def build_trainer(config: dict[str, Any]) -> SupervisedTrainer:
    """Build the pinned offline acceptance recipe for ``train-sl``."""
    deadline = time.monotonic() + 21600
    data = source(config["data"]["manifest"])
    if len(data.episodes) != 1024:
        data.close()
        raise ValueError(
            "Formal offline BC acceptance requires exactly1024 accepted episodes"
        )
    environment_path = data.path.parent / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment_hash = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()
    if environment_hash != data.manifest["config_sha256"]:
        data.close()
        raise ValueError("Corpus physical/camera configuration hash differs")
    output = Path(config["trainer"]["output_dir"]).parent
    output.mkdir(parents=True, exist_ok=True)
    statistics = frozen_statistics(data)
    torch.save(
        {"source_id": data.source_id, "statistics": statistics},
        output / "normalization.pt",
    )
    cfg = SupervisedTrainerCfg(**config["trainer"])
    if (cfg.total_updates, cfg.batch_size, cfg.validation_interval) != (
        20000,
        64,
        2000,
    ):
        data.close()
        raise ValueError(
            "Formal BC update/batch/evaluation budgets differ from the pinned recipe"
        )
    torch.manual_seed(cfg.seed)
    model = ChunkBCModel(ChunkBCModelCfg(**config["model"]))
    model.set_normalization(**statistics)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: _rate(index + 1, cfg.total_updates) / 3e-4
    )
    learner = SupervisedLearner(
        model, _DeadlineObjective(deadline), optimizer, max_grad_norm=1
    )
    development_path = output / "development.json"
    development = (
        json.loads(development_path.read_text())["records"]
        if development_path.exists()
        else []
    )
    trainer: SupervisedTrainer

    def evaluate(current: SupervisedLearner) -> dict[str, float]:
        step = current.num_updates
        checkpoint = Path(cfg.output_dir) / f"step_{step:06d}.pt"
        if checkpoint.exists():
            raise FileExistsError(
                f"Refusing to replace immutable evaluation checkpoint {checkpoint}"
            )
        trainer.save(checkpoint)
        report_path = output / "development" / f"step_{step:06d}.json"
        report = _evaluate(
            checkpoint,
            report_path,
            scenes=20,
            seed=20271025,
            config=config,
            deadline=deadline,
        )
        development.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "report": str(report_path),
                "success_rate": report["success_rate"],
                "goal_error": report["goal_error"],
            }
        )
        _write(
            development_path,
            {"protocol": "sl-bc-offline-pickplace-v1", "records": development},
        )
        return {key: report[key] for key in ("success_rate", "goal_error", "lift_rate")}

    trainer = SupervisedTrainer(
        learner,
        HDF5BCSource(data),
        cfg,
        signature={
            "recipe": "sl-bc-offline-pickplace-v1",
            "model": model.get_extra_state(),
            "camera_profile_version": "pick-place-v2",
            "action_horizon": 8,
            "environment": environment,
            "data_source": data.source_id,
            "normalization_episodes": [
                record["sha256"] for record in data.episodes[:32]
            ],
            "development_seed": 20271025,
            "test_seed": 20281025,
        },
        scheduler=scheduler,
        evaluate=evaluate,
    )
    # The acceptance driver also invokes the same initial-policy and final-test
    # callbacks; these are experiment-local orchestration, not learner methods.
    trainer.acceptance_development = development
    trainer.acceptance_deadline = deadline
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--evaluation-visible-devices", default="7,1")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--policy-device", default="cuda:1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "factory": "experiments.learning.sl.train_bc:build_trainer",
        "model": {
            "observation_dim": 27,
            "target_dim": 10,
            "history_length": 2,
            "prediction_horizon": 32,
            "hidden_dims": [512, 512],
            "cameras": list(CAMERAS),
            "target_semantics": TARGET_SEMANTICS,
        },
        "data": {"manifest": str(args.manifest.resolve())},
        "objective": {"type": "masked_normalized_target_mse"},
        "optimizer": {
            "type": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 1e-6,
            "warmup_updates": 500,
            "final_learning_rate": 3e-5,
            "max_grad_norm": 1,
        },
        "trainer": {
            "total_updates": 20000,
            "batch_size": 64,
            "device": args.train_device,
            "seed": args.seed,
            "output_dir": str(args.output.resolve() / "training"),
            "validation_interval": 2000,
            "checkpoint_interval": 0,
        },
        "evaluation": {
            "visible_devices": args.evaluation_visible_devices,
            "sim_device": args.sim_device,
            "policy_device": args.policy_device,
            "development_scenes": 20,
            "development_threshold": 0.9,
            "test_scenes": 100,
            "test_threshold": 0.8,
        },
    }
    _write(args.output / "recipe.json", config)
    trainer = None
    started = time.monotonic()
    try:
        trainer = build_trainer(config)
        if args.prepare_only:
            _write(
                args.output / "prepared.json",
                {
                    "prepared": True,
                    "learning_updates": 0,
                    "source_id": trainer.source.source_id,
                },
            )
            return
        trainer.evaluate(trainer.learner)
        result = trainer.fit()
        updated = [row for row in trainer.acceptance_development if row["step"] > 0]
        gate = len(updated) >= 2 and all(
            row["success_rate"] >= 0.9 for row in updated[-2:]
        )
        selected = min(
            updated,
            key=lambda row: (-row["success_rate"], row["goal_error"], row["step"]),
        )
        selection = {
            "development_gate_passed": gate,
            "selected": selected,
            "test_panel_sealed_at_selection": True,
            "selection_frozen": True,
        }
        _write(args.output / "selection.json", selection)
        final_test = None
        if gate:
            _write(
                args.output / "test-request.json",
                {
                    "checkpoint": selected["checkpoint"],
                    "checkpoint_sha256": hashlib.sha256(
                        Path(selected["checkpoint"]).read_bytes()
                    ).hexdigest(),
                    "seed": 20281025,
                    "scenes": 100,
                    "selection_frozen": True,
                },
            )
            final_test = _evaluate(
                Path(selected["checkpoint"]),
                args.output / "final-test.json",
                scenes=100,
                seed=20281025,
                config=config,
                deadline=trainer.acceptance_deadline,
            )
        passed = gate and final_test is not None and final_test["success_rate"] >= 0.8
        _write(
            args.output / "summary.json",
            {
                "protocol": "sl-bc-offline-pickplace-v1",
                "seed": args.seed,
                "num_updates": result["num_updates"],
                "development_gate_passed": gate,
                "selected": selected,
                "test_opened": final_test is not None,
                "final_test_success_rate": (
                    None if final_test is None else final_test["success_rate"]
                ),
                "learning_target_met": passed,
                "wall_seconds": time.monotonic() - started,
            },
        )
        if not passed:
            raise SystemExit(1)
    except Exception as error:
        _write(
            args.output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "wall_seconds": time.monotonic() - started,
                "completed_updates": (
                    0 if trainer is None else trainer.learner.num_updates
                ),
            },
        )
        raise
    finally:
        if trainer is not None:
            trainer.source.close()


if __name__ == "__main__":
    main()
