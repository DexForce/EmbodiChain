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

"""Qualify RWM fitting on controlled accelerations and a 29-joint layout.

This is a supervised interface/gradient acceptance experiment, not a physical
robot benchmark or a SuperMimic policy reproduction. All artifacts remain in
the caller's output directory. Run through ``embodichain run-example learning/rwm``.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module, metadata
import resource
import subprocess
import json
import sys
from pathlib import Path
import time

import torch

from embodichain_models.dynamics.rwm.robot_state import integrate_robot_state
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.dynamics.rwm import (
    RobotWorldModelObjective,
    build_rwm_trainer,
    RobotWorldModel,
    RobotWorldModelCfg,
    frozen_world_model,
)


def _corpus(seed: int, count: int, horizon: int, *, joints: int = 1) -> SupervisedBatch:
    rng = torch.Generator().manual_seed(seed)
    initial = torch.zeros(count, 18 + 2 * joints)
    initial[:, 2] = 1
    initial[:, 3:12] = torch.eye(3).flatten()
    initial[:, 18:] = torch.randn(count, 2 * joints, generator=rng) * 0.3
    actions = torch.randn(count, horizon, joints, generator=rng)
    predictions = []
    state = initial
    for index in range(horizon):
        increment = torch.zeros(count, 12 + 2 * joints)
        increment[:, 12 : 12 + joints] = (
            0.1 * state[:, 18 + joints :] + 0.005 * actions[:, index]
        )
        increment[:, 12 + joints :] = 0.1 * actions[:, index]
        state = integrate_robot_state(state, increment)
        predictions.append(state)
    return SupervisedBatch(
        {"initial_state": initial, "actions": actions},
        torch.stack(predictions, 1),
        {"valid": torch.ones(count, horizon, dtype=torch.bool)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, count, seed in (
        ("train", 512, args.seed + 100),
        ("validation", 128, args.seed + 10000),
    ):
        corpus = _corpus(seed, count, 4)
        torch.save(
            {
                "inputs": corpus.inputs,
                "targets": corpus.targets,
                "masks": corpus.masks,
                "metadata": {
                    "split": split,
                    "action_semantics": "generalized_effort",
                    "step_dt": 0.1,
                },
            },
            args.output / f"{split}.pt",
        )
    config = {
        "factory": "embodichain_models.dynamics.rwm.sl_recipe:build_rwm_trainer",
        "model": {
            "num_joints": 1,
            "action_dim": 1,
            "hidden_dims": [48, 48],
            "step_dt": 0.1,
        },
        "data": {
            "path": str(args.output / "train.pt"),
            "validation_path": str(args.output / "validation.pt"),
        },
        "objective": {"error_floor": 0.01},
        "optimizer": {"type": "adam", "learning_rate": 0.002, "max_grad_norm": 1.0},
        "trainer": {
            "total_updates": args.updates,
            "batch_size": 64,
            "seed": args.seed,
            "device": args.device,
            "output_dir": str(args.output / "training"),
            "validation_interval": 50,
            "checkpoint_interval": 150,
        },
    }
    (args.output / "recipe.json").write_text(json.dumps(config, indent=2))
    trainer = build_rwm_trainer(config)
    initial = dict(trainer.evaluate(trainer.learner))
    started = time.monotonic()
    result = trainer.fit()
    elapsed = time.monotonic() - started
    final = dict(trainer.evaluate(trainer.learner))

    # A larger robot state layout exercises joint ordering, contexts and graph
    # connectivity independently of the one-joint learning threshold.
    robot = RobotWorldModel(
        RobotWorldModelCfg(
            num_joints=29, action_dim=29, context_dim=3, hidden_dims=(32,), step_dt=0.1
        )
    )
    batch = _corpus(1234, 8, 3, joints=29)
    batch.inputs["contexts"] = torch.ones(8, 3, 3)
    objective = RobotWorldModelObjective()
    learner = SupervisedLearner(
        robot, objective, torch.optim.Adam(robot.parameters(), lr=1e-3), max_grad_norm=1
    )
    for _ in range(5):
        layout_metrics = learner.update(batch)
    controls = batch.inputs["actions"].clone().requires_grad_()
    current = batch.inputs["initial_state"]
    with frozen_world_model(robot):
        for index in range(3):
            current = robot(
                current, controls[:, index], batch.inputs["contexts"][:, index]
            )
        gradient = torch.autograd.grad(current[:, 18:].square().sum(), controls)[0]
    passed = (
        final["loss"] < initial["loss"] * 0.15
        and bool(torch.isfinite(gradient).all())
        and float(gradient[:, 0].norm()) > 0
    )
    repository = Path(__file__).resolve().parents[4]
    sources = {"examples/learning/rwm/_support/_controlled_fit.py": Path(__file__)}
    for name in (
        "embodichain_models.dynamics.rwm.model",
        "embodichain_models.dynamics.rwm.supervised",
        "embodichain_models.dynamics.rwm.sl_recipe",
        "embodichain_models.dynamics.rwm.robot_state",
        "embodichain.learning.sl.learner",
        "embodichain.learning.sl.trainer",
        "embodichain.learning.sl.sources",
    ):
        prefix = (
            "extensions/embodichain_models/"
            if name.startswith("embodichain_models.")
            else ""
        )
        sources[f"{prefix}{name.replace('.', '/')}.py"] = Path(
            import_module(name).__file__
        )
    device = torch.device(args.device)
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "allocator_peak_bytes": torch.cuda.max_memory_allocated(device),
            "allocator_reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
        }
    summary = {
        "protocol": "rwm-controlled-fit-v1",
        "seed": args.seed,
        "device": args.device,
        "updates": result["num_updates"],
        "initial_validation": initial,
        "final_validation": final,
        "loss_ratio": final["loss"] / initial["loss"],
        "fit_seconds": elapsed,
        "fit_timing_scope": "Synchronous training loop including periodic validation and checkpoint I/O; excludes imports and final validation",
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "torch_cuda_memory": cuda_memory,
        "provenance": {
            "git_commit": (
                subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
                ).strip()
                if (repository / ".git").exists()
                else None
            ),
            "distribution_version": metadata.version("embodichain"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "source_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in sources.items()
            },
            "data_sha256": {
                name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                for name in ("train.pt", "validation.pt")
            },
        },
        "robot_layout": {
            "num_joints": 29,
            "context_dim": 3,
            "updates": 5,
            "final_loss": layout_metrics["loss"],
            "initial_action_gradient_norm": float(gradient[:, 0].norm()),
        },
        "passed": passed,
        "claim": "Controlled supervised fit and interface acceptance; no robot-policy performance claim",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    print(json.dumps(summary, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
