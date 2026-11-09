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

"""Learn local Franka reaching through fixed differentiable forward kinematics.

Run: embodichain run-example learning/differentiable_fk/learn_pose_reaching
Only target TCP poses supervise the network. The example uses the common SL
trainer and does not call an IK solver or start a simulation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from embodichain.compute.kinematics import SerialChainKinematics, pose_error
from embodichain.data import get_data_path
from embodichain.learning.layers import DifferentiableFKLayer
from embodichain.learning.models import MLP
from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
    TensorBatchSource,
)
from embodichain.utils import configclass

__all__ = ["PoseReachingCfg", "PosePredictor", "load_predictor", "run"]


@configclass
class PoseReachingCfg:
    """Small, fixed local-reaching experiment; smoke runs check execution only."""

    device: str = "cpu"
    seed: int = 17
    train_count: int = 1024
    validation_count: int = 128
    test_count: int = 128
    updates: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    state_radius: float = 0.10
    target_radius: float = 0.10
    correction_bound: float = 0.15
    position_scale: float = 0.05
    rotation_scale: float = 0.20
    posture_weight: float = 1e-3
    output: str = "outputs/learn_fk_reaching"
    smoke: bool = False

    def __post_init__(self) -> None:
        for name in (
            "train_count",
            "validation_count",
            "test_count",
            "updates",
            "batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "learning_rate",
            "state_radius",
            "target_radius",
            "correction_bound",
            "position_scale",
            "rotation_scale",
            "posture_weight",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.target_radius > self.correction_bound:
            raise ValueError("Target perturbations must fit within the output bound")


class PosePredictor(nn.Module):
    """Example-local joint predictor with a fixed FK layer and bounded corrections.

    Args:
        cfg: Feature scales and joint correction bound.
    """

    def __init__(self, cfg: PoseReachingCfg) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg)
        urdf = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
        self.fk = DifferentiableFKLayer(
            SerialChainKinematics.from_urdf(urdf.read_text(), "fr3_hand_tcp", "base")
        )
        self.register_buffer(
            "ready", torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        )
        self.network = MLP(13, 7, [64, 64], activation="silu")
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, q_current: torch.Tensor, target_pose: torch.Tensor
    ) -> torch.Tensor:
        """Predict joints using current joints and a root-frame TCP command.

        Args:
            q_current: Admitted current joints, shape ``(..., 7)``.
            target_pose: Desired TCP transforms, shape ``(..., 4, 4)``.

        Returns:
            Joints within ``correction_bound`` of the current posture. Dataset
            sampling reserves that margin inside the physical limits.
        """
        position, rotation = pose_error(self.fk(q_current), target_pose)
        features = torch.cat(
            (
                (q_current - self.ready) / self.cfg.state_radius,
                position / self.cfg.position_scale,
                rotation / self.cfg.rotation_scale,
            ),
            -1,
        )
        return q_current + self.cfg.correction_bound * self.network(features).tanh()


class _PoseObjective:
    def compute_loss(self, model: PosePredictor, batch: SupervisedBatch) -> LossResult:
        q_pred = model(**batch.inputs)
        position, rotation = pose_error(model.fk(q_pred), batch.targets)
        pose_loss = (
            (position / model.cfg.position_scale).square().sum(-1)
            + (rotation / model.cfg.rotation_scale).square().sum(-1)
        ).mean()
        posture = (q_pred - batch.inputs["q_current"]).square().sum(-1).mean()
        return LossResult(
            pose_loss + model.cfg.posture_weight * posture,
            {"pose_loss": pose_loss.detach(), "posture_loss": posture.detach()},
            weight=q_pred.shape[0],
        )


def _sample(model: PosePredictor, count: int, seed: int) -> SupervisedBatch:
    """Generate reachable TCP labels; generating joint solutions are discarded."""
    cfg = model.cfg
    low = torch.maximum(
        model.ready - cfg.state_radius, model.fk.lower + cfg.correction_bound
    )
    high = torch.minimum(
        model.ready + cfg.state_radius, model.fk.upper - cfg.correction_bound
    )
    if (low > high).any():
        raise ValueError("Reference posture has no admitted sampling interval")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_current = low + torch.rand(count, 7, generator=generator) * (high - low)
    q_goal = q_current + cfg.target_radius * (
        2 * torch.rand(count, 7, generator=generator) - 1
    )
    with torch.no_grad():
        target = model.fk(q_goal)
    return SupervisedBatch(
        {"q_current": q_current, "target_pose": target},
        target,
        metadata={
            "root": "base",
            "tcp": "fr3_hand_tcp",
            "position_unit": "m",
            "joint_unit": "rad",
        },
    )


def _errors(
    model: PosePredictor, joints: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    position, rotation = pose_error(model.fk(joints), targets)
    if (
        not torch.isfinite(joints).all()
        or not torch.isfinite(position).all()
        or not torch.isfinite(rotation).all()
    ):
        raise FloatingPointError("Non-finite reaching prediction")
    if ((joints < model.fk.lower - 1e-6) | (joints > model.fk.upper + 1e-6)).any():
        raise ValueError("Predicted joints exceed physical limits")
    p_mm = position.norm(dim=-1) * 1000
    r_deg = torch.rad2deg(rotation.norm(dim=-1))
    return {
        "position_mm": p_mm,
        "rotation_deg": r_deg,
        "success": (p_mm <= 2) & (r_deg <= 2),
    }


def _summary(errors: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "position_mean_mm": float(errors["position_mm"].mean()),
        "position_p95_mm": float(torch.quantile(errors["position_mm"], 0.95)),
        "rotation_mean_deg": float(errors["rotation_deg"].mean()),
        "rotation_p95_deg": float(torch.quantile(errors["rotation_deg"], 0.95)),
        "success_rate": float(errors["success"].float().mean()),
    }


def _cpu_state(model: nn.Module) -> dict[str, Any]:
    return {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
        for name, value in model.state_dict().items()
    }


def load_predictor(run_dir: Path, device: str = "cpu") -> PosePredictor:
    """Restore a validation-selected frozen predictor on the requested device.

    Args:
        run_dir: Directory containing ``predictor.pt``.
        device: Inference device, independent of the training device.

    Returns:
        An evaluation-mode model with fixed weights and checked FK identity.
    """
    saved = torch.load(run_dir / "predictor.pt", map_location="cpu", weights_only=True)
    model = PosePredictor(PoseReachingCfg(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval().requires_grad_(False)


def _plot(output: Path, history: list[dict[str, float]], best_step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, key, label in zip(
        axes,
        ("position_mean_mm", "rotation_mean_deg"),
        ("Position error / mm", "Rotation error / degrees"),
    ):
        axis.plot(
            [row["step"] for row in history],
            [row[key] for row in history],
            label="Independent validation",
        )
        axis.axvline(
            best_step, color="tab:green", linestyle="--", label="Selected checkpoint"
        )
        axis.set(xlabel="Optimizer update", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "learning.png", dpi=160)
    plt.close(figure)


def run(cfg: PoseReachingCfg) -> dict[str, Any]:
    """Train with pose supervision and evaluate every independent test sample.

    Args:
        cfg: Experiment sizes, numerical settings and output location.

    Returns:
        Metrics, selected update, learning checks and configuration.
    """
    cfg.validate()
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    model = PosePredictor(cfg)
    datasets = {
        name: _sample(model, count, seed)
        for name, count, seed in (
            ("train", cfg.train_count, 23),
            ("validation", cfg.validation_count, 24),
            ("test", cfg.test_count, 25),
        )
    }
    source = TensorBatchSource(datasets["train"], source_id="franka-fk-reaching-v1")
    model.to(cfg.device)
    fixed_fk = _cpu_state(model.fk)
    objective = _PoseObjective()
    learner = SupervisedLearner(
        model, objective, torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    )
    validation = datasets["validation"].to(cfg.device)
    history: list[dict[str, float]] = []
    best_loss, best_step, best_state = float("inf"), 0, _cpu_state(model)

    def evaluate(current: SupervisedLearner) -> dict[str, float]:
        nonlocal best_loss, best_step, best_state
        with torch.no_grad():
            joints = model(**validation.inputs)
            stats = _summary(_errors(model, joints, validation.targets))
            value = float(objective.compute_loss(model, validation).loss)
        history.append({"step": current.num_updates, "loss": value, **stats})
        if value < best_loss:
            best_loss, best_step, best_state = (
                value,
                current.num_updates,
                _cpu_state(model),
            )
        print(
            f"step={current.num_updates:4d} validation={value:.6g} position={stats['position_mean_mm']:.3f} mm rotation={stats['rotation_mean_deg']:.3f} deg",
            flush=True,
        )
        return {"loss": value, **stats}

    evaluate(learner)
    interval = min(50, cfg.updates)
    trainer = SupervisedTrainer(
        learner,
        source,
        SupervisedTrainerCfg(
            total_updates=cfg.updates,
            batch_size=cfg.batch_size,
            device=cfg.device,
            seed=cfg.seed + 12,
            output_dir=cfg.output,
            checkpoint_interval=0,
            validation_interval=interval,
        ),
        signature={
            "recipe": "franka-fk-reaching-v1",
            "config": cfg.to_dict(),
            "model": "13-64-64-7-silu",
            "selection": "validation-loss-only",
        },
        evaluate=evaluate,
    )
    trainer.fit()
    if cfg.updates % interval:
        evaluate(learner)
    trained_fk = _cpu_state(model.fk)
    for name, value in fixed_fk.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, trained_fk[name], atol=0, rtol=0)
        else:
            assert value == trained_fk[name]
    model.load_state_dict(best_state)
    test = datasets["test"].to(cfg.device)
    with torch.no_grad():
        before_q = test.inputs["q_current"]
        after_q = model(**test.inputs)
        before = _errors(model, before_q, test.targets)
        after = _errors(model, after_q, test.targets)
    # Check only the pose term: the posture regularizer cannot hide a detached FK.
    probe = SupervisedBatch(
        {name: value[:4] for name, value in validation.inputs.items()},
        validation.targets[:4],
    )
    predicted = model(**probe.inputs)
    p, r = pose_error(model.fk(predicted), probe.targets)
    pose_loss = (p.square().sum(-1) + r.square().sum(-1)).mean()
    gradients = torch.autograd.grad(pose_loss, tuple(model.network.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0

    output = Path(cfg.output)
    torch.save(
        {"config": cfg.to_dict(), "state_dict": best_state}, output / "predictor.pt"
    )
    reloaded = load_predictor(output)
    with torch.no_grad():
        restored_q = reloaded(**datasets["test"].inputs)
    torch.testing.assert_close(restored_q, after_q.cpu(), atol=2e-5, rtol=1e-4)
    np.savez_compressed(
        output / "datasets.npz",
        **{
            f"{split}_{name}": tensor.numpy()
            for split, batch in datasets.items()
            for name, tensor in batch.inputs.items()
        },
    )
    np.savez_compressed(
        output / "evaluation.npz",
        current_q=before_q.cpu().numpy(),
        target_pose=test.targets.cpu().numpy(),
        after_q=after_q.cpu().numpy(),
        **{
            f"{method}_{name}": value.cpu().numpy()
            for method, errors in (("before", before), ("after", after))
            for name, value in errors.items()
        },
    )
    before_stats, after_stats = _summary(before), _summary(after)
    improvement = {
        name: 1 - after_stats[key] / before_stats[key]
        for name, key in (
            ("position", "position_mean_mm"),
            ("rotation", "rotation_mean_deg"),
        )
    }
    passed = all(value >= 0.5 for value in improvement.values())
    metrics = {
        "config": cfg.to_dict(),
        "best_step": best_step,
        "history": history,
        "test_count": cfg.test_count,
        "before": before_stats,
        "after": after_stats,
        "improvement": improvement,
        "learning_acceptance_passed": passed,
        "checks": {
            "pose_gradients": True,
            "fixed_fk": True,
            "physical_limits": True,
            "cpu_reload": True,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    _plot(output, history, best_step)
    rows = [
        "# Learning reaching through differentiable FK",
        "",
        f"Independent test samples: {cfg.test_count}; validation-selected update: {best_step}.",
        "",
        "| Model | Position mean / mm | Position P95 / mm | Rotation mean / deg | Rotation P95 / deg | Within 2 mm and 2 deg |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in (
        ("Before training", before_stats),
        ("After training", after_stats),
    ):
        rows.append(
            f"| {name} | {stats['position_mean_mm']:.4f} | {stats['position_p95_mm']:.4f} | {stats['rotation_mean_deg']:.4f} | {stats['rotation_p95_deg']:.4f} | {stats['success_rate']:.1%} |"
        )
    rows.extend(
        [
            "",
            "![Validation curves](learning.png)",
            "",
            "All test samples, including failures, contribute to these metrics. Targets are local and reachable by construction. Smoke runs check execution only. The viewer shows kinematic configurations without physical execution.",
        ]
    )
    (output / "report.md").write_text("\n".join(rows) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "before": before_stats,
                "after": after_stats,
                "improvement": improvement,
            },
            indent=2,
        ),
        flush=True,
    )
    if not cfg.smoke and not passed:
        raise RuntimeError(
            "Held-out position and rotation mean errors must each improve by at least 50%; inspect report.md"
        )
    return metrics


def main() -> None:
    """Train and evaluate a standalone FK-supervised joint predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--output", default="outputs/learn_fk_reaching")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    options = dict(
        device=args.device,
        seed=args.seed,
        updates=args.updates,
        output=args.output,
        smoke=args.smoke,
    )
    if args.smoke:
        options.update(
            updates=20,
            train_count=64,
            validation_count=16,
            test_count=16,
            batch_size=32,
        )
    run(PoseReachingCfg(**options))


if __name__ == "__main__":
    main()
