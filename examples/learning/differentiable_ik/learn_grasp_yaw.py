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

"""Train a conditional grasp-yaw predictor through differentiable IK.

Run: python -m examples.learning.differentiable_ik.learn_grasp_yaw
Models remain example-local. Validation selects checkpoints; held-out targets
never select parameters. Numerical poses are not collision-checked trajectories.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from embodichain.compute.kinematics import DifferentiableIKCfg, SerialChainKinematics
from embodichain.data import get_data_path
from embodichain.learning.layers import DifferentiableIKLayer, KinematicsLayer
from embodichain.learning.models import MLP
from embodichain.utils import configclass
from ._benchmark import _measure, _sync, _table

METHODS = ("heading", "constant", "learned", "supervised", "online", "grid")
SPLITS = ("iid", "spatial", "state_shift")


@configclass
class GraspLearningCfg:
    """Fixed, small training and evaluation budgets for the example."""

    device: str = "cuda"
    seeds: tuple[int, ...] = (17, 29, 43)
    train_count: int = 1024
    validation_count: int = 128
    test_count: int = 128
    updates: int = 200
    batch_size: int = 64
    learning_rate: float = 0.002
    ik_steps: int = 24
    check_steps: int = 64
    reference_updates: int = 50
    grid_points: int = 129
    chunk_size: int = 256
    grid_chunk_size: int = 16384
    output: str = "outputs/learn_grasp_yaw"

    def __post_init__(self) -> None:
        for name in (
            "train_count",
            "validation_count",
            "test_count",
            "updates",
            "batch_size",
            "ik_steps",
            "check_steps",
            "reference_updates",
            "chunk_size",
            "grid_chunk_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.grid_points < 3 or self.check_steps < self.ik_steps:
            raise ValueError("Need grid_points >= 3 and check_steps >= ik_steps")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def load_kinematics(device: str = "cpu") -> KinematicsLayer:
    """Build the official Franka arm chain without importing simulation."""
    path = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
    return KinematicsLayer(
        SerialChainKinematics.from_urdf(
            path.read_text(), "fr3_hand_tcp", "base", device=device
        )
    )


def target_pose(angle: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Construct Rz(yaw) @ Rx(pi): position fixed and tool Z always downward."""
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    rotation = torch.stack((c, s, zero, s, -c, zero, zero, zero, -one), -1).reshape(
        -1, 3, 3
    )
    return torch.cat(
        (
            torch.cat((rotation, position[..., None]), -1),
            angle.new_tensor([0, 0, 0, 1]).expand(len(angle), 1, 4),
        ),
        1,
    )


def ready_joints(k: KinematicsLayer) -> torch.Tensor:
    """Return the official reference posture in the model's dtype/device."""
    return k.tcp.new_tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])


def current_heading(k: KinematicsLayer, q: torch.Tensor) -> torch.Tensor:
    """Project current TCP X into root XY; data stays near downward grasps."""
    pose = k(q)
    return torch.atan2(pose[..., 1, 0], pose[..., 0, 0])


class GraspYawPredictor(nn.Module):
    """Predict a bounded yaw correction from current joints and grasp position.

    The zero-initialized output starts at current heading. Normalization uses
    declared task scales, not statistics from validation or test targets.
    """

    def __init__(self, k: KinematicsLayer) -> None:
        super().__init__()
        base = ready_joints(k)
        with torch.no_grad():
            center = k(base)[:3, 3]
        self.register_buffer("q_center", base.detach().clone())
        self.register_buffer("p_center", center.detach().clone())
        self.register_buffer("q_scale", base.new_tensor([0.12] * 6 + [0.7]))
        self.network = MLP(10, 1, [64, 64], "silu").to(
            device=base.device, dtype=base.dtype
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, q: torch.Tensor, p: torch.Tensor, heading: torch.Tensor
    ) -> torch.Tensor:
        """Return a grasp yaw; the task permits any yaw about downward tool Z."""
        features = torch.cat(
            ((q - self.q_center) / self.q_scale, (p - self.p_center) / 0.05), -1
        )
        return heading + (torch.pi / 2) * torch.tanh(self.network(features).squeeze(-1))


class ConstantYaw(nn.Module):
    """Fit one absolute grasp angle shared by every task, without conditioning."""

    def __init__(self, k: KinematicsLayer) -> None:
        super().__init__()
        self.angle = nn.Parameter(k.tcp.new_zeros(()))

    def forward(
        self, q: torch.Tensor, p: torch.Tensor, heading: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the learned scalar angle over the task batch."""
        return self.angle.expand(len(q))


@torch.no_grad()
def sample_tasks(
    k: KinematicsLayer, count: int, seed: int, split: str
) -> dict[str, torch.Tensor]:
    """Sample independent local problems with a disjoint Cartesian-Y holdout."""
    g = torch.Generator().manual_seed(seed)
    base = ready_joints(k)
    center = k(base)[:3, 3].cpu()
    q = base.cpu() + 0.24 * torch.rand(count, 7, generator=g) - 0.12
    q[:, 6] += 1.4 * torch.rand(count, generator=g) - 0.7
    if split == "state_shift":
        q[:, 0] += 0.20
        q[:, 2] += 0.15
    q = q.to(k.tcp).clamp(k.lower, k.upper)
    p = center + 0.10 * torch.rand(count, 3, generator=g) - 0.05
    # Train/validation/IID/state-shift: Y in [-5,+1] cm relative to center.
    p[:, 1] = center[1] - 0.05 + 0.06 * torch.rand(count, generator=g)
    if split == "spatial":
        p[:, 1] = center[1] + 0.025 + 0.03 * torch.rand(count, generator=g)
    p = p.to(k.tcp)
    return {"q": q, "p": p, "heading": current_heading(k, q)}


def _subset(data: dict[str, torch.Tensor], index: Any) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in data.items()}


def yaw_objective(
    ik: DifferentiableIKLayer, data: dict[str, torch.Tensor], angle: torch.Tensor
) -> tuple[torch.Tensor, Any]:
    """Evaluate joint displacement and feasibility penalties, with full gradients."""
    result = ik.solve(target_pose(angle, data["p"]), data["q"])
    cost = (result.qpos - data["q"]).square().sum(-1)
    loss = (
        cost
        + 1e4 * result.position_error.square().sum(-1)
        + 100 * result.rotation_error.square().sum(-1)
    )
    return loss, result


@torch.no_grad()
def evaluate_angles(
    ik: DifferentiableIKLayer,
    data: dict[str, torch.Tensor],
    angle: torch.Tensor,
    chunk: int = 256,
) -> dict[str, torch.Tensor]:
    """Preserve all failures and return numerical task metrics in bounded chunks."""
    rows = []
    for start in range(0, len(angle), chunk):
        d = _subset(data, slice(start, start + chunk))
        a = angle[start : start + chunk]
        loss, r = yaw_objective(ik, d, a)
        rows.append(
            {
                "angle": a,
                "q": r.qpos,
                "loss": loss,
                "cost": (r.qpos - d["q"]).square().sum(-1),
                "valid": r.success,
                "position_mm": r.position_error.norm(dim=-1) * 1000,
                "rotation_rad": r.rotation_error.norm(dim=-1),
            }
        )
    return {name: torch.cat([row[name] for row in rows]) for name in rows[0]}


def optimize_task_angles(
    ik: DifferentiableIKLayer,
    data: dict[str, torch.Tensor],
    updates: int = 50,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize each task separately; retain the best feasible inner-budget angle.

    Start at current heading and also retain fixed yaw as an incumbent. This is
    the existing demo's Adam/feasible-incumbent method with a stronger start.
    If no feasible candidate exists, return heading with a false feasibility flag.
    """
    selected, feasible = [], []
    with torch.enable_grad():
        for start in range(0, len(data["q"]), chunk):
            d = _subset(data, slice(start, start + chunk))
            angle = nn.Parameter(d["heading"].detach().clone())
            best_angle = angle.detach().clone()
            best_cost = torch.full_like(angle, float("inf"))
            with torch.no_grad():
                zero = torch.zeros_like(angle)
                _, r = yaw_objective(ik, d, zero)
                cost = (r.qpos - d["q"]).square().sum(-1)
                best_cost = torch.where(r.success, cost, best_cost)
                best_angle = torch.where(r.success, zero, best_angle)
            optimizer = torch.optim.Adam([angle], lr=0.08)
            finite = torch.ones((), dtype=torch.bool, device=angle.device)
            for step in range(updates + 1):
                loss, r = yaw_objective(ik, d, angle)
                with torch.no_grad():
                    cost = (r.qpos - d["q"]).square().sum(-1)
                    better = r.success & (cost < best_cost)
                    best_cost = torch.where(better, cost, best_cost)
                    best_angle = torch.where(better, angle.detach(), best_angle)
                if step < updates:
                    optimizer.zero_grad(set_to_none=True)
                    loss.mean().backward()
                    finite &= torch.isfinite(angle.grad).all()
                    optimizer.step()
            if not bool(finite):
                raise RuntimeError("Nonfinite numerical-reference yaw gradient")
            selected.append(best_angle)
            feasible.append(torch.isfinite(best_cost))
    return torch.cat(selected), torch.cat(feasible)


@torch.no_grad()
def grid_reference(
    ik: DifferentiableIKLayer, data: dict[str, torch.Tensor], points: int, chunk: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return the best feasible sampled yaw and its curve, not a certified optimum."""
    count = len(data["q"])
    angles = torch.linspace(
        -torch.pi, torch.pi, points, device=data["q"].device, dtype=data["q"].dtype
    )
    expanded = {
        key: value[:, None]
        .expand(count, points, *value.shape[1:])
        .reshape(count * points, *value.shape[1:])
        for key, value in data.items()
    }
    result = evaluate_angles(ik, expanded, angles.expand(count, -1).reshape(-1), chunk)
    costs, valid = result["cost"].reshape(count, points), result["valid"].reshape(
        count, points
    )
    index = costs.masked_fill(~valid, float("inf")).argmin(-1)
    flat_index = torch.arange(count, device=index.device) * points + index
    best = {key: value[flat_index] for key, value in result.items()}
    best["valid"] = valid.any(-1)
    return best, angles, costs.masked_fill(~valid, float("nan"))


def fit_model(
    k: KinematicsLayer,
    cfg: GraspLearningCfg,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    seed: int,
    kind: str,
    labels: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train with fixed minibatches; select checkpoints only on validation IK loss."""
    if kind not in ("learned", "constant", "supervised"):
        raise ValueError(f"Unknown learning method: {kind}")
    if kind == "supervised" and labels is None:
        raise ValueError("Supervised training requires training-only yaw labels")
    torch.manual_seed(seed)
    model = ConstantYaw(k) if kind == "constant" else GraspYawPredictor(k)
    ik = DifferentiableIKLayer(k, DifferentiableIKCfg(max_iterations=cfg.ik_steps))
    lr = 0.04 if kind == "constant" else cfg.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1000)
    indices = torch.randint(
        cfg.train_count, (cfg.updates, cfg.batch_size), generator=g
    ).to(k.tcp.device)
    best_loss = float("inf")
    best_state, best_step = deepcopy(model.state_dict()), 0
    history = []
    finite = torch.ones((), dtype=torch.bool, device=k.tcp.device)
    _sync(k.tcp.device)
    start = time.perf_counter()
    for step in range(cfg.updates + 1):
        if step % 25 == 0 or step == cfg.updates:
            model.eval()
            with torch.no_grad():
                angle = model(validation["q"], validation["p"], validation["heading"])
                rows = evaluate_angles(ik, validation, angle, cfg.chunk_size)
                val_loss = float(rows["loss"].mean())
                history.append(
                    {
                        "step": step,
                        "validation_loss": val_loss,
                        "success_rate": float(rows["valid"].float().mean()),
                    }
                )
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                best_state = deepcopy(model.state_dict())
        if step == cfg.updates:
            break
        model.train()
        idx = indices[step]
        d = _subset(train, idx)
        angle = model(d["q"], d["p"], d["heading"])
        if kind == "supervised":
            label, valid = labels[0][idx], labels[1][idx]
            # Periodic loss avoids the discontinuity of scalar angle labels.
            loss = (
                (1 - torch.cos(angle - label)) * valid
            ).sum() / valid.sum().clamp_min(1)
        else:
            loss = yaw_objective(ik, d, angle)[0].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite &= torch.isfinite(loss)
        for parameter in model.parameters():
            if parameter.grad is not None:
                finite &= torch.isfinite(parameter.grad).all()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, foreach=False)
        optimizer.step()
    _sync(k.tcp.device)
    seconds = time.perf_counter() - start
    if not bool(finite):
        raise RuntimeError(f"Nonfinite {kind} training gradient")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, {
        "seed": seed,
        "method": kind,
        "seconds": seconds,
        "selected_step": best_step,
        "finite_gradients": True,
        "history": history,
    }


def summarize(
    result: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Report failures independently and paired-valid relative costs explicitly."""
    valid = result["valid"]
    paired = valid & baseline["valid"]
    ref_paired = valid & reference["valid"]
    return {
        "cases": len(valid),
        "success_rate": float(valid.float().mean()),
        "mean_cost_valid": float(result["cost"][valid].mean()) if valid.any() else None,
        "mean_loss_all": float(result["loss"].mean()),
        "p95_position_mm_all": float(torch.quantile(result["position_mm"], 0.95)),
        "p95_rotation_rad_all": float(torch.quantile(result["rotation_rad"], 0.95)),
        "paired_with_heading": int(paired.sum()),
        "mean_relative_improvement": (
            float((1 - result["cost"][paired] / baseline["cost"][paired]).mean())
            if paired.any()
            else None
        ),
        "improved_cases": int(
            (result["cost"][paired] < baseline["cost"][paired]).sum()
        ),
        "paired_with_grid": int(ref_paired.sum()),
        "mean_relative_grid_gap": (
            float(
                (result["cost"][ref_paired] / reference["cost"][ref_paired] - 1).mean()
            )
            if ref_paired.any()
            else None
        ),
    }


def validate_gradients(k: KinematicsLayer, iterations: int = 24) -> dict[str, Any]:
    """Check complete policy→IK task gradients and reject hidden input truncation."""
    # These are executable demo checks, not a new tutorial-importing pytest suite.
    torch.manual_seed(101)
    cpu = deepcopy(k).cpu().double()
    data = sample_tasks(cpu, 2, 319, "iid")
    model = GraspYawPredictor(cpu).double()
    with torch.no_grad():
        model.network[-1].weight.fill_(0.015)
    ik = DifferentiableIKLayer(cpu, DifferentiableIKCfg(max_iterations=iterations))
    q, p = data["q"].clone().requires_grad_(), data["p"].clone().requires_grad_()

    def loss_fn(q_value: torch.Tensor, p_value: torch.Tensor) -> torch.Tensor:
        heading = current_heading(cpu, q_value)
        angle = model(q_value, p_value, heading)
        return yaw_objective(ik, {"q": q_value, "p": p_value}, angle)[0]

    checked = torch.autograd.gradcheck(
        loss_fn, (q, p), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True
    )
    loss = loss_fn(q, p).sum()
    params = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, (q, p, *params))
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    assert finite and all(g.norm() > 0 for g in gradients)
    # Verify a network parameter derivative independently of input gradcheck.
    param = model.network[-1].bias
    analytic = gradients[-1].clone()
    with torch.no_grad():
        original = param.clone()
        param.copy_(original + 1e-6)
    plus = loss_fn(q, p).sum().detach()
    with torch.no_grad():
        param.copy_(original - 1e-6)
    minus = loss_fn(q, p).sum().detach()
    with torch.no_grad():
        param.copy_(original)
    torch.testing.assert_close(
        analytic.squeeze(), (plus - minus) / 2e-6, atol=1e-4, rtol=1e-3
    )
    return {
        "ik_steps": iterations,
        "input_gradcheck": checked,
        "parameter_finite_difference": True,
        "all_input_and_parameter_gradients_finite_nonzero": finite,
    }


def load_predictors(
    run_dir: Path, device: str = "cpu"
) -> tuple[KinematicsLayer, dict[str, nn.Module], GraspLearningCfg]:
    """Restore the first seeded run's validation-selected, frozen predictors."""
    meta = json.loads((run_dir / "metrics.json").read_text())
    cfg = GraspLearningCfg(**meta["config"])
    k = load_kinematics(device)
    if meta["model_fingerprint"] != k.model.fingerprint:
        raise ValueError("Checkpoint robot geometry differs from the configured Franka")
    saved = torch.load(run_dir / "models.pt", map_location=device, weights_only=True)
    models = {}
    for name, state in saved["states"].items():
        model = ConstantYaw(k) if name == "constant" else GraspYawPredictor(k)
        model.load_state_dict(state)
        models[name] = model.eval().requires_grad_(False)
    return k, models, cfg


def validate_artifacts(run_dir: Path) -> dict[str, Any]:
    """Independently verify saved predictions, held-outs and frozen model inputs."""
    from embodichain.compute.kinematics import pose_error

    k, models, cfg = load_predictors(run_dir, "cpu")
    meta = json.loads((run_dir / "metrics.json").read_text())
    selected_seed = meta["selected_viewer_seed"]
    check = DifferentiableIKLayer(
        k, DifferentiableIKCfg(max_iterations=cfg.check_steps)
    )
    with np.load(run_dir / "datasets.npz") as source:
        datasets = {name: source[name] for name in source.files}
    with np.load(run_dir / "paired_results.npz") as source:
        results = {name: source[name] for name in source.files}
    assert datasets["spatial_p"][:, 1].min() > datasets["train_p"][:, 1].max()
    all_inputs = np.concatenate(
        [
            np.concatenate((datasets[f"{split}_q"], datasets[f"{split}_p"]), -1)
            for split in ("train", "validation", *SPLITS)
        ]
    )
    assert len(np.unique(all_inputs, axis=0)) == len(all_inputs)
    for split in SPLITS:
        data = {
            name: torch.from_numpy(datasets[f"{split}_{name}"])
            for name in ("q", "p", "heading")
        }
        original = {name: value.clone() for name, value in data.items()}
        with torch.no_grad():
            for method, model in models.items():
                assert not any(
                    parameter.requires_grad for parameter in model.parameters()
                )
                angle = model(data["q"], data["p"], data["heading"])
                stored = torch.from_numpy(
                    results[f"{split}_{method}_{selected_seed}_angle"]
                )
                torch.testing.assert_close(angle, stored, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(
                    angle, model(data["q"], data["p"], data["heading"]), atol=0, rtol=0
                )
            for method in METHODS:
                suffix = f"{method}_{selected_seed}" if method in models else method
                angle = torch.from_numpy(results[f"{split}_{suffix}_angle"])
                joints = torch.from_numpy(results[f"{split}_{suffix}_q"])
                target = target_pose(angle, data["p"])
                position, rotation = pose_error(k(joints), target)
                valid = (
                    (position.norm(dim=-1) < check.cfg.pos_eps)
                    & (rotation.norm(dim=-1) < check.cfg.rot_eps)
                    & (joints >= k.lower).all(-1)
                    & (joints <= k.upper).all(-1)
                )
                torch.testing.assert_close(
                    valid, torch.from_numpy(results[f"{split}_{suffix}_valid"])
                )
                torch.testing.assert_close(target[:, :3, 3], data["p"], atol=0, rtol=0)
                torch.testing.assert_close(
                    target[:, :3, 2],
                    target.new_tensor([0, 0, -1]).expand(len(joints), -1),
                    atol=0,
                    rtol=0,
                )
        for name in data:
            torch.testing.assert_close(data[name], original[name], atol=0, rtol=0)
    q = data["q"][:2].clone().requires_grad_()
    p = data["p"][:2].clone().requires_grad_()
    angle = models["learned"](q, p, current_heading(k, q))
    loss = yaw_objective(check, {"q": q, "p": p}, angle)[0].sum()
    gradients = torch.autograd.grad(loss, (q, p))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    bad = {
        "q": q.detach()[:1],
        "p": torch.full((1, 3), 10.0),
        "heading": data["heading"][:1],
    }
    with torch.no_grad():
        angle = models["learned"](bad["q"], bad["p"], bad["heading"])
    failed = evaluate_angles(check, bad, angle)
    assert not failed["valid"].any() and torch.isfinite(failed["q"]).all()
    reference_angle, reference_valid = optimize_task_angles(check, bad, updates=2)
    assert not reference_valid.any()
    torch.testing.assert_close(reference_angle, bad["heading"])
    iid_heading = next(
        row["mean_loss_all"]
        for row in meta["quality"]
        if row["split"] == "iid" and row["method"] == "heading"
    )
    improved = all(
        row["mean_loss_all"] < iid_heading
        for row in meta["quality"]
        if row["split"] == "iid" and row["method"] == "learned"
    )
    return {
        "checkpoint_cpu_prediction_parity": True,
        "repeatable_frozen_prediction": True,
        "inputs_unchanged": True,
        "independent_fk_success_and_task_invariants": True,
        "disjoint_spatial_holdout": True,
        "distinct_train_validation_test_tasks": True,
        "frozen_model_input_gradients": True,
        "unreachable_failure_and_reference_fallback": True,
        "iid_loss_improved_all_training_seeds": improved,
    }


def _plot(
    output: Path, training: list[dict[str, Any]], quality: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in training:
        if run["seed"] == training[0]["seed"]:
            history = run["history"]
            axes[0].plot(
                [x["step"] for x in history],
                [x["validation_loss"] for x in history],
                label=run["method"],
            )
    axes[0].set(
        xlabel="Training update",
        ylabel="Validation task loss",
        title="Checkpoint selection (first seed)",
    )
    axes[0].legend()
    for j, method in enumerate(METHODS):
        values = [
            np.mean(
                [
                    x["mean_cost_valid"]
                    for x in quality
                    if x["split"] == split
                    and x["method"] == method
                    and x["mean_cost_valid"] is not None
                ]
            )
            for split in SPLITS
        ]
        axes[1].bar(
            np.arange(len(SPLITS)) + (j - 2.5) * 0.13, values, width=0.13, label=method
        )
    axes[1].set(
        xticks=np.arange(len(SPLITS)),
        xticklabels=SPLITS,
        ylabel="Joint displacement² / rad²",
        title="Held-out tasks; valid costs, see failure rates",
    )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)


def run(cfg: GraspLearningCfg) -> dict[str, Any]:
    """Train paired models, evaluate fixed held-outs, and write reproducible artifacts."""
    cfg.__post_init__()
    torch.set_num_threads(4)
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    k = load_kinematics(cfg.device)
    device = k.tcp.device
    inner = DifferentiableIKLayer(k, DifferentiableIKCfg(max_iterations=cfg.ik_steps))
    check = DifferentiableIKLayer(
        k, DifferentiableIKCfg(max_iterations=cfg.check_steps)
    )
    train = sample_tasks(k, cfg.train_count, 23, "train")
    validation = sample_tasks(k, cfg.validation_count, 24, "validation")
    tests = {
        name: sample_tasks(k, cfg.test_count, 25 + i, name)
        for i, name in enumerate(SPLITS)
    }
    np.savez_compressed(
        output / "datasets.npz",
        **{
            f"{split}_{name}": value.cpu().numpy()
            for split, data in {
                "train": train,
                "validation": validation,
                **tests,
            }.items()
            for name, value in data.items()
        },
    )
    gradient_checks = validate_gradients(k, cfg.ik_steps)
    _sync(device)
    start = time.perf_counter()
    labels = optimize_task_angles(inner, train, cfg.reference_updates, cfg.chunk_size)
    _sync(device)
    label_seconds = time.perf_counter() - start
    np.savez_compressed(
        output / "teacher_labels.npz",
        angle=labels[0].cpu().numpy(),
        valid=labels[1].cpu().numpy(),
    )
    print(
        f"Teacher labels: {label_seconds:.2f}s, feasible {int(labels[1].sum())}/{cfg.train_count}",
        flush=True,
    )
    all_models, training = [], []
    for seed in cfg.seeds:
        models = {}
        for method in ("learned", "constant", "supervised"):
            models[method], history = fit_model(
                k,
                cfg,
                train,
                validation,
                seed,
                method,
                labels if method == "supervised" else None,
            )
            training.append(history)
            print(
                f"Trained {method} seed={seed}: {history['seconds']:.2f}s, step={history['selected_step']}",
                flush=True,
            )
        all_models.append(models)
    torch.save(
        {"states": {name: model.state_dict() for name, model in all_models[0].items()}},
        output / "models.pt",
    )
    for seed, models in zip(cfg.seeds[1:], all_models[1:]):
        torch.save(
            {"states": {name: model.state_dict() for name, model in models.items()}},
            output / f"models_seed_{seed}.pt",
        )
    quality, paired, replay_parts = [], {}, []
    _sync(device)
    start = time.perf_counter()
    for split, data in tests.items():
        heading = evaluate_angles(check, data, data["heading"], cfg.chunk_size)
        online_angle, online_found = optimize_task_angles(
            inner, data, cfg.reference_updates, cfg.chunk_size
        )
        online = evaluate_angles(check, data, online_angle, cfg.chunk_size)
        grid, grid_angles, grid_costs = grid_reference(
            check, data, cfg.grid_points, cfg.grid_chunk_size
        )
        fixed = {"heading": heading, "online": online, "grid": grid}
        for name, result in fixed.items():
            quality.append(
                {
                    "split": split,
                    "seed": None,
                    "method": name,
                    **summarize(result, heading, grid),
                }
            )
            for field, value in result.items():
                paired[f"{split}_{name}_{field}"] = value.detach().cpu().numpy()
        first_results = dict(fixed)
        for seed, models in zip(cfg.seeds, all_models):
            for name, model in models.items():
                with torch.no_grad():
                    angle = model(data["q"], data["p"], data["heading"])
                result = evaluate_angles(check, data, angle, cfg.chunk_size)
                quality.append(
                    {
                        "split": split,
                        "seed": seed,
                        "method": name,
                        **summarize(result, heading, grid),
                    }
                )
                for field, value in result.items():
                    paired[f"{split}_{name}_{seed}_{field}"] = (
                        value.detach().cpu().numpy()
                    )
                if seed == cfg.seeds[0]:
                    first_results[name] = result
        n = min(8, cfg.test_count)
        replay_parts.append(
            {
                "current_q": data["q"][:n],
                "positions": data["p"][:n],
                "split": np.array([split] * n),
                "grid_cost": grid_costs[:n],
                "reference_found": online_found[:n],
                **{
                    field: torch.stack(
                        [first_results[name][field][:n] for name in METHODS], 1
                    )
                    for field in heading
                },
            }
        )
    _sync(device)
    reference_seconds = time.perf_counter() - start
    replay = {
        key: np.concatenate(
            [
                (
                    part[key]
                    if isinstance(part[key], np.ndarray)
                    else part[key].detach().cpu().numpy()
                )
                for part in replay_parts
            ]
        )
        for key in replay_parts[0]
    }
    np.savez_compressed(
        output / "replay.npz", **replay, grid_angles=grid_angles.cpu().numpy()
    )
    np.savez_compressed(output / "paired_results.npz", **paired)
    timing = []
    for batch in (1, 64):
        data = sample_tasks(k, batch, 401 + batch, "iid")
        # Heading FK, prediction/search, and final check_steps IK are timed together.
        for method in METHODS:

            def operation() -> torch.Tensor:
                with torch.no_grad():
                    d = {**data, "heading": current_heading(k, data["q"])}
                    if method == "heading":
                        angle = d["heading"]
                    elif method in all_models[0]:
                        angle = all_models[0][method](d["q"], d["p"], d["heading"])
                    elif method == "grid":
                        return grid_reference(
                            check, d, cfg.grid_points, cfg.grid_chunk_size
                        )[0]["q"]
                if method == "online":
                    angle = optimize_task_angles(
                        inner, d, cfg.reference_updates, cfg.chunk_size
                    )[0]
                with torch.no_grad():
                    return check(target_pose(angle, d["p"]), d["q"])

            timing.append(
                {
                    "method": method,
                    "batch_size": batch,
                    **_measure(operation, device, 5),
                }
            )
    metrics = {
        "config": asdict(cfg),
        "methods": METHODS,
        "training": training,
        "quality": quality,
        "timing": timing,
        "teacher_label_seconds": label_seconds,
        "teacher_feasible_count": int(labels[1].sum()),
        "evaluation_reference_seconds": reference_seconds,
        "gradient_checks": gradient_checks,
        "model_fingerprint": k.model.fingerprint,
        "joint_names": k.model.joint_names,
        "urdf": get_data_path("Franka/Panda/PandaWithHand.urdf"),
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "torch": str(torch.__version__),
        "selected_viewer_seed": cfg.seeds[0],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    gradient_checks.update(validate_artifacts(output))
    metrics["gradient_checks"] = gradient_checks
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot(output, training, quality)
    metric_rows = []
    for split in SPLITS:
        for method in METHODS:
            rows = [r for r in quality if r["split"] == split and r["method"] == method]
            metric_rows.append(
                {
                    "split": split,
                    "method": method,
                    "success_rate": np.mean([r["success_rate"] for r in rows]),
                    "mean_cost_valid": (
                        np.mean(
                            [
                                r["mean_cost_valid"]
                                for r in rows
                                if r["mean_cost_valid"] is not None
                            ]
                        )
                        if any(r["mean_cost_valid"] is not None for r in rows)
                        else None
                    ),
                    "improvement_pct": (
                        100
                        * np.mean(
                            [
                                r["mean_relative_improvement"]
                                for r in rows
                                if r["mean_relative_improvement"] is not None
                            ]
                        )
                        if any(r["mean_relative_improvement"] is not None for r in rows)
                        else None
                    ),
                    "grid_gap_pct": (
                        100
                        * np.mean(
                            [
                                r["mean_relative_grid_gap"]
                                for r in rows
                                if r["mean_relative_grid_gap"] is not None
                            ]
                        )
                        if any(r["mean_relative_grid_gap"] is not None for r in rows)
                        else None
                    ),
                }
            )
    leaderboard = []
    for method in METHODS:
        rows = [r for r in metric_rows if r["method"] == method]
        leaderboard.append(
            {
                "method": method,
                "success_rate": np.mean([r["success_rate"] for r in rows]),
                "mean_cost_valid": (
                    np.mean(
                        [
                            r["mean_cost_valid"]
                            for r in rows
                            if r["mean_cost_valid"] is not None
                        ]
                    )
                    if any(r["mean_cost_valid"] is not None for r in rows)
                    else None
                ),
            }
        )
    leaderboard.sort(
        key=lambda r: (
            -r["success_rate"],
            float("inf") if r["mean_cost_valid"] is None else r["mean_cost_valid"],
        )
    )
    report = [
        "# Learning grasp yaw through differentiable IK",
        "",
        f"Hardware: {metrics['hardware']}; {cfg.device}; float32; Torch {metrics['torch']}; four CPU threads.",
        f"Train {cfg.train_count}, validation {cfg.validation_count}, {cfg.test_count} cases per held-out split. Seeds: {cfg.seeds}. Updates {cfg.updates}; batch {cfg.batch_size}.",
        f"Teacher-label generation: {label_seconds:.3f}s ({int(labels[1].sum())}/{cfg.train_count} feasible). This is additional supervised-baseline cost.",
        "",
        "Training wall time including validation/checkpoint selection:",
    ]
    report.extend(
        f"- {r['method']}, seed {r['seed']}: {r['seconds']:.3f}s; selected update {r['selected_step']}."
        for r in training
    )
    report.extend(
        [
            "",
            "## Time & Memory",
            "",
            _table(timing),
            "",
            "## Success & Other Metrics",
            "",
            _table(metric_rows),
            "",
            "## Leaderboard",
            "",
            _table(leaderboard),
            "",
            "## Protocol and limits",
            "",
            "- Learned and supervised networks have paired initialization and minibatches. Both select checkpoints on independent validation IK task loss. Constant yaw is one absolute scalar trained through IK. Heading equals the network's untrained output.",
            "- Teacher labels come only from training tasks; no labels or losses from held-out sets select a model. All three splits have independent RNG streams. Spatial Y holdout is disjoint from the training Y interval; state shift changes current joints, not grasp positions.",
            f"- Training and online search use {cfg.ik_steps}-step IK; every final method is evaluated with {cfg.check_steps} steps and the same current joints as seed. Failed targets are retained. Costs/gaps use paired-valid masks; exact counts and all-target errors are in metrics.json and paired_results.npz.",
            "- Runtime includes current-heading FK, network prediction or numerical search, and final IK. Grid is direct sampled search with a separate 16,384-candidate inference chunk by default. Two warmups, five synchronized samples; memory is measured in a separate warmed call. CPU RSS is retained delta, not peak. Training and label-generation wall times are single observed runs.",
            "- First seed supplies the viewer, without selecting by test performance. Per-seed results are retained; common test sets are not independent population replicates.",
            "- Numerical references are local Adam search and a sampled angular grid, neither a certified optimum. Unreachable/reference failures remain flagged. This is a local kinematic task, without collision/contact/dynamics or trajectory execution guarantees.",
            "- See validation.json for executable gradient, checkpoint and repeatability checks; formal compute/layer contracts remain covered by production tests.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n")
    (output / "validation.json").write_text(
        json.dumps(gradient_checks, indent=2) + "\n"
    )
    print(json.dumps(metric_rows, indent=2), flush=True)
    print(f"Report: {output/'report.md'}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--output", default="outputs/learn_grasp_yaw")
    args = parser.parse_args()
    run(
        GraspLearningCfg(
            device=args.device,
            seeds=tuple(args.seeds),
            updates=args.updates,
            train_count=args.train_count,
            test_count=args.test_count,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
