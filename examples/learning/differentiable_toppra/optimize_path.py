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

"""Optimize free Franka path nodes through physical retiming duration.

Run: embodichain run-example learning/differentiable_toppra/optimize_path --output outputs/retiming_demo
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import torch

from embodichain.compute.kinematics import SerialChainKinematics
from embodichain.compute.trajectory import JointPathTimingResult
from embodichain.learning.layers import (
    DifferentiableFKLayer,
    DifferentiableRetimingLayer,
)
from embodichain.utils import configclass

CASES = ("detour", "via_joint", "stationary")


@configclass
class ShowcaseCfg:
    """CPU-only demonstration settings, independent of a training recipe."""

    case: str = "detour"
    mode: str = "optimize"
    detach_duration: bool = False
    steps: int = 30
    velocity_scale: float = 1.0
    acceleration_scale: float = 1.0
    bottleneck: int = -1
    bottleneck_velocity: float = 0.25
    bottleneck_acceleration: float = 0.25

    def __post_init__(self) -> None:
        if self.case not in CASES or self.mode not in ("optimize", "retime"):
            raise ValueError("unknown showcase case or mode")
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or not 0 <= self.steps <= 100
        ):
            raise ValueError("steps must be an integer in [0,100]")
        for value in (
            self.velocity_scale,
            self.acceleration_scale,
            self.bottleneck_velocity,
            self.bottleneck_acceleration,
        ):
            if not np.isfinite(value) or not 0.25 <= value <= 1.0:
                raise ValueError("budget multipliers must be in [0.25,1]")
        if not -1 <= self.bottleneck < 7:
            raise ValueError("bottleneck must be -1 or a controlled joint index")


class ShowcaseRobot:
    """Fixed CPU geometry, limits and retiming layers; no per-call graph cache."""

    def __init__(self, urdf: Path | None = None) -> None:
        if urdf is None:
            from embodichain.data import get_data_path

            urdf = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
        self.urdf = urdf.resolve()
        xml = self.urdf.read_text()
        self.fk = DifferentiableFKLayer(
            SerialChainKinematics.from_urdf(
                xml, "fr3_hand_tcp", "base", dtype=torch.float64
            )
        ).eval()
        self.names = self.fk.model.joint_names
        if self.names != tuple(f"fr3_joint{i}" for i in range(1, 8)):
            raise ValueError(
                "This showcase requires the official seven-joint Franka chain"
            )
        limits = {
            j.attrib["name"]: float(j.find("limit").attrib["velocity"])
            for j in ET.fromstring(xml).findall("joint")
            if j.find("limit") is not None
        }
        self.velocity = torch.tensor(
            [min(limits[n], 1.0) for n in self.names], dtype=torch.float64
        )
        self.acceleration = torch.full((7,), 2.0, dtype=torch.float64)
        self.layer = (
            DifferentiableRetimingLayer(
                self.velocity, self.acceleration, sample_count=128
            )
            .double()
            .eval()
        )
        self.checker = (
            DifferentiableRetimingLayer(
                self.velocity, self.acceleration, sample_count=512
            )
            .double()
            .eval()
        )
        self.sha256 = hashlib.sha256(self.urdf.read_bytes()).hexdigest()

    def budgets(self, cfg: ShowcaseCfg) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve one immutable pair of shared request-level budgets."""
        velocity = self.velocity * cfg.velocity_scale
        acceleration = self.acceleration * cfg.acceleration_scale
        if cfg.bottleneck >= 0:
            velocity[cfg.bottleneck] *= cfg.bottleneck_velocity
            acceleration[cfg.bottleneck] *= cfg.bottleneck_acceleration
        return velocity, acceleration

    def initial(self, case: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic six-node path and its fixed-node mask."""
        q0 = torch.tensor(
            [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=torch.float64
        ).clamp(min=self.fk.lower + 0.5, max=self.fk.upper - 0.5)
        s = torch.linspace(0, 1, 6, dtype=torch.float64)[:, None]
        d = q0.new_tensor([0.30, 0.15, -0.20, 0.20, 0.15, -0.10, 0.20])
        b = q0.new_tensor([0.25, -0.20, 0.30, -0.15, 0.25, 0.20, -0.25])
        points = q0 + s * d + torch.sin(2 * torch.pi * s) * b
        fixed = torch.tensor([True, False, False, case == "via_joint", False, True])
        if case == "stationary":
            points = q0.expand(6, -1).clone()
        return points, fixed

    def solve(
        self,
        points: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        *,
        dense: bool = False,
    ) -> JointPathTimingResult:
        """Call the public layer without rewriting its timing or derivatives."""
        layer = self.checker if dense else self.layer
        return layer(
            points[None], velocity_limits=velocity, acceleration_limits=acceleration
        )

    def valid(
        self,
        result: JointPathTimingResult,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> bool:
        """Check the native samples; this is not continuous collision certification."""
        if not result.success.item():
            return False
        n = int(result.sample_lengths.item())
        q, v, a = (
            result.positions[0, :n],
            result.velocities[0, :n],
            result.accelerations[0, :n],
        )
        return bool(
            (q >= self.fk.lower - 1e-8).all()
            and (q <= self.fk.upper + 1e-8).all()
            and (v.abs() / velocity <= 1 + 1e-5).all()
            and (a.abs() / acceleration <= 1 + 1e-5).all()
        )

    @torch.no_grad()
    def row(
        self, points: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor
    ) -> dict:
        """Make an owned, graph-free display snapshot from actual solver output."""
        result = self.solve(points, velocity, acceleration)
        dense = self.solve(points, velocity, acceleration, dense=True)
        n = int(result.sample_lengths.item())
        copy = lambda x: x.detach().cpu().numpy().copy()
        return {
            "duration": float(result.duration[0]) if result.success.item() else None,
            "success": bool(result.success.item()),
            "valid": self.valid(dense, velocity, acceleration),
            "length": n,
            "positions": copy(result.positions[0, :n]),
            "velocities": copy(result.velocities[0, :n]),
            "accelerations": copy(result.accelerations[0, :n]),
            "dt": copy(result.dt[0, :n]),
            "tcp": (
                copy(self.fk(result.positions[0, :n])[:, :3, 3])
                if n
                else np.empty((0, 3))
            ),
            "nodes_tcp": copy(self.fk(points)[:, :3, 3]),
            "velocity": copy(velocity),
            "acceleration": copy(acceleration),
        }


@dataclass(frozen=True)
class Snapshot:
    """A complete immutable-by-ownership UI/export generation."""

    cfg: ShowcaseCfg
    initial: np.ndarray
    current: np.ndarray
    fixed: np.ndarray
    before: dict
    after: dict
    gradient: np.ndarray | None
    history: tuple[dict, ...]
    status: str
    reason: str
    seconds: float

    @property
    def iteration(self) -> int:
        """Number of actual accepted updates."""
        return len(self.history) - 1


def optimize(
    robot: ShowcaseRobot,
    cfg: ShowcaseCfg,
    *,
    resume: Snapshot | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    publish: Callable[[Snapshot], None] | None = None,
) -> Snapshot:
    """Apply first-order duration descent, never differentiating optimizer steps.

    Detached mode performs the same retiming forward but has no autograd
    objective. It stops without substituting a finite-difference optimizer.
    """
    start = time.perf_counter()
    initial, fixed = robot.initial(cfg.case)
    ids = (~fixed).nonzero().flatten()
    if resume is not None:
        old, new = asdict(resume.cfg), asdict(cfg)
        old.pop("steps")
        new.pop("steps")
        if old != new or cfg.steps < resume.iteration:
            raise ValueError(
                "resume requires identical case/budgets and a nondecreasing step target"
            )
    q = initial.clone() if resume is None else torch.from_numpy(resume.current.copy())
    velocity, acceleration = robot.budgets(cfg)
    before_v, before_a = (
        (robot.velocity, robot.acceleration)
        if cfg.mode == "retime"
        else (velocity, acceleration)
    )
    before = robot.row(initial, before_v, before_a)
    history = [] if resume is None else list(resume.history)
    accepted = 0 if resume is None else resume.iteration
    last_step, last_trials = 0.0, 0

    def snapshot(g: torch.Tensor | None, status: str, reason: str) -> Snapshot:
        row = robot.row(q, velocity, acceleration)
        if len(history) <= accepted:
            history.append(
                {
                    "iteration": accepted,
                    "duration": row["duration"],
                    "step_size": last_step,
                    "trials": last_trials,
                    "gradient_norm": None if g is None else float(g.norm()),
                }
            )
        result = Snapshot(
            cfg,
            initial.numpy().copy(),
            q.detach().numpy().copy(),
            fixed.numpy().copy(),
            before,
            row,
            None if g is None else g.numpy().copy(),
            tuple(history),
            status,
            reason,
            time.perf_counter() - start,
        )
        if publish is not None:
            publish(result)
        return result

    if cfg.mode == "retime":
        return snapshot(None, "retimed", "Fixed geometry; only physical timing changed")
    while True:
        if cancelled():
            return snapshot(None, "cancelled", "Last accepted path retained")
        free = q[ids].detach().clone().requires_grad_(True)
        points = initial.index_copy(0, ids, free)
        timed = robot.solve(points, velocity, acceleration)
        if not timed.success.item() or not before["valid"]:
            return snapshot(None, "failed", "Initial path could not be validated")
        duration = timed.duration[0]
        if cfg.detach_duration:
            duration = duration.detach()
        if not duration.requires_grad:
            return snapshot(
                None,
                "disconnected",
                "duration.detach(): no autograd objective; no update",
            )
        g = torch.autograd.grad(duration, free)[0].detach()
        if not torch.isfinite(g).all():
            return snapshot(None, "failed", "Non-finite duration gradient")
        if accepted >= cfg.steps:
            return snapshot(g, "complete", "Requested accepted-step budget reached")
        if g.abs().max() < 1e-8:
            return snapshot(
                g,
                "stationary" if duration.item() == 0 else "converged",
                "No update needed: zero/small duration gradient",
            )
        snapshot(
            g,
            "optimizing",
            "Autograd through spline and timing; fixed nodes remain constant",
        )
        direction = -g / g.abs().max()
        improved = False
        for trial in range(10):
            if cancelled():
                return snapshot(g, "cancelled", "Last accepted path retained")
            alpha = 0.02 * 0.5**trial
            candidate_free = (free.detach() + alpha * direction).clamp(
                min=robot.fk.lower + 0.02, max=robot.fk.upper - 0.02
            )
            delta = candidate_free - free.detach()
            candidate = initial.index_copy(0, ids, candidate_free)
            with torch.no_grad():
                check = robot.solve(candidate, velocity, acceleration, dense=True)
            slope = float((g * delta).sum())
            if (
                slope < 0
                and robot.valid(check, velocity, acceleration)
                and check.duration.item() <= duration.item() + 1e-4 * slope
            ):
                q = candidate.detach()
                accepted += 1
                last_step, last_trials = alpha, trial + 1
                improved = True
                break
        if not improved:
            return snapshot(g, "stalled", "Backtracking found no valid descending step")


def check_gradient(robot: ShowcaseRobot, state: Snapshot) -> dict:
    """Compare a local autograd directional derivative with centered differences."""
    if not state.after["success"]:
        return {
            "status": "failed",
            "passed": None,
            "reason": "No valid timing solution",
        }
    if state.cfg.detach_duration or state.cfg.mode != "optimize":
        return {
            "status": "disconnected",
            "passed": None,
            "reason": "No autograd objective in this mode",
        }
    q = torch.from_numpy(state.current.copy())
    fixed = torch.from_numpy(state.fixed.copy())
    ids = (~fixed).nonzero().flatten()
    free = q[ids].clone().requires_grad_(True)
    v, a = robot.budgets(state.cfg)

    def duration(x: torch.Tensor) -> torch.Tensor:
        return robot.solve(q.index_copy(0, ids, x), v, a).duration[0]

    value = duration(free)
    g = torch.autograd.grad(value, free)[0]
    if g.abs().max() < 1e-8:
        return {
            "status": "no_direction",
            "passed": None,
            "autograd": None,
            "finite_difference": None,
            "gradient_norm": float(g.norm().detach()),
            "reason": "Zero/small autograd gradient: no descent direction; difference check skipped",
        }
    d = -g.detach() / g.detach().abs().max()
    with torch.no_grad():
        fd = (duration(free + 1e-6 * d) - duration(free - 1e-6 * d)) / 2e-6
    expected = (g * d).sum()
    return {
        "status": "checked",
        "passed": bool(torch.isclose(fd, expected, rtol=2e-3, atol=2e-4)),
        "autograd": float(expected.detach()),
        "finite_difference": float(fd),
        "absolute_error": float((fd - expected).abs().detach()),
        "relative_error": float(
            ((fd - expected).abs() / expected.abs().clamp_min(1e-12)).detach()
        ),
    }


def save_run(robot: ShowcaseRobot, state: Snapshot, output: Path) -> None:
    """Save a fresh, portable result directory without overwriting earlier runs."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": asdict(state.cfg),
        "urdf_sha256": robot.sha256,
        "joint_names": list(robot.names),
        "kinematics_fingerprint": robot.fk.model.fingerprint,
        "status": state.status,
        "reason": state.reason,
        "seconds": state.seconds,
        "iterations": state.iteration,
        "before_duration": state.before["duration"],
        "after_duration": state.after["duration"],
        "sampled_valid": state.after["valid"],
        "fixed_node_max_error": float(
            np.abs(state.current[state.fixed] - state.initial[state.fixed]).max()
        ),
        "gradient_check": check_gradient(robot, state),
        "history": state.history,
    }
    (output / "run.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    arrays = {
        "initial": state.initial,
        "current": state.current,
        "fixed": state.fixed,
        "gradient": state.gradient if state.gradient is not None else np.empty((0, 7)),
    }
    for side, row in (("before", state.before), ("after", state.after)):
        for key in (
            "positions",
            "velocities",
            "accelerations",
            "dt",
            "velocity",
            "acceleration",
            "tcp",
            "nodes_tcp",
        ):
            arrays[f"{side}_{key}"] = row[key]
    np.savez_compressed(output / "trajectories.npz", **arrays)
    plot_run(state, output)


def load_run(robot: ShowcaseRobot, source: Path) -> Snapshot:
    """Load recorded output for playback, verifying fixed robot identity."""
    meta = json.loads((source / "run.json").read_text())
    if meta["urdf_sha256"] != robot.sha256 or tuple(meta["joint_names"]) != robot.names:
        raise ValueError("Saved result belongs to a different robot geometry/order")
    with np.load(source / "trajectories.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    cfg = ShowcaseCfg(**meta["config"])
    before = robot.row(
        torch.from_numpy(data["initial"]),
        torch.from_numpy(data["before_velocity"]),
        torch.from_numpy(data["before_acceleration"]),
    )
    after = robot.row(
        torch.from_numpy(data["current"]),
        torch.from_numpy(data["after_velocity"]),
        torch.from_numpy(data["after_acceleration"]),
    )
    for side, row in (("before", before), ("after", after)):
        for key in ("positions", "velocities", "accelerations", "dt"):
            np.testing.assert_allclose(
                row[key], data[f"{side}_{key}"], rtol=1e-7, atol=1e-9
            )
    return Snapshot(
        cfg,
        data["initial"],
        data["current"],
        data["fixed"],
        before,
        after,
        data["gradient"] if data["gradient"].size else None,
        tuple(meta["history"]),
        meta["status"],
        meta["reason"],
        meta["seconds"],
    )


def plot_run(state: Snapshot, output: Path) -> None:
    """Export scientific plots of actual solver samples, gradients and costs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    h = state.history
    axes[0, 0].plot(
        [x["iteration"] for x in h],
        [x["duration"] for x in h],
        "o-",
        color="#1b9e77",
        markersize=3,
    )
    axes[0, 0].set(xlabel="Accepted update", ylabel="Duration / s", title=state.status)
    for label, row, color in (
        ("Initial", state.before, "#568ce8"),
        ("Current", state.after, "#22ae89"),
    ):
        times = row["dt"].cumsum()
        if len(times):
            axes[0, 1].plot(
                row["tcp"][:, 0], row["tcp"][:, 2], label=label, color=color
            )
            axes[1, 0].plot(
                times,
                (np.abs(row["velocities"]) / row["velocity"]).max(-1),
                color=color,
                label=label + " velocity",
            )
            axes[1, 0].plot(
                times,
                (np.abs(row["accelerations"]) / row["acceleration"]).max(-1),
                color=color,
                linestyle="--",
                label=label + " acceleration",
            )
    axes[0, 1].set(
        xlabel="TCP x / m", ylabel="TCP z / m", title="Root-frame path projection"
    )
    axes[0, 1].legend()
    axes[1, 0].axhline(1, color="grey", linewidth=1)
    axes[1, 0].set(xlabel="Physical time / s", ylabel="Max joint limit utilization")
    axes[1, 0].legend(fontsize=7)
    if state.gradient is not None:
        heat = axes[1, 1].imshow(state.gradient, aspect="auto", cmap="RdBu_r")
        fig.colorbar(heat, ax=axes[1, 1], label="s/rad")
    else:
        axes[1, 1].text(0.5, 0.5, "No autograd objective", ha="center")
    axes[1, 1].set(
        title="Current free-node duration gradient",
        xlabel="Joint index",
        ylabel="Free-node index",
    )
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)


def smoke(robot: ShowcaseRobot, output: Path) -> None:
    """Exercise actual default cases, detached loss, time scaling and exports."""
    output.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        cfg = ShowcaseCfg(case=case)
        initial = optimize(robot, ShowcaseCfg(case=case, steps=0))
        initial_check = check_gradient(robot, initial)
        if case == "stationary":
            assert (
                initial_check["passed"] is None
                and initial_check["status"] == "no_direction"
            )
        else:
            assert initial_check["passed"]
        state = optimize(robot, cfg)
        assert state.after["valid"] and np.array_equal(
            state.initial[state.fixed], state.current[state.fixed]
        )
        durations = [h["duration"] for h in state.history]
        assert all(b <= a + 1e-10 for a, b in zip(durations, durations[1:]))
        if case == "detour":
            assert state.after["duration"] < 0.95 * state.before["duration"]
        if case == "stationary":
            assert (
                state.after["duration"] == 0
                and state.after["length"] == 2
                and state.iteration == 0
            )
        save_run(robot, state, output / case)
        replay = load_run(robot, output / case)
        np.testing.assert_array_equal(replay.current, state.current)
    disconnected = optimize(robot, ShowcaseCfg(detach_duration=True))
    assert disconnected.gradient is None and disconnected.iteration == 0
    np.testing.assert_array_equal(disconnected.current, disconnected.initial)
    retimed = optimize(
        robot, ShowcaseCfg(mode="retime", velocity_scale=0.5, acceleration_scale=0.25)
    )
    np.testing.assert_array_equal(retimed.current, retimed.initial)
    np.testing.assert_allclose(
        retimed.after["duration"], 2 * retimed.before["duration"], rtol=1e-6
    )
    save_run(robot, disconnected, output / "detached")
    print(
        "Numeric smoke passed: descent, fixed anchors, derivatives, static/detached controls, timing and replay",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--case", choices=CASES, default="detour")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--detach-duration", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    robot = ShowcaseRobot(args.urdf)
    if args.smoke:
        smoke(robot, args.output)
    else:
        state = optimize(
            robot,
            ShowcaseCfg(
                case=args.case, steps=args.steps, detach_duration=args.detach_duration
            ),
        )
        save_run(robot, state, args.output)
        print(
            json.dumps(
                {
                    "status": state.status,
                    "before": state.before["duration"],
                    "after": state.after["duration"],
                    "iterations": state.iteration,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        if not state.after["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
