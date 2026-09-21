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

"""Overlay manipulability-guided joint-trajectory variants on one reference.

Every variant is produced by
:func:`~embodichain.lab.sim.motion.expansion.manipulability.manipulability_guided_residual`
against the same annotated reference. One run writes two views of that same
set, so the two figures can never disagree:

* the joint paths, showing that phase endpoints stay exact while the interior
  of each joint moves;
* the manipulability measured along those paths, showing where each variant's
  bottleneck lands relative to the band edges that selected it.

The robot is driven through its kinematic chain and solver only. No simulation
manager, renderer, physics backend or display is started, so this runs on a
plain checkout::

    python scripts/tutorials/sim/planner/manipulability_augmentation_plot.py

Colors encode the band a variant actually reached, not the band it requested.
A variant that could not reach its target is drawn dashed, which is the honest
outcome when the requested band is unreachable at the configured residual
scale. Manipulability ranks postures only: none of these paths is certified
collision free or dynamically feasible.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/embodichain-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from embodichain.lab.sim.motion.expansion.contracts import (
    TrajectoryPhase,
    TrajectoryTemplate,
)
from embodichain.lab.sim.motion.expansion.manipulability import (
    GuidedResidual,
    ManipulabilityBands,
    ManipulabilityProfile,
    describe_manipulability,
    manipulability_guided_residual,
)
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.utils import logger

DEFAULT_OUTPUT = Path("docs/source/_static/tutorials/manipulability_augmentation.png")
"""Repository-relative destination tracked alongside the other tutorial figures."""

DEFAULT_PROFILE_OUTPUT = Path(
    "docs/source/_static/tutorials/manipulability_augmentation_profile.png"
)
"""Destination of the companion manipulability-versus-time figure."""

START_QPOS = (0.0, -1.2, 1.6, -1.2, -1.57, 0.0)
"""Well-conditioned UR5 start posture; the zero posture is singular."""

MID_QPOS = (0.45, -1.4, 1.4, -1.1, -0.45, 0.2)
"""Interior waypoint that swings the wrist toward its singular alignment.

The reference has to be least manipulable somewhere in its interior for band
targeting to be visible at all. ``joint_residual`` preserves every phase
endpoint exactly, so a reference whose bottleneck sits on an endpoint cannot
have that bottleneck moved by any residual, and every variant would report the
reference score. Passing near the wrist singularity puts the bottleneck in the
middle of the free phase, where the operator can act on it.
"""

END_QPOS = (0.9, -1.6, 1.2, -1.0, -1.57, 0.4)
"""End posture reached by the reference path."""

BAND_EDGES = (0.8, 1.2)
"""Ratios against the reference bottleneck; three bands, both ends open."""

BAND_LABELS = ("tighter than reference", "near reference", "more open")
BAND_COLORS = ("#c0392b", "#7f8c8d", "#1f77b4")


def build_solver(device: torch.device):
    """Build a UR5 arm solver from assets without starting a simulation.

    ``WorkspaceAnalyzer`` and the augmentation operators only need forward
    kinematics, joint limits and the solver Jacobian, so an asset-backed
    kinematic chain is enough. This mirrors
    ``scripts/benchmark/workspace_analyzer/benchmark_robot_workspace.py``.

    Args:
        device: Torch device used for kinematics.

    Returns:
        The initialized arm solver.
    """
    # Batched solver kernels run under Warp; nothing else initializes it here.
    wp.init()
    preset = URRobotCfg.from_dict({"robot_type": "ur5"})
    chain = preset.build_pk_serial_chain(device)["arm"]
    cfg = preset.solver_cfg["arm"]
    cfg.joint_names = chain.get_joint_parameter_names()
    solver = cfg.init_solver(device=device, pk_serial_chain=chain)
    solver.compiled_fk = chain.forward_kinematics_tensor
    return solver


def build_reference(solver, *, samples: int, control_dt: float) -> TrajectoryTemplate:
    """Interpolate one annotated reference between two fixed postures.

    The path runs start to end through :data:`MID_QPOS`, so its least
    manipulable sample lies in the interior rather than on an endpoint. The
    whole path is a single free phase that authorizes ``joint_residual``, which
    is what lets an operator reshape its interior. Real references carry contact
    and hold phases too; those are the phases an operator must leave untouched.

    Args:
        solver: Arm solver supplying the joint names.
        samples: Number of trajectory samples, at least three.
        control_dt: Uniform arrival interval in seconds.

    Returns:
        The annotated reference template.
    """
    device = solver.lower_qpos_limits.device
    start = torch.tensor(START_QPOS, dtype=torch.float32, device=device)
    middle = torch.tensor(MID_QPOS, dtype=torch.float32, device=device)
    end = torch.tensor(END_QPOS, dtype=torch.float32, device=device)
    first = (samples + 1) // 2
    lead = torch.linspace(0.0, 1.0, first, device=device)[:, None]
    tail = torch.linspace(0.0, 1.0, samples - first + 1, device=device)[1:, None]
    positions = torch.cat(
        (start + lead * (middle - start), middle + tail * (end - middle)), dim=0
    )
    dt = torch.full((samples,), control_dt, dtype=torch.float32, device=device)
    dt[0] = 0.0
    joint_names = tuple(solver.joint_names)
    return TrajectoryTemplate(
        source_id="handwritten_qpos",
        source_revision="v1",
        template_id="ur5_reference",
        joint_names=joint_names,
        positions=positions,
        dt=dt,
        phases=(
            TrajectoryPhase(
                "move", 0, samples, kind="free", allowed_operators=("joint_residual",)
            ),
        ),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=tuple(range(len(joint_names))),
    )


def collect_variants(
    solver,
    reference: TrajectoryTemplate,
    *,
    bands: ManipulabilityBands,
    per_band: int,
    normalized_scale: float,
    proposals: int,
    seed: int,
) -> list[GuidedResidual]:
    """Steer residual proposals toward every band in turn.

    Each draw uses its own generator so the figure is reproducible and no two
    variants share a stream. Nothing here consumes the global RNG.

    Args:
        solver: Arm solver supplying Jacobians for scoring.
        reference: Annotated reference the operator reshapes.
        bands: Reference-normalized bands the draws are steered toward.
        per_band: Number of draws per band.
        normalized_scale: Residual size as a fraction of each joint range.
        proposals: Proposals scored per draw before the nearest is kept.
        seed: Base seed; each draw offsets it deterministically.

    Returns:
        One selected residual per draw, in band-major order.
    """
    joint_limits = torch.stack(
        (solver.lower_qpos_limits, solver.upper_qpos_limits), dim=-1
    )
    results: list[GuidedResidual] = []
    for target_band in range(bands.count):
        for draw in range(per_band):
            generator = torch.Generator()
            generator.manual_seed(seed + target_band * 1000 + draw)
            results.append(
                manipulability_guided_residual(
                    reference,
                    joint_limits=joint_limits,
                    normalized_scale=normalized_scale,
                    generator=generator,
                    jacobian_fn=solver.get_jacobian,
                    bands=bands,
                    target_band=target_band,
                    proposals=proposals,
                )
            )
    return results


def legend_handles() -> list[Line2D]:
    """Build the band legend shared by both figures.

    Returns:
        Proxy artists for the reference, each reached band, and the dashed
        style used when a variant could not reach its requested band.
    """
    handles = [Line2D([], [], color="black", linewidth=2.2, label="reference")]
    handles += [
        Line2D([], [], color=color, linewidth=1.4, label=f"reached: {label}")
        for color, label in zip(BAND_COLORS, BAND_LABELS)
    ]
    # Black keeps this entry reading as a line style: unmatched variants stay
    # colored by the band they actually reached.
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.4,
            linestyle="--",
            label="target band not reached",
        )
    )
    return handles


def build_figure(
    reference: TrajectoryTemplate,
    variants: list[GuidedResidual],
    *,
    bands: ManipulabilityBands,
    reference_bottleneck: float,
) -> Figure:
    """Draw every joint of every variant over the reference it came from.

    Args:
        reference: The unmodified reference path.
        variants: Selected residuals to overlay.
        bands: Bands whose edges annotate the figure.
        reference_bottleneck: Reference bottleneck score normalizing the bands.

    Returns:
        The assembled figure.
    """
    names = reference.joint_names
    time = torch.cumsum(reference.dt, dim=0).cpu().numpy()
    positions = reference.positions.cpu().numpy()
    columns = 3
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(13.5, 3.1 * rows), sharex=True)
    flat = axes.reshape(-1)

    for index, name in enumerate(names):
        axis = flat[index]
        for variant in variants:
            axis.plot(
                time,
                variant.template.positions[:, index].cpu().numpy(),
                color=BAND_COLORS[variant.band],
                linewidth=1.0,
                linestyle="-" if variant.matched else "--",
                alpha=0.75,
            )
        axis.plot(time, positions[:, index], color="black", linewidth=2.2, zorder=5)
        axis.set_title(name, fontsize=10)
        axis.set_ylabel("rad", fontsize=9)
        axis.grid(alpha=0.25)
    for index in range(len(names), flat.size):
        flat[index].set_visible(False)
    for axis in flat[-columns:]:
        if axis.get_visible():
            axis.set_xlabel("time [s]", fontsize=9)

    figure.legend(
        handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9
    )
    matched = sum(1 for variant in variants if variant.matched)
    figure.suptitle(
        "Manipulability-guided joint residuals on one UR5 reference\n"
        f"{len(variants)} variants, {matched} reached their target band; "
        f"band edges {BAND_EDGES} x reference bottleneck w={reference_bottleneck:.4f}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


def build_profile_figure(
    reference_profile: ManipulabilityProfile,
    variants: list[GuidedResidual],
    *,
    time: "np.ndarray",
    reference_bottleneck: float,
) -> Figure:
    """Draw measured manipulability along every variant and the reference.

    Scores come from the profile each variant was selected with, not from a
    fresh evaluation, so the curves are exactly the evidence the operator
    ranked. The marker on each curve is that trajectory's bottleneck, which is
    the single value its band membership is decided by.

    Args:
        reference_profile: Manipulability measured along the reference.
        variants: Selected residuals to overlay.
        time: Sample arrival times shared by every trajectory.
        reference_bottleneck: Reference bottleneck normalizing the band edges.

    Returns:
        The assembled figure.
    """
    figure, axis = plt.subplots(figsize=(13.5, 5.4))
    for variant in variants:
        scores = variant.profile.scores.numpy()
        color = BAND_COLORS[variant.band]
        axis.plot(
            time,
            scores,
            color=color,
            linewidth=1.0,
            linestyle="-" if variant.matched else "--",
            alpha=0.75,
        )
        lowest = int(scores.argmin())
        axis.plot(time[lowest], scores[lowest], marker="o", markersize=4.5, color=color)

    reference_scores = reference_profile.scores.numpy()
    axis.plot(time, reference_scores, color="black", linewidth=2.2, zorder=5)
    lowest = int(reference_scores.argmin())
    axis.plot(
        time[lowest],
        reference_scores[lowest],
        marker="o",
        markersize=6.5,
        color="black",
        zorder=6,
    )

    for edge in BAND_EDGES:
        level = edge * reference_bottleneck
        axis.axhline(level, color="0.55", linestyle=":", linewidth=1.0, zorder=1)
        # Anchored left, where the curves run high and leave the band lines clear.
        axis.annotate(
            f"{edge:g}x reference bottleneck",
            xy=(time[0], level),
            xytext=(6, 4),
            textcoords="offset points",
            va="bottom",
            ha="left",
            fontsize=8,
            color="0.35",
        )

    axis.set_xlabel("time [s]", fontsize=9)
    axis.set_ylabel("Yoshikawa manipulability w", fontsize=9)
    axis.grid(alpha=0.25)
    axis.margins(x=0.02)
    ratio = axis.secondary_yaxis(
        "right",
        functions=(
            lambda value: value / reference_bottleneck,
            lambda value: value * reference_bottleneck,
        ),
    )
    ratio.set_ylabel("ratio to reference bottleneck", fontsize=9)

    figure.legend(
        handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9
    )
    matched = sum(1 for variant in variants if variant.matched)
    figure.suptitle(
        "Manipulability measured along the same variants\n"
        f"{len(variants)} variants, {matched} reached their target band; "
        f"markers are each trajectory's bottleneck, the value its band is read from",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    return figure


def parse_arguments() -> argparse.Namespace:
    """Parse the figure's optional shape and destination arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG destination (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=DEFAULT_PROFILE_OUTPUT,
        help=f"Manipulability PNG (default: {DEFAULT_PROFILE_OUTPUT.as_posix()}).",
    )
    parser.add_argument(
        "--samples", type=int, default=41, help="Samples in the reference path."
    )
    parser.add_argument(
        "--per-band", type=int, default=4, help="Variants drawn per band."
    )
    parser.add_argument(
        "--normalized-scale",
        type=float,
        default=0.012,
        help="Residual size as a fraction of each joint's declared range.",
    )
    parser.add_argument(
        "--proposals",
        type=int,
        default=6,
        help="Proposals scored per draw before the nearest band is kept.",
    )
    parser.add_argument(
        "--control-dt", type=float, default=0.05, help="Arrival interval in seconds."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base sampling seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for kinematics.")
    return parser.parse_args()


def main() -> None:
    """Build the reference, steer residuals toward each band and write both PNGs."""
    args = parse_arguments()
    if args.samples < 3 or args.per_band < 1 or args.proposals < 1:
        raise SystemExit("samples must be >= 3; per-band and proposals must be >= 1")
    device = torch.device(args.device)
    solver = build_solver(device)
    reference = build_reference(
        solver, samples=args.samples, control_dt=args.control_dt
    )

    profile = describe_manipulability(
        solver.get_jacobian(reference.positions), phases=reference.phases
    )
    bottleneck = profile.bottleneck
    if bottleneck <= 0:
        raise SystemExit(
            "The reference passes through a singular posture; choose another path."
        )
    bands = ManipulabilityBands(BAND_EDGES, reference=bottleneck)

    variants = collect_variants(
        solver,
        reference,
        bands=bands,
        per_band=args.per_band,
        normalized_scale=args.normalized_scale,
        proposals=args.proposals,
        seed=args.seed,
    )
    time = torch.cumsum(reference.dt, dim=0).cpu().numpy()
    figures = {
        args.output: build_figure(
            reference, variants, bands=bands, reference_bottleneck=bottleneck
        ),
        args.output_profile: build_profile_figure(
            profile,
            variants,
            time=time,
            reference_bottleneck=bottleneck,
        ),
    }
    for destination, figure in figures.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)

    reached = sum(1 for variant in variants if variant.matched)
    # Each variant's own selection profile, never a fresh evaluation.
    spread = [variant.profile.bottleneck for variant in variants]
    logger.log_info(
        f"Reference bottleneck w={bottleneck:.5f}; "
        f"{reached}/{len(variants)} variants reached their target band; "
        f"variant bottlenecks span [{min(spread):.5f}, {max(spread):.5f}]."
    )
    for destination in figures:
        logger.log_info(f"Wrote {destination.resolve()}")


if __name__ == "__main__":
    main()
