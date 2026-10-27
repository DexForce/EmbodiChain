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

"""Visualize pure-trajectory task-space deformation and segment stitching.

This example is analytic and CPU-only: three prismatic joints stand in for a
Cartesian tool frame, and two abstract joints illustrate multi-demo phase
recombination. It intentionally uses no images, affordance labels, simulator,
or task environment. The generated trajectories still require physical
validation before they may be treated as executable expert data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import torch

from embodichain.compute.kinematics import DifferentiableIKCfg
from embodichain.lab.sim.motion.expansion import (
    MinimumJerkBridgePlanner,
    TaskSpaceDeformationConstraint,
    TrajectoryPhase,
    TrajectorySegment,
    TrajectorySegmentBank,
    TrajectoryTemplate,
    deform_task_space_trajectory,
    extract_trajectory_segments,
    project_task_space_trajectory,
    stitch_trajectory_segments,
    task_space_from_joint_template,
)

__all__ = ["main"]

_DT = 0.05
_TASK_SPACE_SEEDS = (2, 0, 1, 4)
_SOURCE_COLORS = {
    "demo_A": "#4C78A8",
    "demo_B": "#59A14F",
    "demo_C": "#F28E2B",
}


def _prismatic_kinematics(qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map three analytic prismatic joints onto Cartesian translation."""
    poses = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(
        qpos.shape[0], 1, 1
    )
    poses[:, :3, 3] = qpos
    jacobians = torch.zeros(qpos.shape[0], 6, 3, dtype=qpos.dtype, device=qpos.device)
    jacobians[:, :3] = torch.eye(3, dtype=qpos.dtype, device=qpos.device)
    return poses, jacobians


def _task_space_template(sample_count: int = 81) -> TrajectoryTemplate:
    progress = torch.linspace(0, 1, sample_count, dtype=torch.float64)
    positions = torch.stack(
        (
            0.25 + 0.50 * progress,
            0.04 * torch.sin(2 * torch.pi * progress),
            0.30 + 0.08 * torch.sin(torch.pi * progress),
        ),
        dim=-1,
    )
    dt = torch.full((sample_count,), _DT, dtype=torch.float64)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id="analytic_prismatic_demo",
        source_revision="one",
        template_id="nominal_transfer",
        joint_names=("x", "y", "z"),
        positions=positions,
        dt=dt,
        phases=(TrajectoryPhase("transfer", 0, sample_count),),
        controlled_joint_indices=(0, 1, 2),
        validator_id="demo_only",
    )


def _render_task_space_demo(output_path: Path, *, show: bool) -> dict[str, object]:
    from matplotlib import pyplot as plt

    source = _task_space_template()
    task_space = task_space_from_joint_template(
        source,
        frame_id="analytic_tool",
        kinematics=_prismatic_kinematics,
        editable_phase_ids=("transfer",),
    )
    constraint = TaskSpaceDeformationConstraint(
        "transfer",
        translation_axes=(False, True, False),
        rotation_axes=(False, False, False),
        max_translation_m=0.14,
    )
    limits = torch.tensor([[0.0, 1.0], [-0.3, 0.3], [0.0, 0.6]], dtype=torch.float64)
    ik_cfg = DifferentiableIKCfg(
        max_iterations=8,
        damping=1e-4,
        step_size=1.0,
        pos_eps=1e-6,
        rot_eps=1e-6,
    )
    times = source.dt.cumsum(0).numpy()
    nominal = source.positions.numpy()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(
        nominal[:, 0],
        nominal[:, 1],
        color="black",
        linewidth=2.5,
        label="nominal",
    )
    axes[1].plot(
        times,
        nominal[:, 1],
        color="black",
        linewidth=2.5,
        label="nominal",
    )
    colors = plt.cm.viridis(torch.linspace(0.15, 0.9, len(_TASK_SPACE_SEEDS)).numpy())
    variants = []
    for ordinal, (seed, color) in enumerate(
        zip(_TASK_SPACE_SEEDS, colors, strict=True), 1
    ):
        target = deform_task_space_trajectory(
            task_space,
            constraints={"transfer": constraint},
            generator=torch.Generator().manual_seed(seed),
        )
        projection = project_task_space_trajectory(
            target,
            source,
            kinematics=_prismatic_kinematics,
            joint_limits=limits,
            ik_cfg=ik_cfg,
        )
        if not projection.accepted:
            raise RuntimeError(f"task-space demo variant {ordinal} failed analytic IK")
        positions = projection.template.positions.numpy()
        label = f"variant {ordinal} · seed {seed}"
        axes[0].plot(positions[:, 0], positions[:, 1], color=color, label=label)
        axes[1].plot(times, positions[:, 1], color=color, label=label)
        offset = target.poses[:, :3, 3] - task_space.poses[:, :3, 3]
        variants.append(
            {
                "seed": seed,
                "accepted": projection.accepted,
                "max_translation_offset_m": float(offset.norm(dim=-1).max()),
                "max_ik_position_error_m": float(
                    projection.position_error.norm(dim=-1).max()
                ),
            }
        )

    axes[0].scatter(
        nominal[[0, -1], 0],
        nominal[[0, -1], 1],
        color="black",
        marker="o",
        s=42,
        zorder=5,
        label="locked endpoints",
    )
    axes[0].set_title("Task-space deformation: XY tool path")
    axes[0].set_xlabel("tool x (m)")
    axes[0].set_ylabel("tool y (m)")
    axes[0].axis("equal")
    axes[1].set_title("Trajectory IK result: projected y joint")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("q_y (m)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Pure trajectory augmentation · phase endpoints remain exact")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    if show:
        plt.show()
    plt.close(figure)
    return {
        "output": str(output_path),
        "frame_id": task_space.frame_id,
        "sample_count": source.positions.shape[0],
        "variants": variants,
    }


def _curved_segment(
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    bend: float,
    sample_count: int = 9,
) -> torch.Tensor:
    progress = torch.linspace(0, 1, sample_count, dtype=torch.float64)
    start_tensor = torch.tensor(start, dtype=torch.float64)
    goal_tensor = torch.tensor(goal, dtype=torch.float64)
    delta = goal_tensor - start_tensor
    normal = torch.stack((-delta[1], delta[0])) / torch.linalg.vector_norm(delta)
    baseline = torch.lerp(start_tensor, goal_tensor, progress[:, None])
    return baseline + bend * torch.sin(torch.pi * progress)[:, None] * normal


def _source_demo(
    source_id: str,
    *,
    pickup: tuple[float, float],
    place: tuple[float, float],
    home: tuple[float, float],
    bends: tuple[float, float, float],
) -> TrajectoryTemplate:
    paths = (
        _curved_segment((0.0, -0.25), pickup, bend=bends[0]),
        _curved_segment(pickup, place, bend=bends[1]),
        _curved_segment(place, home, bend=bends[2]),
    )
    positions = torch.cat(paths)
    dt = torch.full((positions.shape[0],), _DT, dtype=torch.float64)
    dt[0] = 0
    phase_length = paths[0].shape[0]
    phases = tuple(
        TrajectoryPhase(name, index * phase_length, (index + 1) * phase_length)
        for index, name in enumerate(("approach", "transport", "retreat"))
    )
    return TrajectoryTemplate(
        source_id=source_id,
        source_revision="one",
        template_id=f"{source_id}_trajectory",
        joint_names=("joint_0", "joint_1"),
        positions=positions,
        dt=dt,
        phases=phases,
        controlled_joint_indices=(0, 1),
        validator_id="demo_only",
    )


def _phase_map(
    segments: tuple[TrajectorySegment, ...],
) -> dict[str, TrajectorySegment]:
    return {segment.phase_id: segment for segment in segments}


def _render_stitching_demo(output_path: Path, *, show: bool) -> dict[str, object]:
    from matplotlib import pyplot as plt

    sources = (
        _source_demo(
            "demo_A",
            pickup=(1.0, 0.0),
            place=(2.0, 0.0),
            home=(2.65, 0.35),
            bends=(0.05, 0.10, -0.08),
        ),
        _source_demo(
            "demo_B",
            pickup=(1.0, 0.0),
            place=(2.0, 0.0),
            home=(2.55, 0.45),
            bends=(-0.10, 0.42, 0.10),
        ),
        _source_demo(
            "demo_C",
            pickup=(1.05, -0.05),
            place=(2.25, 0.15),
            home=(2.70, 0.68),
            bends=(0.15, -0.24, -0.12),
        ),
    )
    extracted = tuple(extract_trajectory_segments(source) for source in sources)
    reference = _phase_map(extracted[0])
    bank = TrajectorySegmentBank((*extracted[1], *extracted[2]))
    selected = (
        reference["approach"],
        bank.select(reference["transport"], strategy="fifo", ordinal=0),
        bank.select(reference["retreat"], strategy="fifo", ordinal=1),
    )
    bridge_planner = MinimumJerkBridgePlanner(
        control_dt=_DT,
        velocity_limits=torch.tensor([1.0, 1.0], dtype=torch.float64),
        acceleration_limits=torch.tensor([4.0, 4.0], dtype=torch.float64),
        minimum_sample_count=5,
    )
    stitched = stitch_trajectory_segments(
        selected,
        source_id="stitched_demo",
        source_revision="policy:one",
        template_id="approach_A_transport_B_retreat_C",
        bridge_planner=bridge_planner,
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for source in sources:
        positions = source.positions.numpy()
        color = _SOURCE_COLORS[source.source_id]
        axes[0].plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            alpha=0.8,
            linewidth=2,
            label=source.source_id,
        )
        boundary_indices = [phase.start_index for phase in source.phases]
        boundary_indices.append(source.phases[-1].stop_index - 1)
        axes[0].scatter(
            positions[boundary_indices, 0],
            positions[boundary_indices, 1],
            color=color,
            s=22,
        )
    axes[0].set_title("Segment bank: three demonstrations")
    axes[0].set_xlabel("joint 0")
    axes[0].set_ylabel("joint 1")
    axes[0].legend(loc="best")

    template = stitched.template
    positions = template.positions.numpy()
    placements = {value.output_phase_id: value for value in stitched.placements}
    for phase in template.phases:
        start = max(0, phase.start_index - 1)
        stop = phase.stop_index
        if phase.phase_id in stitched.bridge_phase_ids:
            color = "#E15759"
            label = "planned bridge"
            linestyle = "--"
        else:
            source_id = placements[phase.phase_id].source_id
            color = _SOURCE_COLORS[source_id]
            label = f"{phase.phase_id.split(':', 1)[1]} from {source_id}"
            linestyle = "-"
        axes[1].plot(
            positions[start:stop, 0],
            positions[start:stop, 1],
            color=color,
            linewidth=3,
            linestyle=linestyle,
            label=label,
        )
        axes[1].scatter(
            positions[phase.stop_index - 1, 0],
            positions[phase.stop_index - 1, 1],
            color=color,
            s=28,
            zorder=5,
        )
    axes[1].set_title("Stitched result: A → B → bridge → C")
    axes[1].set_xlabel("joint 0")
    axes[1].set_ylabel("joint 1")
    axes[1].legend(fontsize=8, loc="best")

    times = template.dt.cumsum(0).numpy()
    axes[2].plot(times, positions[:, 0], label="joint 0", color="#4C78A8")
    axes[2].plot(times, positions[:, 1], label="joint 1", color="#B279A2")
    for phase in template.phases:
        start_time = times[max(0, phase.start_index - 1)]
        stop_time = times[phase.stop_index - 1]
        if phase.phase_id in stitched.bridge_phase_ids:
            axes[2].axvspan(
                start_time,
                stop_time,
                color="#E15759",
                alpha=0.18,
                label="bridge interval",
            )
        axes[2].axvline(stop_time, color="black", linewidth=0.7, alpha=0.25)
    axes[2].set_title("Composite qpos timeline")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("joint position")
    axes[2].legend(fontsize=8, loc="best")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Phase-aligned multi-demo recombination with explicit bridging")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    if show:
        plt.show()
    plt.close(figure)

    return {
        "output": str(output_path),
        "bank_size": len(bank),
        "candidate_sources": {
            phase_id: [
                candidate.source_id
                for candidate in bank.candidates(reference[phase_id])
            ]
            for phase_id in ("transport", "retreat")
        },
        "selected_sources": [segment.source_id for segment in selected],
        "bridge_phase_ids": list(stitched.bridge_phase_ids),
        "sample_count": template.positions.shape[0],
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pure-trajectory augmentation without a simulator."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPOSITORY_ROOT / "output" / "pure_trajectory_augmentation",
        help="Directory for the two PNGs and summary JSON.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib figures in addition to saving them.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate both pure-trajectory augmentation visualizations.

    Args:
        argv: Optional command-line arguments for programmatic invocation.
    """
    args = _create_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_space_path = args.output_dir / "task_space_deformation.png"
    stitching_path = args.output_dir / "segment_stitching.png"
    summary_path = args.output_dir / "summary.json"
    summary = {
        "task_space_deformation": _render_task_space_demo(
            task_space_path, show=args.show
        ),
        "segment_stitching": _render_stitching_demo(stitching_path, show=args.show),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"task-space figure: {task_space_path}")
    print(f"segment-stitching figure: {stitching_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
