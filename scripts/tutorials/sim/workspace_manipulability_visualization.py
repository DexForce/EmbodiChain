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

"""Render a workspace colored by manipulability, plus a selected-point detail view.

The script runs :class:`~embodichain.lab.sim.motion.workspace.analyzer.WorkspaceAnalyzer`
with the manipulability metric enabled and writes three PNGs through
:class:`~embodichain.lab.sim.motion.workspace.visualizers.manipulability_visualizer.ManipulabilityVisualizer`:

1. ``workspace_manipulability_cartesian.png`` — every Cartesian sample, reachable
   points colored by the Yoshikawa index ``w`` with a visible color bar, and
   unreachable samples kept visually distinct (small grey crosses). The raw
   ``w`` range is annotated separately from the percentile-clipped color-bar
   span, so the clipping never hides the true magnitudes.
2. ``workspace_manipulability_joint_space.png`` — the joint-space path, where
   scores align one-to-one with ``workspace_points``, rendered on a log color
   scale because manipulability spans orders of magnitude.
3. ``workspace_manipulability_inspection.png`` — the best and worst conditioned
   points from figure 1 with their ``w``, Jacobian condition number, joint
   configuration and *translational* manipulability ellipsoid. Ellipsoids are
   computed only for these few points, never for the whole cloud.

The robot is driven through its kinematic chain and solver only, so the script
is fully headless: no renderer, physics backend or display is required.

Run:
    python scripts/tutorials/sim/workspace_manipulability_visualization.py
    python scripts/tutorials/sim/workspace_manipulability_visualization.py \
        --robot ur5 --num_samples 4000 --output_dir /tmp/figs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import matplotlib

# Force a non-interactive backend before pyplot is imported anywhere else.
matplotlib.use("Agg")

import numpy as np
import torch
import warp as wp

from embodichain.lab.sim.motion.workspace.analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.configs import (
    CacheConfig,
    MetricConfig,
    MetricType,
    SamplingConfig,
    VisualizationConfig,
)
from embodichain.lab.sim.motion.workspace.visualizers import (
    ManipulabilityColorCfg,
    ManipulabilityVisualizer,
    align_manipulability_scores,
    inspect_points,
    select_inspection_indices,
)
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.robots import CobotMagicCfg, URRobotCfg
from embodichain.utils import logger
from embodichain.utils.utility import reset_all_seeds

DEFAULT_OUTPUT_DIR = Path("docs/source/_static/tutorials")

ROBOT_PRESETS = {
    "ur5": ("arm", 6),
    "cobotmagic": ("left_arm", 6),
}


def build_kinematic_robot(kind: str, device: torch.device) -> tuple[Any, str]:
    """Build a solver-backed robot adapter that needs no simulation backend.

    ``WorkspaceAnalyzer`` only uses forward/inverse kinematics, joint limits and
    the solver Jacobian, so binding the real ``Robot`` batch methods onto an
    asset-backed kinematic chain is enough to run the full analysis headlessly.
    This mirrors ``scripts/benchmark/workspace_analyzer/benchmark_robot_workspace.py``.

    Args:
        kind: Robot preset key, one of :data:`ROBOT_PRESETS`.
        device: Torch device used for sampling and kinematics.

    Returns:
        Tuple of the robot adapter and the control part name to analyze.

    Raises:
        ValueError: If ``kind`` is not a known preset.
    """
    if kind not in ROBOT_PRESETS:
        raise ValueError(
            f"unknown robot preset {kind!r}; choose from {sorted(ROBOT_PRESETS)}."
        )
    # Batched IK runs Warp kernels; without a SimulationManager nothing else
    # initializes the Warp runtime.
    wp.init()
    part, num_joints = ROBOT_PRESETS[kind]
    preset = (
        URRobotCfg.from_dict({"robot_type": "ur5"})
        if kind == "ur5"
        else CobotMagicCfg.from_dict({})
    )
    chain = preset.build_pk_serial_chain(device)[part]
    solver_cfg = preset.solver_cfg[part]
    solver_cfg.joint_names = chain.get_joint_parameter_names()
    solver = solver_cfg.init_solver(device=device, pk_serial_chain=chain)
    solver.compiled_fk = chain.forward_kinematics_tensor

    base = torch.eye(4, device=device).unsqueeze(0)
    joint_limits = torch.stack(
        (solver.lower_qpos_limits, solver.upper_qpos_limits), dim=-1
    )[None]
    robot = SimpleNamespace(
        device=device,
        cfg=preset,
        num_envs=1,
        _all_indices=[0],
        _solvers={part: solver},
        control_parts={part: solver_cfg.joint_names},
        joint_names=solver_cfg.joint_names,
        body_data=SimpleNamespace(qpos_limits=joint_limits),
        get_joint_ids=lambda *args, **kwargs: list(range(num_joints)),
        get_qpos=lambda: solver.get_default_qpos_seed()[None],
        get_link_pose=lambda **kwargs: base,
    )
    for name in ("compute_fk", "compute_batch_fk", "compute_batch_ik", "get_solver"):
        setattr(robot, name, MethodType(getattr(Robot, name), robot))
    return robot, part


def analyzer_config(
    mode: AnalysisMode,
    num_samples: int,
    seed: int,
    control_part: str,
    ik_seeds: int = 4,
) -> WorkspaceAnalyzerConfig:
    """Build an analyzer configuration with the manipulability metric enabled."""
    return WorkspaceAnalyzerConfig(
        mode=mode,
        sampling=SamplingConfig(num_samples=num_samples, seed=seed, batch_size=2048),
        cache=CacheConfig(enabled=False),
        metric=MetricConfig(enabled_metrics=[MetricType.MANIPULABILITY]),
        visualization=VisualizationConfig(enabled=False),
        # Keep ``all_points`` and ``reachability_mask`` so unreachable samples
        # can be drawn next to the scored ones.
        retain_diagnostics=True,
        ik_samples_per_point=ik_seeds,
        control_part_name=control_part,
    )


def render_workspace(
    results: dict[str, Any],
    output_path: Path,
    *,
    title: str,
    log_scale: bool,
    highlight: np.ndarray | None = None,
) -> tuple[Path, Any]:
    """Render one manipulability-colored workspace figure.

    Args:
        results: Analysis dictionary from ``WorkspaceAnalyzer.analyze``.
        output_path: Destination PNG path.
        title: Figure title.
        log_scale: Use a logarithmic color scale.
        highlight: Optional point indices to circle.

    Returns:
        Tuple of the written path and the aligned point set.
    """
    point_set = align_manipulability_scores(results)
    color_cfg = ManipulabilityColorCfg(
        log_scale=log_scale,
        percentile_clip=(2.0, 98.0),
        point_size=9.0,
        unreachable_point_size=6.0,
    )
    visualizer = ManipulabilityVisualizer(backend="matplotlib", color_cfg=color_cfg)
    visualizer.visualize(
        point_set.points,
        point_set=point_set,
        title=title,
        highlight=highlight,
    )
    visualizer.save(output_path)
    mapping = visualizer.map_colors(point_set.scores, point_set.reachable_mask)
    logger.log_info(
        f"{output_path.name}: {point_set.num_reachable} reachable / "
        f"{point_set.num_unreachable} unreachable, raw w range "
        f"[{mapping.raw_range[0]:.4g}, {mapping.raw_range[1]:.4g}], "
        f"color bar span [{mapping.clip_range[0]:.4g}, {mapping.clip_range[1]:.4g}]"
    )
    return output_path, point_set


def render_inspection(
    point_set: Any,
    robot: Any,
    control_part: str,
    output_path: Path,
    *,
    top_k: int,
    bottom_k: int,
) -> tuple[Path, np.ndarray]:
    """Render the detail view for the best and worst conditioned points.

    Args:
        point_set: Aligned point set from :func:`align_manipulability_scores`.
        robot: Robot adapter providing the solver Jacobian.
        control_part: Control part whose solver is used.
        output_path: Destination PNG path.
        top_k: Number of highest-``w`` points to inspect.
        bottom_k: Number of lowest-``w`` points to inspect.

    Returns:
        Tuple of the written path and the inspected point indices.
    """
    solver = robot.get_solver(control_part)
    jacobian_calls: list[int] = []

    def jacobian_fn(qpos: np.ndarray) -> torch.Tensor:
        jacobian_calls.append(len(qpos))
        batch = torch.as_tensor(qpos, dtype=torch.float32, device=solver.device)
        with torch.no_grad():
            return solver.get_jacobian(batch)

    selection = select_inspection_indices(
        point_set.scores, top_k=top_k, bottom_k=bottom_k
    )
    inspections = inspect_points(
        selection,
        points=point_set.points,
        scores=point_set.scores,
        jacobian_fn=jacobian_fn,
        joint_configurations=point_set.joint_configurations,
        score_indices=point_set.score_indices,
    )
    visualizer = ManipulabilityVisualizer(backend="matplotlib")
    visualizer.visualize_inspection(
        inspections,
        title="Translational manipulability ellipsoids at selected points",
    )
    visualizer.save(output_path)
    logger.log_info(
        f"{output_path.name}: {sum(jacobian_calls)} Jacobians computed for "
        f"{len(inspections)} inspected points out of {len(point_set.points)} drawn."
    )
    for inspection in inspections:
        logger.log_info(" | ".join(inspection.summary_lines()))
    return output_path, selection.indices


def main() -> None:
    """Parse arguments, run the analyses and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robot",
        default="ur5",
        choices=sorted(ROBOT_PRESETS),
        help="Robot preset to analyze.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4000,
        help="Samples per analysis mode.",
    )
    parser.add_argument(
        "--ik_seeds",
        type=int,
        default=4,
        help="Random IK seeds per Cartesian sample; more seeds raise reachability.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives the workspace_manipulability_*.png files.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for kinematics.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Highest-w points to inspect."
    )
    parser.add_argument(
        "--bottom_k", type=int, default=1, help="Lowest-w points to inspect."
    )
    args = parser.parse_args()

    reset_all_seeds(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    robot, control_part = build_kinematic_robot(args.robot, device)

    # --- Cartesian mode: reachable and unreachable samples side by side ------
    cartesian = WorkspaceAnalyzer(
        robot,
        analyzer_config(
            AnalysisMode.CARTESIAN_SPACE,
            args.num_samples,
            args.seed,
            control_part,
            ik_seeds=args.ik_seeds,
        ),
    )
    cartesian_results = cartesian.analyze(visualize=False)

    cartesian_path, cartesian_points = render_workspace(
        cartesian_results,
        output_dir / "workspace_manipulability_cartesian.png",
        title=f"{args.robot}: Cartesian workspace colored by manipulability",
        log_scale=False,
    )

    # --- Joint-space mode: scores align 1:1 with workspace_points ------------
    joint_space = WorkspaceAnalyzer(
        robot,
        analyzer_config(
            AnalysisMode.JOINT_SPACE, args.num_samples, args.seed, control_part
        ),
    )
    joint_results = joint_space.analyze(visualize=False)
    joint_path, _ = render_workspace(
        joint_results,
        output_dir / "workspace_manipulability_joint_space.png",
        title=f"{args.robot}: joint-space workspace, log manipulability scale",
        log_scale=True,
    )

    # --- Detail view for a handful of points only ----------------------------
    inspection_path, highlighted = render_inspection(
        cartesian_points,
        robot,
        control_part,
        output_dir / "workspace_manipulability_inspection.png",
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )

    # Redraw figure 1 with the inspected points circled so the two figures read
    # together.
    render_workspace(
        cartesian_results,
        cartesian_path,
        title=f"{args.robot}: Cartesian workspace colored by manipulability",
        log_scale=False,
        highlight=highlighted,
    )

    metrics = cartesian_results.get("metrics", {}).get("manipulability", {})
    if metrics:
        logger.log_info(f"Cartesian manipulability aggregates: {metrics}")
    for path in (cartesian_path, joint_path, inspection_path):
        logger.log_info(f"Wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
