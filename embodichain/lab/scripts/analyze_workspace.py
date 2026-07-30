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

"""Analyze a robot's reachable workspace from a URDF/USD asset or a preset.

Usage examples::

    # Predefined EmbodiChain robot -- control parts + solver come from the
    # preset, so only --control-part (optional) is needed.
    embodichain analyze-workspace \\
        --robot franka_panda --mode joint_space --num-samples 20000

    embodichain analyze-workspace \\
        --robot cobotmagic --control-part left_arm --mode cartesian_space \\
        --bounds -0.5 0.5 -0.5 0.5 0.6 1.5 --ik-samples-per-point 5

    embodichain analyze-workspace \\
        --robot dexforce_w1 \\
        --robot-params '{"version":"v021","arm_kind":"industrial"}' \\
        --control-part left_arm --mode joint_space

    # Generic URDF/USD asset -- requires --ee-link (and --joints for grippers)
    embodichain analyze-workspace \\
        --asset /path/to/panda.urdf \\
        --ee-link fr3_hand_tcp --joints "fr3_joint[1-7]" \\
        --mode joint_space --num-samples 20000

    # USD asset (needs a companion URDF for the kinematics solver)
    embodichain analyze-workspace \\
        --asset /path/to/robot.usd --urdf /path/to/robot.urdf \\
        --ee-link tcp_link --joints "joint_[1-6]"

    # Preview an already-computed workspace cache (no robot/analysis needed)
    embodichain analyze-workspace --preview-cache ~/.cache/embodichain_data/robot_workspace/<key>

Results are cached to disk (``--cache-dir``, default
``~/.cache/embodichain_data/robot_workspace``) keyed by the robot + parameters,
so repeated runs and other applications can reuse the reachable workspace
without recomputing. Pass ``--output`` to export a copy to a user path.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from embodichain.utils.logger import log_info, log_warning, log_error

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
        WorkspaceAnalyzerConfig,
    )

__all__ = [
    "build_sim_cfg",
    "build_robot_cfg",
    "build_preset_robot_cfg",
    "build_analyzer_config",
    "preview_cache",
    "parse_args",
    "main",
    "cli",
]


# Mapping from ``--solver`` choice to solver config class (resolved lazily).
_SOLVER_CHOICES = ("pytorch", "pinocchio", "pink")

# Predefined EmbodiChain robots selectable via ``--robot``.
_ROBOT_CHOICES = ("franka_panda", "cobotmagic", "dexforce_w1", "ur")


def _robot_cfg_cls(name: str):
    """Resolve a ``--robot`` choice to its RobotCfg subclass (lazy import).

    Args:
        name: One of :data:`_ROBOT_CHOICES`.

    Returns:
        The RobotCfg subclass for the preset.
    """
    from embodichain.lab.sim.robots import (
        CobotMagicCfg,
        DexforceW1Cfg,
        FrankaPandaCfg,
        URRobotCfg,
    )

    return {
        "franka_panda": FrankaPandaCfg,
        "cobotmagic": CobotMagicCfg,
        "dexforce_w1": DexforceW1Cfg,
        "ur": URRobotCfg,
    }[name]


def _resolve_control_part(robot, requested: str | None) -> str | None:
    """Resolve the control part name, auto-selecting when none is requested.

    Mirrors ``WorkspaceAnalyzer.DEFAULT_CONTROL_PART_PRIORITY`` so the CLI and
    the analyzer agree on the active part.

    Args:
        robot: Loaded Robot instance.
        requested: User-specified control part name, or None.

    Returns:
        The resolved control part name, or None if the robot has no control
        parts.
    """
    if requested is not None:
        return requested
    parts = robot.control_parts
    if not parts:
        return None
    for priority in ("left_arm", "right_arm"):
        if priority in parts:
            return priority
    return next(iter(parts))


def _parse_joints(joints: str | None) -> list[str]:
    """Parse the ``--joints`` argument into a list of joint names/regexes.

    Args:
        joints: A comma-separated list of joint names, a single regex, or None
            (defaults to all joints).

    Returns:
        List of joint name specifiers (supports regex expansion at robot init).
    """
    if joints is None:
        return [".*"]
    parts = [j.strip() for j in joints.split(",") if j.strip()]
    return parts if parts else [".*"]


def _build_tcp(tcp_args: Sequence[float] | None) -> np.ndarray:
    """Build a 4x4 TCP matrix from ``[tx, ty, tz, rx, ry, rz]`` (deg).

    Args:
        tcp_args: Translation (m) followed by rotation (xyz euler, degrees),
            or None for identity.

    Returns:
        4x4 homogeneous transform as a numpy array.
    """
    if tcp_args is None:
        return np.eye(4)
    from scipy.spatial.transform import Rotation as R

    tx, ty, tz, rx, ry, rz = (float(v) for v in tcp_args)
    matrix = np.eye(4)
    matrix[:3, 3] = [tx, ty, tz]
    matrix[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    return matrix


def build_sim_cfg(args: argparse.Namespace):
    """Build a SimulationManagerCfg from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        SimulationManagerCfg: Simulation configuration.
    """
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

    return SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.sim_device,
        width=args.width,
        height=args.height,
        render_cfg=RenderCfg(renderer=args.renderer),
    )


def build_robot_cfg(
    args: argparse.Namespace,
) -> tuple[RobotCfg, str | None, str | None]:
    """Build a RobotCfg from either a URDF/USD asset or a predefined robot.

    - ``--robot NAME``: use a built-in robot preset. The control parts and
      kinematics solver come from the preset, so ``--ee-link`` and ``--joints``
      are not required -- only ``--control-part`` (optional, auto-selected).
    - ``--asset PATH``: build a generic RobotCfg from a URDF/USD asset; requires
      ``--ee-link`` (and ``--urdf`` for USD assets).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (RobotCfg, control_part_name_or_None, solver_urdf_path_or_None).

    Raises:
        ValueError: For invalid configurations (e.g. USD without --urdf, missing
            --ee-link for --asset, or an unknown preset control part).
    """
    if args.robot:
        return build_preset_robot_cfg(args)
    return _build_asset_robot_cfg(args)


def build_preset_robot_cfg(
    args: argparse.Namespace,
) -> tuple[RobotCfg, str | None, str | None]:
    """Build a RobotCfg from a predefined EmbodiChain robot preset.

    Uses the robot's built-in ``control_parts`` and ``solver_cfg``, so the
    end-effector link and joint names come from the preset. Pass
    ``--robot-params`` (JSON) for variant overrides such as
    ``{"robot_type": "ur5"}`` or ``{"version": "v021", "arm_kind": "industrial"}``.

    Args:
        args: Parsed CLI arguments with ``args.robot`` and optionally
            ``args.robot_params`` and ``args.control_part``.

    Returns:
        Tuple of (RobotCfg, control_part_name_or_None, None).

    Raises:
        ValueError: If ``--control-part`` is not one of the preset's parts.
    """
    import json

    cfg_cls = _robot_cfg_cls(args.robot)
    params = json.loads(args.robot_params) if args.robot_params else {}
    cfg = cfg_cls.from_dict(params)
    if args.uid:
        cfg.uid = args.uid

    control_part = args.control_part
    if (
        control_part is not None
        and cfg.control_parts
        and control_part not in cfg.control_parts
    ):
        raise ValueError(
            f"Control part {control_part!r} is not a control part of robot "
            f"{args.robot!r}. Available: {list(cfg.control_parts)}."
        )
    return cfg, control_part, None


def _build_asset_robot_cfg(
    args: argparse.Namespace,
) -> tuple[RobotCfg, str, str]:
    """Build a RobotCfg from a URDF/USD asset and CLI arguments.

    Args:
        args: Parsed CLI arguments. Must include ``--asset`` (URDF or USD) and
            ``--ee-link``. USD assets additionally require ``--urdf`` because
            the kinematics solver (pytorch-kinematics) needs a URDF.

    Returns:
        Tuple of (RobotCfg, control_part_name, solver_urdf_path).

    Raises:
        ValueError: If ``--ee-link`` is missing, or a USD/non-URDF asset is
            given without ``--urdf``.
    """
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.solvers import (
        PinkSolverCfg,
        PinocchioSolverCfg,
        PytorchSolverCfg,
    )

    if not args.ee_link:
        raise ValueError(
            "--ee-link is required when using --asset (the end-effector link "
            "for FK/IK). For predefined robots, use --robot instead."
        )

    asset = os.path.abspath(args.asset)
    is_urdf = asset.lower().endswith(".urdf")
    if is_urdf:
        solver_urdf = asset
    else:
        if not args.urdf:
            raise ValueError(
                "USD/non-URDF assets require --urdf <path> for the kinematics "
                "solver (FK/IK). Provide the URDF that matches the asset."
            )
        solver_urdf = os.path.abspath(args.urdf)

    solver_cls = {
        "pytorch": PytorchSolverCfg,
        "pinocchio": PinocchioSolverCfg,
        "pink": PinkSolverCfg,
    }[args.solver]

    tcp = _build_tcp(args.tcp)
    control_part = args.control_part or "arm"
    joints = _parse_joints(args.joints)

    solver_cfg = solver_cls(
        end_link_name=args.ee_link,
        root_link_name=args.root_link,
        tcp=tcp,
    )
    solver_cfg.urdf_path = solver_urdf

    cfg = RobotCfg()
    cfg.uid = args.uid or os.path.splitext(os.path.basename(asset))[0]
    cfg.fpath = asset
    cfg.init_pos = tuple(args.init_pos)
    cfg.init_rot = tuple(args.init_rot)
    cfg.fix_base = args.fix_base
    cfg.use_usd_properties = args.use_usd_properties
    cfg.control_parts = {control_part: joints}
    cfg.solver_cfg = {control_part: solver_cfg}
    return cfg, control_part, solver_urdf


def build_analyzer_config(
    args: argparse.Namespace, control_part_name: str
) -> WorkspaceAnalyzerConfig:
    """Build a WorkspaceAnalyzerConfig from CLI arguments.

    Args:
        args: Parsed CLI arguments.
        control_part_name: Control part name shared with the robot config.

    Returns:
        WorkspaceAnalyzerConfig: Analyzer configuration.
    """
    import torch

    from embodichain.lab.sim.utility.workspace_analyzer.configs import (
        CacheConfig,
        DimensionConstraint,
        SamplingConfig,
        SamplingStrategy,
        VisualizationConfig,
        VisualizationType,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
        AnalysisMode,
        WorkspaceAnalyzerConfig,
    )

    mode = AnalysisMode(args.mode)
    visualize = args.visualize and not args.headless

    sampling = SamplingConfig(
        strategy=SamplingStrategy(args.sampler),
        num_samples=args.num_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    visualization = VisualizationConfig(
        enabled=visualize,
        vis_type=VisualizationType(args.vis_type),
        point_size=args.point_size,
        voxel_size=args.voxel_size,
        show_unreachable_points=not args.hide_unreachable,
    )
    cache = CacheConfig(
        enabled=not args.no_cache,
        cache_dir=args.cache_dir,
        compression=True,
    )

    if args.bounds:
        b = args.bounds
        constraint = DimensionConstraint(
            min_bounds=[b[0], b[2], b[4]],
            max_bounds=[b[1], b[3], b[5]],
        )
    else:
        constraint = DimensionConstraint()
    constraint.joint_limits_scale = args.joint_limits_scale

    config = WorkspaceAnalyzerConfig(
        mode=mode,
        sampling=sampling,
        visualization=visualization,
        cache=cache,
        constraint=constraint,
        ik_samples_per_point=args.ik_samples_per_point,
        control_part_name=control_part_name,
    )

    if mode == AnalysisMode.PLANE_SAMPLING:
        config.plane_normal = torch.tensor(args.plane_normal, dtype=torch.float32)
        config.plane_point = torch.tensor(args.plane_point, dtype=torch.float32)
        if args.plane_bounds:
            config.plane_bounds = torch.tensor(
                args.plane_bounds, dtype=torch.float32
            ).reshape(2, 2)

    return config


def _print_summary(results: dict, analyzer) -> None:
    """Log a human-readable summary of the analysis results.

    Args:
        results: Results dict from ``WorkspaceAnalyzer.analyze``.
        analyzer: The WorkspaceAnalyzer instance (for cache path lookup).
    """
    mode = results.get("mode", "unknown")
    num_samples = results.get("num_samples", 0)
    analysis_time = results.get("analysis_time", 0.0)

    if mode == "joint_space":
        num_valid = results.get("num_valid", 0)
        log_info(
            f"Mode: {mode} | valid points: {num_valid}/{num_samples} "
            f"| time: {analysis_time:.2f}s",
            color="green",
        )
    else:
        num_reachable = results.get("num_reachable", 0)
        log_info(
            f"Mode: {mode} | reachable: {num_reachable}/{num_samples} "
            f"| time: {analysis_time:.2f}s",
            color="green",
        )

    metrics = results.get("metrics") or {}
    bbox = metrics.get("bounding_box")
    if bbox:
        log_info(
            f"Bounding box: min={bbox.get('min')} max={bbox.get('max')} "
            f"volume={metrics.get('bounding_box_volume', 0.0):.4f} m^3"
        )

    cache_path = analyzer.get_results_cache_path()
    if cache_path is not None:
        log_info(
            f"Results cached at: {cache_path} "
            "(load results.npz / meta.json from other apps)",
            color="green",
        )
    else:
        log_info("Results cache disabled (use --cache-dir to enable).")


def _load_preview_data(path: str, cache_dir: str) -> tuple[dict, str | None]:
    """Load workspace arrays and mode from a cached result.

    ``path`` may be:

    - a cache entry directory (containing ``results.npz`` + ``meta.json``),
    - a ``results.npz`` file directly, or
    - a bare cache key, looked up under ``cache_dir``.

    Args:
        path: Path to a cache entry directory, an ``.npz`` file, or a cache key.
        cache_dir: Cache root used to resolve a bare key.

    Returns:
        Tuple of (arrays dict of numpy arrays, mode string or None).

    Raises:
        FileNotFoundError: If no ``results.npz`` can be resolved for ``path``.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        npz_path = p / "results.npz"
    elif p.is_file():
        npz_path = p
    else:
        # Treat as a cache key under the cache root.
        npz_path = Path(cache_dir) / path / "results.npz"

    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Preview cache not found for {path!r}: expected results.npz at "
            f"{npz_path}. Pass a cache entry directory, an .npz file, or a key "
            f"under --cache-dir."
        )

    meta_path = npz_path.parent / "meta.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}

    with np.load(str(npz_path)) as npz:
        arrays = {name: npz[name] for name in npz.files}

    mode = meta.get("mode")
    if mode is None:
        mode = "cartesian_space" if "reachable_points" in arrays else "joint_space"
    return arrays, mode


def _preview_points_and_colors(
    arrays: dict, mode: str | None, hide_unreachable: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Select preview points and color them by reachability.

    Args:
        arrays: Arrays dict from :func:`_load_preview_data`.
        mode: Analysis mode (``joint_space`` / ``cartesian_space`` /
            ``plane_sampling``) or None.
        hide_unreachable: If True in IK modes, show only reachable points.

    Returns:
        Tuple of (points (N, 3), colors (N, 3) RGB in [0, 1]). Reachable points
        are green, unreachable red (IK modes only); joint-space points are all
        green.
    """
    is_ik = mode in ("cartesian_space", "plane_sampling")

    if (
        is_ik
        and hide_unreachable
        and "reachable_points" in arrays
        and len(arrays["reachable_points"]) > 0
    ):
        points = np.asarray(arrays["reachable_points"])
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1.0  # green
        return points, colors

    if "workspace_points" in arrays:
        points = np.asarray(arrays["workspace_points"])
    elif "all_points" in arrays:
        points = np.asarray(arrays["all_points"])
    else:
        points = np.asarray(arrays[next(iter(arrays))])

    colors = np.zeros((len(points), 3))
    mask = arrays.get("reachability_mask") if is_ik else None
    if mask is not None and len(mask) == len(points):
        mask_bool = np.asarray(mask).astype(bool)
        colors[mask_bool, 1] = 1.0  # green reachable
        colors[~mask_bool, 0] = 1.0  # red unreachable
    else:
        colors[:, 1] = 1.0  # all green
    return points, colors


def preview_cache(args: argparse.Namespace) -> None:
    """Visualize an already-computed workspace cache without recomputing.

    Loads the cached results from ``--preview-cache`` and renders them with an
    Open3D window. No robot or simulation is required.

    Args:
        args: Parsed CLI arguments. Uses ``args.preview_cache``, ``args.cache_dir``,
            ``args.vis_type``, ``args.point_size``, ``args.voxel_size`` and
            ``args.hide_unreachable``.
    """
    from embodichain.lab.sim.utility.workspace_analyzer.caches.results_cache import (
        DEFAULT_RESULTS_CACHE_DIR,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.configs import (
        VisualizationType,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.visualizers import (
        VisualizerFactory,
    )

    cache_dir = args.cache_dir or DEFAULT_RESULTS_CACHE_DIR
    arrays, mode = _load_preview_data(args.preview_cache, cache_dir)
    points, colors = _preview_points_and_colors(arrays, mode, args.hide_unreachable)
    log_info(
        f"Previewing workspace cache: {args.preview_cache} | mode={mode} | "
        f"{len(points)} points",
        color="green",
    )

    viz_type = VisualizationType(args.vis_type)
    factory = VisualizerFactory()
    kwargs: dict = {"backend": "open3d"}
    if viz_type == VisualizationType.POINT_CLOUD:
        kwargs["point_size"] = args.point_size
    elif viz_type == VisualizationType.VOXEL:
        kwargs["voxel_size"] = args.voxel_size
    visualizer = factory.create_visualizer(viz_type=viz_type, **kwargs)
    visualizer.visualize(points, colors=colors)
    log_info("Preview window open. Close the window to exit.", color="green")
    visualizer.show()


def main(args: argparse.Namespace) -> None:
    """Run the workspace analysis end-to-end.

    Loads the robot, runs analysis (with caching), prints a summary, optionally
    exports results, and keeps the visualization window open until Ctrl+C.
    With ``--preview-cache``, instead loads and visualizes an already-computed
    cache without running any analysis.

    Args:
        args: Parsed CLI arguments.
    """
    if args.preview_cache:
        preview_cache(args)
        return

    import torch

    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
        WorkspaceAnalyzer,
    )

    robot_cfg, control_part, _solver_urdf = build_robot_cfg(args)
    source = args.robot or f"asset {args.asset}"
    log_info(f"Loading robot ({source}) ...", color="green")

    sim = SimulationManager(build_sim_cfg(args))
    try:
        robot = sim.add_robot(cfg=robot_cfg)
        if robot is None:
            log_error("Failed to load robot into the simulation.")
            return
        control_part = _resolve_control_part(robot, control_part)
        joints_desc = (
            robot.control_parts.get(control_part) if control_part else "all joints"
        )
        log_info(
            f"Robot '{robot.cfg.uid}' loaded | control part: {control_part} "
            f"| joints: {joints_desc}",
            color="green",
        )

        if args.init_qpos:
            qpos = torch.tensor(args.init_qpos, dtype=torch.float32)
            joint_ids = robot.get_joint_ids(control_part)
            if len(qpos) != len(joint_ids):
                log_warning(
                    f"--init-qpos has {len(qpos)} values but control part "
                    f"'{control_part}' has {len(joint_ids)} joints; skipping."
                )
            else:
                robot.set_qpos(qpos=qpos, joint_ids=joint_ids)

        analyzer_cfg = build_analyzer_config(args, control_part)
        analyzer = WorkspaceAnalyzer(robot=robot, config=analyzer_cfg, sim_manager=sim)

        visualize = args.visualize and not args.headless
        results = analyzer.analyze(
            num_samples=args.num_samples,
            force_recompute=args.force_recompute,
            visualize=visualize,
        )

        _print_summary(results, analyzer)

        if args.output:
            log_info(f"Exporting results to {args.output} ...", color="green")
            analyzer.export_results(args.output, format=args.export_format)

        if visualize:
            log_info(
                "Workspace visualization window open. Press Ctrl+C to exit.",
                color="green",
            )
            try:
                while True:
                    sim.update(step=1)
            except KeyboardInterrupt:
                pass
    finally:
        try:
            sim.destroy()
        except Exception as e:  # noqa: BLE001
            log_warning(f"Failed to destroy simulation: {e}")
        try:
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()
        except Exception as e:  # noqa: BLE001
            log_warning(f"Failed to flush simulation cleanup queue: {e}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse workspace-analysis CLI arguments.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.

    Returns:
        The parsed argument namespace, with post-parse defaults (cache dir,
        visualize intent) applied.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain analyze-workspace",
        description=(
            "Analyze a robot's reachable workspace from a URDF/USD asset. "
            "Supports joint-space, Cartesian-space and plane-sampling modes; "
            "caches results to disk for reuse; visualizes in the sim window."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Source (choose one) ------------------------------------------------
    robot = parser.add_argument_group("Source (choose one)")
    source = robot.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Path to a robot asset (.urdf/.usd/.usda/.usdc). Builds a generic "
        "RobotCfg; requires --ee-link (and --urdf for USD assets).",
    )
    source.add_argument(
        "--robot",
        type=str,
        choices=_ROBOT_CHOICES,
        default=None,
        help="Use a predefined EmbodiChain robot. The control parts and "
        "kinematics solver come from the preset, so --ee-link/--joints are not "
        "needed -- only --control-part (optional). Choices: franka_panda, "
        "cobotmagic, dexforce_w1, ur.",
    )
    source.add_argument(
        "--preview-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Preview an already-computed workspace cache without recomputing. "
        "PATH is a cache entry directory, a results.npz file, or a cache key "
        "(looked up under --cache-dir). Opens an Open3D window; no "
        "--robot/--asset needed.",
    )
    robot.add_argument(
        "--robot-params",
        type=str,
        default=None,
        help="JSON dict of variant overrides for --robot, e.g. "
        '{"robot_type":"ur5"} or {"version":"v021","arm_kind":"industrial"}.',
    )
    robot.add_argument(
        "--control-part",
        type=str,
        default=None,
        help="Control part to analyze. For --robot: defaults to left_arm/right_arm "
        "or the first available part. For --asset: defaults to 'arm'.",
    )
    robot.add_argument("--uid", type=str, default=None, help="Robot UID in the scene.")
    robot.add_argument(
        "--init-qpos",
        type=float,
        nargs="+",
        default=None,
        help="Initial joint positions for the control part (also used as the IK "
        "reference pose).",
    )

    # --- Robot asset options (apply to --asset only) ------------------------
    asset_opts = parser.add_argument_group(
        "Robot asset options (used with --asset; ignored for --robot)"
    )
    asset_opts.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="URDF for the kinematics solver. Required for USD assets; defaults "
        "to --asset when it is a URDF.",
    )
    asset_opts.add_argument(
        "--ee-link",
        type=str,
        default=None,
        help="End-effector link name (FK/IK target). Required with --asset. Must "
        "match a link in the URDF.",
    )
    asset_opts.add_argument(
        "--joints",
        type=str,
        default=None,
        help="Comma-separated joint names or a regex for the control part "
        "(e.g. 'fr3_joint[1-7]' or 'joint1,joint2'). Default: all joints ('.*'). "
        "For arms with a gripper, specify the arm joints to avoid dof mismatch.",
    )
    asset_opts.add_argument(
        "--root-link",
        type=str,
        default=None,
        help="Root/base link name for the solver. If omitted, the URDF root is used.",
    )
    asset_opts.add_argument(
        "--solver",
        type=str,
        choices=_SOLVER_CHOICES,
        default="pytorch",
        help="Kinematics solver (default: pytorch, works for any URDF).",
    )
    asset_opts.add_argument(
        "--tcp",
        type=float,
        nargs=6,
        default=None,
        metavar=("TX", "TY", "TZ", "RX", "RY", "RZ"),
        help="Tool center point: translation (m) + rotation (xyz euler, deg).",
    )
    asset_opts.add_argument(
        "--init-pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Robot base initial position (default: 0 0 0).",
    )
    asset_opts.add_argument(
        "--init-rot",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Robot base initial rotation in degrees (default: 0 0 0).",
    )
    asset_opts.add_argument(
        "--fix-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix the robot base (default: fixed).",
    )
    asset_opts.add_argument(
        "--use-usd-properties",
        action="store_true",
        default=False,
        help="Use physical properties from the USD file (USD assets only).",
    )

    # --- Analysis -----------------------------------------------------------
    analysis = parser.add_argument_group("Analysis")
    analysis.add_argument(
        "--mode",
        type=str,
        choices=["joint_space", "cartesian_space", "plane_sampling"],
        default="joint_space",
        help="Workspace analysis mode (default: joint_space).",
    )
    analysis.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples (default: 10000).",
    )
    analysis.add_argument(
        "--ik-samples-per-point",
        type=int,
        default=1,
        help="Cartesian/plane mode: random IK seeds per point (default: 1).",
    )
    analysis.add_argument(
        "--joint-limits-scale",
        type=float,
        default=1.0,
        help="Scale joint limits to a fraction of full range (default: 1.0).",
    )

    # --- Sampling -----------------------------------------------------------
    sampling = parser.add_argument_group("Sampling")
    sampling.add_argument(
        "--sampler",
        type=str,
        choices=["random", "sobol", "halton", "lhs", "uniform", "gaussian"],
        default="random",
        help="Sampling strategy (default: random).",
    )
    sampling.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    sampling.add_argument(
        "--batch-size", type=int, default=1000, help="FK/IK batch size (default: 1000)."
    )

    # --- Workspace / plane --------------------------------------------------
    space = parser.add_argument_group("Workspace bounds & plane")
    space.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Cartesian workspace bounds (m). If omitted in cartesian mode, "
        "bounds are computed dynamically from joint-space FK.",
    )
    space.add_argument(
        "--plane-normal",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("NX", "NY", "NZ"),
        help="Plane normal for plane_sampling mode (default: 0 0 1).",
    )
    space.add_argument(
        "--plane-point",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Point on the plane for plane_sampling mode (default: 0 0 0).",
    )
    space.add_argument(
        "--plane-bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("UMIN", "UMAX", "VMIN", "VMAX"),
        help="2D plane coordinate bounds for plane_sampling mode. If omitted, "
        "computed dynamically from joint-space FK.",
    )

    # --- Cache --------------------------------------------------------------
    cache = parser.add_argument_group("Cache")
    cache.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached results. Default: "
        "~/.cache/embodichain_data/robot_workspace.",
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable results caching.",
    )
    cache.add_argument(
        "--force-recompute",
        action="store_true",
        default=False,
        help="Recompute even if a cached result exists.",
    )
    cache.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export a copy of the results to this path (see --export-format).",
    )
    cache.add_argument(
        "--export-format",
        type=str,
        choices=["npz", "pkl", "json"],
        default="npz",
        help="Export format for --output (default: npz).",
    )

    # --- Visualization ------------------------------------------------------
    viz = parser.add_argument_group("Visualization")
    viz.add_argument(
        "--vis-type",
        type=str,
        choices=["point_cloud", "voxel", "sphere", "axis"],
        default="point_cloud",
        help="Visualization type (default: point_cloud).",
    )
    viz.add_argument(
        "--point-size", type=float, default=4.0, help="Point size (default: 4.0)."
    )
    viz.add_argument(
        "--voxel-size", type=float, default=0.05, help="Voxel size (default: 0.05)."
    )
    viz.add_argument(
        "--hide-unreachable",
        action="store_true",
        default=False,
        help="Hide unreachable points in cartesian/plane modes.",
    )
    viz.add_argument(
        "--no-visualize",
        action="store_true",
        default=False,
        help="Do not visualize the workspace in the sim window.",
    )

    # --- Simulation ---------------------------------------------------------
    sim = parser.add_argument_group("Simulation")
    sim.add_argument(
        "--sim-device",
        type=str,
        default="cpu",
        help="Simulation device (default: cpu).",
    )
    sim.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without a rendering window (disables visualization).",
    )
    sim.add_argument(
        "--renderer",
        type=str,
        choices=["hybrid", "fast-rt", "rt"],
        default="hybrid",
        help="Renderer backend (default: hybrid).",
    )
    sim.add_argument(
        "--width", type=int, default=1920, help="Window width (default: 1920)."
    )
    sim.add_argument(
        "--height", type=int, default=1080, help="Window height (default: 1080)."
    )

    args = parser.parse_args(argv)

    # Resolve cache dir default here (after parse) to keep --help output clean.
    if args.cache_dir is None and not args.no_cache:
        from embodichain.lab.sim.utility.workspace_analyzer.caches import (
            DEFAULT_RESULTS_CACHE_DIR,
        )

        args.cache_dir = DEFAULT_RESULTS_CACHE_DIR

    # ``args.visualize`` is the user intent; ``main()`` ANDs it with ``--headless``.
    args.visualize = not args.no_visualize

    return args


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for workspace analysis.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    main(parse_args(argv))


if __name__ == "__main__":
    cli()
