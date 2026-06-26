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

"""Benchmark Press atomic action across object presets and start positions.

The benchmark sweeps object presets such as bottle and mug against multiple
initial XY positions that cover all four workspace quadrants. It reports
planning latency, memory usage, planning success, and whether the generated
trajectory reaches the object's top center.
Run: python -m scripts.benchmark.atomic_action.press_benchmark
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

CPU_MEMORY_BACKEND = "psutil" if psutil is not None else "resource"
_RUNTIME_IMPORTS_READY = False

# Keep these constants aligned with scripts/tutorials/atomic_action/press.py.
DEFAULT_PRESS_TOLERANCE = 0.01
MOVE_SAMPLE_INTERVAL = 60
PRESS_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 12
TABLE_TOP_Z = -0.045
PRESS_CLEARANCE = 0.13
PRESS_SURFACE_OFFSET = 0.003


def _ensure_runtime_imports() -> None:
    """Import simulation dependencies only when the benchmark is executed."""
    global _RUNTIME_IMPORTS_READY
    if _RUNTIME_IMPORTS_READY:
        return

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import torch as torch_module
        from embodichain.lab.sim import SimulationManager as simulation_manager_cls
        from embodichain.lab.sim.atomic_actions import (
            AtomicActionEngine as atomic_action_engine_cls,
            EndEffectorPoseTarget as end_effector_pose_target_cls,
            MoveEndEffector as move_end_effector_cls,
            MoveEndEffectorCfg as move_end_effector_cfg_cls,
            Press as press_cls,
            PressCfg as press_cfg_cls,
        )
        from embodichain.lab.sim.cfg import (
            RigidBodyAttributesCfg as rigid_body_attributes_cfg_cls,
            RigidObjectCfg as rigid_object_cfg_cls,
        )
        from embodichain.lab.sim.material import (
            VisualMaterialCfg as visual_material_cfg_cls,
        )
        from embodichain.lab.sim.objects import RigidObject as rigid_object_cls
        from embodichain.lab.sim.objects import Robot as robot_cls
        from embodichain.lab.sim.planners import (
            MotionGenerator as motion_generator_cls,
            MotionGenCfg as motion_gen_cfg_cls,
            ToppraPlannerCfg as toppra_planner_cfg_cls,
        )
        from embodichain.lab.sim.shapes import CubeCfg as cube_cfg_cls
        from scripts.tutorials.atomic_action.press import (
            create_robot as create_robot_fn,
            create_table as create_table_fn,
            get_hand_close_qpos as get_hand_close_qpos_fn,
            initialize_simulation as initialize_simulation_fn,
            make_top_down_eef_pose as make_top_down_eef_pose_fn,
            settle_object as settle_object_fn,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Atomic action benchmark requires the EmbodiChain simulation runtime "
            f"and PyTorch. Missing module: {exc.name}."
        ) from exc

    globals().update(
        {
            "torch": torch_module,
            "SimulationManager": simulation_manager_cls,
            "AtomicActionEngine": atomic_action_engine_cls,
            "EndEffectorPoseTarget": end_effector_pose_target_cls,
            "MoveEndEffector": move_end_effector_cls,
            "MoveEndEffectorCfg": move_end_effector_cfg_cls,
            "Press": press_cls,
            "PressCfg": press_cfg_cls,
            "RigidBodyAttributesCfg": rigid_body_attributes_cfg_cls,
            "RigidObjectCfg": rigid_object_cfg_cls,
            "VisualMaterialCfg": visual_material_cfg_cls,
            "RigidObject": rigid_object_cls,
            "Robot": robot_cls,
            "MotionGenerator": motion_generator_cls,
            "MotionGenCfg": motion_gen_cfg_cls,
            "ToppraPlannerCfg": toppra_planner_cfg_cls,
            "CubeCfg": cube_cfg_cls,
            "create_robot": create_robot_fn,
            "create_table": create_table_fn,
            "get_hand_close_qpos": get_hand_close_qpos_fn,
            "initialize_simulation": initialize_simulation_fn,
            "make_top_down_eef_pose": make_top_down_eef_pose_fn,
            "settle_object": settle_object_fn,
        }
    )
    _RUNTIME_IMPORTS_READY = True


@dataclass(frozen=True)
class ObjectPreset:
    """Primitive object preset used by the atomic-action benchmark."""

    object_type: str
    material_name: str
    size: tuple[float, float, float]
    base_color: tuple[float, float, float, float]
    roughness: float
    dynamic_friction: float = 0.8
    static_friction: float = 0.9


@dataclass(frozen=True)
class PositionCase:
    """Initial object position case with a quadrant label."""

    name: str
    quadrant: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class PressCaseResult:
    """Result for one Press benchmark case."""

    case_id: str
    object_type: str
    material_name: str
    quadrant: str
    position_case: str
    init_xy: tuple[float, float]
    repeat_index: int
    planning_success: bool
    center_hit: bool
    cost_time_ms: float
    cpu_delta_mb: float
    gpu_delta_mb: float
    peak_gpu_mb: float
    xy_error_m: float | None
    hit_step: int | None
    trajectory_waypoints: int
    failure_reason: str


OBJECT_PRESETS: dict[str, ObjectPreset] = {
    "bottle": ObjectPreset(
        object_type="bottle",
        material_name="green_plastic",
        size=(0.06, 0.06, 0.16),
        base_color=(0.10, 0.45, 0.32, 1.0),
        roughness=0.55,
    ),
    "mug": ObjectPreset(
        object_type="mug",
        material_name="ceramic",
        size=(0.10, 0.08, 0.10),
        base_color=(0.88, 0.85, 0.78, 1.0),
        roughness=0.35,
    ),
    "wooden_block": ObjectPreset(
        object_type="wooden_block",
        material_name="wood",
        size=(0.12, 0.12, 0.06),
        base_color=(0.58, 0.32, 0.14, 1.0),
        roughness=0.85,
    ),
}

POSITION_CASES: dict[str, PositionCase] = {
    "q1_near": PositionCase(name="q1_near", quadrant="q1", xy=(0.02, 0.18)),
    "q1_far": PositionCase(name="q1_far", quadrant="q1", xy=(0.12, 0.36)),
    "q2_near": PositionCase(name="q2_near", quadrant="q2", xy=(-0.42, 0.18)),
    "q2_far": PositionCase(name="q2_far", quadrant="q2", xy=(-0.62, 0.36)),
    "q3_near": PositionCase(name="q3_near", quadrant="q3", xy=(-0.42, -0.18)),
    "q3_far": PositionCase(name="q3_far", quadrant="q3", xy=(-0.62, -0.36)),
    "q4_near": PositionCase(name="q4_near", quadrant="q4", xy=(0.02, -0.18)),
    "q4_far": PositionCase(name="q4_far", quadrant="q4", xy=(0.12, -0.36)),
}

DEFAULT_OBJECT_TYPES = ("bottle", "mug")
DEFAULT_POSITION_CASES = tuple(POSITION_CASES.keys())


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add atomic-action benchmark arguments to an argument parser."""
    parser.add_argument(
        "--object_types",
        nargs="+",
        choices=(*OBJECT_PRESETS.keys(), "all"),
        default=list(DEFAULT_OBJECT_TYPES),
        help="Object presets to benchmark. Use 'all' to include every preset.",
    )
    parser.add_argument(
        "--position_cases",
        nargs="+",
        choices=(*POSITION_CASES.keys(), "all"),
        default=list(DEFAULT_POSITION_CASES),
        help="Initial position cases to benchmark. Use 'all' for all quadrants.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats for every object-position case.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one representative case only for quick validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Simulation device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=("auto", "hybrid", "fast-rt", "rt"),
        default="auto",
        help="Renderer backend used by SimulationManager.",
    )
    parser.add_argument(
        "--press_tolerance",
        type=float,
        default=DEFAULT_PRESS_TOLERANCE,
        help="XY tolerance in meters for the press-center check.",
    )


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the atomic-action benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Press atomic action over object presets and initial "
            "workspace quadrants."
        )
    )
    add_benchmark_args(parser)
    return parser.parse_args()


def _sync_cuda() -> None:
    """Synchronize CUDA stream when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory stats when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated by PyTorch in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def _memory_snapshot() -> dict[str, float]:
    """Return current process memory usage snapshot in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        cpu_mb = process.memory_info().rss / 1024**2
    else:
        cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def _format_float(value: float | None, precision: int = 6) -> str:
    """Format finite floats for tables and use N/A for missing values."""
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def _format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows into a markdown table."""
    if not rows:
        return ["No data."]

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def _write_markdown_report(
    benchmark_name: str,
    perf_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    leaderboard_rows: list[dict[str, object]],
    notes: list[str] | None = None,
) -> Path:
    """Write benchmark results into one markdown report file."""
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{benchmark_name}_{timestamp}.md"

    lines: list[str] = [
        f"# {benchmark_name} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
    ]
    lines.extend(_format_markdown_table(perf_rows))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(_format_markdown_table(metric_rows))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_format_markdown_table(leaderboard_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _select_object_presets(object_types: list[str]) -> list[ObjectPreset]:
    """Resolve selected object preset names."""
    if "all" in object_types:
        return list(OBJECT_PRESETS.values())
    return [OBJECT_PRESETS[name] for name in object_types]


def _select_position_cases(position_cases: list[str]) -> list[PositionCase]:
    """Resolve selected position case names."""
    if "all" in position_cases:
        return list(POSITION_CASES.values())
    return [POSITION_CASES[name] for name in position_cases]


def _create_benchmark_object(
    sim: SimulationManager,
    preset: ObjectPreset,
    position_case: PositionCase,
    repeat_index: int,
) -> RigidObject:
    """Create one static benchmark object at the requested initial position."""
    init_pos = (
        position_case.xy[0],
        position_case.xy[1],
        TABLE_TOP_Z + 0.5 * preset.size[2],
    )
    uid = f"atomic_benchmark_{preset.object_type}_{position_case.name}_{repeat_index}"
    cfg = RigidObjectCfg(
        uid=uid,
        shape=CubeCfg(
            size=list(preset.size),
            visual_material=VisualMaterialCfg(
                uid=f"{preset.object_type}_{preset.material_name}_mat",
                base_color=list(preset.base_color),
                roughness=preset.roughness,
            ),
        ),
        body_type="static",
        attrs=RigidBodyAttributesCfg(
            dynamic_friction=preset.dynamic_friction,
            static_friction=preset.static_friction,
        ),
        init_pos=init_pos,
    )
    return sim.add_rigid_object(cfg=cfg)


def _reset_robot(robot: Robot, initial_qpos: torch.Tensor) -> None:
    """Reset current and target robot qpos to the benchmark initial posture."""
    for target in (False, True):
        robot.set_qpos(initial_qpos, target=target)
    robot.clear_dynamics()


def _build_atomic_engine(
    motion_gen: MotionGenerator,
    robot: Robot,
    device: torch.device,
) -> AtomicActionEngine:
    """Build a Press benchmark engine with MoveEndEffector pre-positioning."""
    hand_close = get_hand_close_qpos(robot, device)
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(
        MoveEndEffector(
            motion_gen,
            cfg=MoveEndEffectorCfg(
                control_part="arm",
                sample_interval=MOVE_SAMPLE_INTERVAL,
            ),
        )
    )
    atomic_engine.register(
        Press(
            motion_gen,
            cfg=PressCfg(
                control_part="arm",
                hand_control_part="hand",
                hand_close_qpos=hand_close,
                sample_interval=PRESS_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    return atomic_engine


def _make_press_targets(
    obj: RigidObject,
    preset: ObjectPreset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pre-press and press poses for the object's top center."""
    obj_pose = obj.get_local_pose(to_matrix=True)
    object_center = obj_pose[0, :3, 3].clone()
    object_top_z = object_center[2] + 0.5 * preset.size[2]

    press_position = object_center.clone()
    press_position[2] = object_top_z + PRESS_SURFACE_OFFSET
    move_position = press_position.clone()
    move_position[2] = object_top_z + PRESS_CLEARANCE

    return make_top_down_eef_pose(move_position), make_top_down_eef_pose(
        press_position
    )


def _compute_press_center_check(
    robot: Robot,
    traj: torch.Tensor,
    obj: RigidObject,
    object_height: float,
    tolerance: float,
) -> tuple[bool, float, int]:
    """Check whether the planned Press trajectory reaches the object top center."""
    if traj.numel() == 0:
        return False, float("inf"), -1

    arm_joint_ids = robot.get_joint_ids(name="arm")
    n_down = (PRESS_SAMPLE_INTERVAL - HAND_INTERP_STEPS) // 2
    press_segment_start = MOVE_SAMPLE_INTERVAL + HAND_INTERP_STEPS
    press_segment_end = min(press_segment_start + n_down, traj.shape[1])
    arm_traj = traj[:, press_segment_start:press_segment_end, arm_joint_ids]
    if arm_traj.shape[1] == 0:
        return False, float("inf"), -1

    fk_pose = torch.stack(
        [
            robot.compute_fk(
                qpos=waypoint.unsqueeze(0),
                name="arm",
                to_matrix=True,
            )[0]
            for waypoint in arm_traj[0]
        ],
        dim=0,
    )

    obj_pose = obj.get_local_pose(to_matrix=True)
    object_center = obj_pose[0, :3, 3]
    object_top_z = object_center[2] + 0.5 * object_height
    target_xy = object_center[:2]
    target_z = object_top_z + PRESS_SURFACE_OFFSET

    xy_error = torch.linalg.norm(fk_pose[:, :2, 3] - target_xy, dim=1)
    z_error = torch.abs(fk_pose[:, 2, 3] - target_z)
    combined_error = xy_error + z_error
    best_idx = int(torch.argmin(combined_error).item())
    best_pos = fk_pose[best_idx, :3, 3]
    center_error = float(torch.linalg.norm(best_pos[:2] - target_xy).item())
    return center_error <= tolerance, center_error, press_segment_start + best_idx


def _timed_atomic_run(
    atomic_engine: AtomicActionEngine,
    move_target: torch.Tensor,
    press_target: torch.Tensor,
) -> tuple[float, dict[str, float], float, bool, torch.Tensor]:
    """Run a timed atomic-action sequence and return timing/memory/results."""
    _reset_peak_gpu_memory()
    mem_before = _memory_snapshot()
    _sync_cuda()

    start = time.perf_counter()
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_end_effector", EndEffectorPoseTarget(xpos=move_target)),
            ("press", EndEffectorPoseTarget(xpos=press_target)),
        ]
    )
    _sync_cuda()
    elapsed = time.perf_counter() - start

    mem_after = _memory_snapshot()
    deltas = {
        "cpu_mb": mem_after["cpu_mb"] - mem_before["cpu_mb"],
        "gpu_mb": mem_after["gpu_mb"] - mem_before["gpu_mb"],
    }
    return elapsed, deltas, _peak_gpu_memory_mb(), is_success, traj


def _run_press_case(
    sim: SimulationManager,
    robot: Robot,
    atomic_engine: AtomicActionEngine,
    initial_qpos: torch.Tensor,
    preset: ObjectPreset,
    position_case: PositionCase,
    repeat_index: int,
    press_tolerance: float,
) -> PressCaseResult:
    """Run one object-position Press benchmark case."""
    case_id = f"{preset.object_type}:{position_case.name}:r{repeat_index}"
    obj: RigidObject | None = None
    try:
        _reset_robot(robot, initial_qpos)
        obj = _create_benchmark_object(sim, preset, position_case, repeat_index)
        settle_object(sim, obj, step=5)
        move_target, press_target = _make_press_targets(obj, preset)

        elapsed, mem_delta, peak_gpu, planning_success, traj = _timed_atomic_run(
            atomic_engine=atomic_engine,
            move_target=move_target,
            press_target=press_target,
        )
        center_hit = False
        xy_error_m: float | None = None
        hit_step: int | None = None
        failure_reason = ""
        if planning_success:
            center_hit, xy_error_m, raw_hit_step = _compute_press_center_check(
                robot=robot,
                traj=traj,
                obj=obj,
                object_height=preset.size[2],
                tolerance=press_tolerance,
            )
            hit_step = raw_hit_step if raw_hit_step >= 0 else None
            if not center_hit:
                failure_reason = "center_miss"
        else:
            failure_reason = "planning_failed"

        return PressCaseResult(
            case_id=case_id,
            object_type=preset.object_type,
            material_name=preset.material_name,
            quadrant=position_case.quadrant,
            position_case=position_case.name,
            init_xy=position_case.xy,
            repeat_index=repeat_index,
            planning_success=planning_success,
            center_hit=center_hit,
            cost_time_ms=elapsed * 1000.0,
            cpu_delta_mb=mem_delta["cpu_mb"],
            gpu_delta_mb=mem_delta["gpu_mb"],
            peak_gpu_mb=peak_gpu,
            xy_error_m=xy_error_m,
            hit_step=hit_step,
            trajectory_waypoints=int(traj.shape[1]) if traj.ndim >= 2 else 0,
            failure_reason=failure_reason,
        )
    except Exception as exc:
        return PressCaseResult(
            case_id=case_id,
            object_type=preset.object_type,
            material_name=preset.material_name,
            quadrant=position_case.quadrant,
            position_case=position_case.name,
            init_xy=position_case.xy,
            repeat_index=repeat_index,
            planning_success=False,
            center_hit=False,
            cost_time_ms=0.0,
            cpu_delta_mb=0.0,
            gpu_delta_mb=0.0,
            peak_gpu_mb=0.0,
            xy_error_m=None,
            hit_step=None,
            trajectory_waypoints=0,
            failure_reason=f"exception:{type(exc).__name__}:{exc}",
        )
    finally:
        if obj is not None:
            sim.remove_asset(obj.uid)


def _build_perf_rows(results: list[PressCaseResult]) -> list[dict[str, object]]:
    """Build Time & Memory table rows."""
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "sample_size": 1,
                "impl": "press",
                "case_id": result.case_id,
                "object_type": result.object_type,
                "material": result.material_name,
                "quadrant": result.quadrant,
                "position_case": result.position_case,
                "init_xy": f"({result.init_xy[0]:.3f},{result.init_xy[1]:.3f})",
                "repeat": result.repeat_index,
                "cost_time_ms": _format_float(result.cost_time_ms),
                "cpu_delta_mb": _format_float(result.cpu_delta_mb),
                "gpu_delta_mb": _format_float(result.gpu_delta_mb),
                "peak_gpu_mb": _format_float(result.peak_gpu_mb),
            }
        )
    return rows


def _build_metric_rows(results: list[PressCaseResult]) -> list[dict[str, object]]:
    """Build Success & Other Metrics table rows."""
    rows: list[dict[str, object]] = []
    for result in results:
        overall_success = result.planning_success and result.center_hit
        rows.append(
            {
                "sample_size": 1,
                "impl": "press",
                "case_id": result.case_id,
                "object_type": result.object_type,
                "material": result.material_name,
                "quadrant": result.quadrant,
                "position_case": result.position_case,
                "success_rate": f"{float(overall_success):.6f}",
                "planning_success_rate": f"{float(result.planning_success):.6f}",
                "center_hit_rate": f"{float(result.center_hit):.6f}",
                "xy_error_m": _format_float(result.xy_error_m),
                "hit_step": result.hit_step if result.hit_step is not None else "N/A",
                "trajectory_waypoints": result.trajectory_waypoints,
                "failure_reason": result.failure_reason or "N/A",
            }
        )
    return rows


def _build_leaderboard_rows(results: list[PressCaseResult]) -> list[dict[str, object]]:
    """Aggregate and rank object-conditioned Press variants by success rate."""
    aggregate: dict[str, dict[str, float | set[str]]] = {}
    for result in results:
        algorithm = f"press:{result.object_type}"
        if algorithm not in aggregate:
            aggregate[algorithm] = {
                "overall_success_sum": 0.0,
                "planning_success_sum": 0.0,
                "xy_error_sum": 0.0,
                "xy_error_count": 0.0,
                "cost_time_sum": 0.0,
                "case_count": 0.0,
                "quadrants": set(),
            }

        stats = aggregate[algorithm]
        stats["overall_success_sum"] = float(stats["overall_success_sum"]) + float(
            result.planning_success and result.center_hit
        )
        stats["planning_success_sum"] = float(stats["planning_success_sum"]) + float(
            result.planning_success
        )
        if result.xy_error_m is not None and math.isfinite(result.xy_error_m):
            stats["xy_error_sum"] = float(stats["xy_error_sum"]) + result.xy_error_m
            stats["xy_error_count"] = float(stats["xy_error_count"]) + 1.0
        stats["cost_time_sum"] = float(stats["cost_time_sum"]) + result.cost_time_ms
        stats["case_count"] = float(stats["case_count"]) + 1.0
        quadrants = stats["quadrants"]
        if isinstance(quadrants, set):
            quadrants.add(result.quadrant)

    ranked = sorted(
        aggregate.items(),
        key=lambda item: (
            float(item[1]["overall_success_sum"])
            / max(float(item[1]["case_count"]), 1.0),
            -float(item[1]["cost_time_sum"]) / max(float(item[1]["case_count"]), 1.0),
        ),
        reverse=True,
    )

    rows: list[dict[str, object]] = []
    for rank, (algorithm, stats) in enumerate(ranked, start=1):
        case_count = max(float(stats["case_count"]), 1.0)
        xy_error_count = float(stats["xy_error_count"])
        avg_xy_error = (
            float(stats["xy_error_sum"]) / xy_error_count
            if xy_error_count > 0.0
            else None
        )
        quadrants = stats["quadrants"]
        quadrant_coverage = (
            ",".join(sorted(quadrants)) if isinstance(quadrants, set) else ""
        )
        rows.append(
            {
                "rank": rank,
                "algorithm": algorithm,
                "overall_success_rate": (
                    f"{float(stats['overall_success_sum']) / case_count:.2%}"
                ),
                "planning_success_rate": (
                    f"{float(stats['planning_success_sum']) / case_count:.2%}"
                ),
                "avg_xy_error_m": _format_float(avg_xy_error),
                "avg_cost_time_ms": _format_float(
                    float(stats["cost_time_sum"]) / case_count
                ),
                "evaluated_cases": int(case_count),
                "quadrant_coverage": quadrant_coverage,
            }
        )
    return rows


def _print_case_result(result: PressCaseResult) -> None:
    """Print one aligned case result line."""
    overall_success = result.planning_success and result.center_hit
    print(
        f"  {result.case_id:<28} "
        f"time={result.cost_time_ms:>10.2f} ms | "
        f"CPU delta={result.cpu_delta_mb:+.1f} MB  "
        f"GPU delta={result.gpu_delta_mb:+.1f} MB  "
        f"peak GPU={result.peak_gpu_mb:.1f} MB | "
        f"success={overall_success} "
        f"xy_error={_format_float(result.xy_error_m, precision=4)}"
    )
    if result.failure_reason:
        print(f"    reason={result.failure_reason}")


def _build_notes(
    object_presets: list[ObjectPreset],
    position_cases: list[PositionCase],
    repeat: int,
) -> list[str]:
    """Build report notes with benchmark coverage metadata."""
    quadrant_counts: dict[str, int] = {}
    for position_case in position_cases:
        quadrant_counts[position_case.quadrant] = (
            quadrant_counts.get(position_case.quadrant, 0) + 1
        )
    return [
        "Object presets: "
        + ", ".join(
            f"{preset.object_type}/{preset.material_name}/size={preset.size}"
            for preset in object_presets
        ),
        "Position cases per quadrant: "
        + ", ".join(
            f"{quadrant}={count}" for quadrant, count in sorted(quadrant_counts.items())
        ),
        f"CPU memory backend: {CPU_MEMORY_BACKEND}",
        f"Repeat per object-position case: {repeat}",
        "success_rate is 1 only when planning succeeds and the Press trajectory "
        "reaches the object top center.",
    ]


def run_all_benchmarks(args: argparse.Namespace | None = None) -> Path:
    """Run all atomic-action benchmarks and write the markdown report."""
    args = _parse_args() if args is None else args
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    _ensure_runtime_imports()

    object_presets = _select_object_presets(args.object_types)
    position_cases = _select_position_cases(args.position_cases)
    repeat = args.repeat
    if args.smoke:
        object_presets = [OBJECT_PRESETS["bottle"]]
        position_cases = [POSITION_CASES["q3_near"]]
        repeat = 1

    print("=" * 60)
    print("Atomic Action Press Benchmark")
    print("=" * 60)
    print(
        "Coverage: "
        f"{len(object_presets)} object presets x "
        f"{len(position_cases)} position cases x {repeat} repeat(s)"
    )

    sim = initialize_simulation(args)
    robot = create_robot(sim)
    create_table(sim)
    initial_qpos = robot.get_qpos().clone()
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    atomic_engine = _build_atomic_engine(motion_gen, robot, sim.device)

    results: list[PressCaseResult] = []
    print("\n=== Press Object/Position Sweep ===")
    for preset in object_presets:
        for position_case in position_cases:
            for repeat_index in range(repeat):
                result = _run_press_case(
                    sim=sim,
                    robot=robot,
                    atomic_engine=atomic_engine,
                    initial_qpos=initial_qpos,
                    preset=preset,
                    position_case=position_case,
                    repeat_index=repeat_index,
                    press_tolerance=args.press_tolerance,
                )
                results.append(result)
                _print_case_result(result)

    perf_rows = _build_perf_rows(results)
    metric_rows = _build_metric_rows(results)
    leaderboard_rows = _build_leaderboard_rows(results)
    report_path = _write_markdown_report(
        benchmark_name="atomic_action_press",
        perf_rows=perf_rows,
        metric_rows=metric_rows,
        leaderboard_rows=leaderboard_rows,
        notes=_build_notes(object_presets, position_cases, repeat),
    )

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print(f"Markdown report saved: {report_path}")
    print("=" * 60)
    return report_path


def main() -> None:
    """Run the CLI entry point."""
    try:
        run_all_benchmarks()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


__all__ = ["add_benchmark_args", "run_all_benchmarks"]
