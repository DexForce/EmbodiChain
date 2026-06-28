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

"""Shared helpers for atomic-action benchmark scripts."""

from __future__ import annotations

import argparse
import math
import os
import re
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

CPU_MEMORY_BACKEND = "psutil" if psutil is not None else "resource"
DEFAULT_VIDEO_DIR = Path("outputs/benchmark_videos")
DEFAULT_VIDEO_FPS = 20
DEFAULT_VIDEO_MAX_MEMORY_MB = 2048
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_HOLD_STEPS = 120
DEFAULT_VIDEO_CASE_LIMIT = 0
SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z = 0.35
SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT = 0.5
BENCHMARK_PROFILES = ("smoke", "coverage", "full")
DEFAULT_BENCHMARK_PROFILE = "coverage"
DEFAULT_VIDEO_LOOK_AT = (
    (-1.25, -1.15, 0.95),
    (-0.25, -0.02, 0.25),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class PositionCase:
    """Initial object position case with a quadrant label."""

    name: str
    quadrant: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class MeshObjectPreset:
    """Real mesh object preset used by object-conditioned benchmarks."""

    object_type: str
    material_name: str
    label: str
    init_rot: tuple[float, float, float]
    body_scale: tuple[float, float, float]
    mass: float
    initial_z: float
    mesh_path: str = ""
    shape_type: str = "mesh"
    cube_size: tuple[float, float, float] | None = None
    use_usd_properties: bool = False
    dynamic_friction: float = 0.97
    static_friction: float = 0.99
    restitution: float = 0.0
    contact_offset: float = 0.002
    rest_offset: float = 0.0
    linear_damping: float = 0.7
    angular_damping: float = 0.7
    max_depenetration_velocity: float = 10.0
    min_position_iters: int = 4
    min_velocity_iters: int = 1
    max_linear_velocity: float = 100.0
    max_angular_velocity: float = 100.0
    max_convex_hull_num: int = 16
    enable_ccd: bool = False


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
FULL_POSITION_CASE_NAMES = tuple(POSITION_CASES.keys())
COVERAGE_POSITION_CASE_NAMES = FULL_POSITION_CASE_NAMES
SMOKE_POSITION_CASE_NAMES = ("q3_near",)

MESH_OBJECT_PRESETS: dict[str, MeshObjectPreset] = {
    "sugar_box": MeshObjectPreset(
        object_type="sugar_box",
        material_name="cardboard",
        label="sugar_box",
        mesh_path="SugarBox/sugar_box_usd/sugar_box.usda",
        init_rot=(0.0, 0.0, 0.0),
        body_scale=(0.8, 0.8, 0.8),
        mass=0.05,
        initial_z=0.05,
        use_usd_properties=False,
    ),
    "coffee_cup": MeshObjectPreset(
        object_type="coffee_cup",
        material_name="ceramic",
        label="coffee_cup",
        mesh_path="CoffeeCup/cup.ply",
        init_rot=(0.0, 0.0, -90.0),
        body_scale=(4.0, 4.0, 4.0),
        mass=0.01,
        initial_z=0.01,
        use_usd_properties=False,
    ),
    "cube": MeshObjectPreset(
        object_type="cube",
        material_name="plastic",
        label="cube",
        shape_type="cube",
        cube_size=(0.05, 0.05, 0.05),
        init_rot=(0.0, 0.0, 0.0),
        body_scale=(1.0, 1.0, 1.0),
        mass=0.05,
        initial_z=0.05,
        use_usd_properties=False,
        dynamic_friction=0.5,
        static_friction=0.5,
        contact_offset=0.003,
        rest_offset=0.001,
        max_depenetration_velocity=10.0,
        min_position_iters=32,
        min_velocity_iters=8,
        max_convex_hull_num=1,
    ),
    "paper_cup": MeshObjectPreset(
        object_type="paper_cup",
        material_name="paper",
        label="paper_cup",
        mesh_path="PaperCup/paper_cup.ply",
        init_rot=(0.0, 0.0, 0.0),
        body_scale=(0.75, 0.75, 1.0),
        mass=0.01,
        initial_z=0.05,
        use_usd_properties=False,
        dynamic_friction=1.0,
        static_friction=1.0,
        contact_offset=0.003,
        rest_offset=0.001,
        linear_damping=2.0,
        angular_damping=2.0,
        max_depenetration_velocity=2.0,
        min_position_iters=32,
        min_velocity_iters=8,
        max_linear_velocity=5.0,
        max_angular_velocity=10.0,
        max_convex_hull_num=8,
    ),
    "scanned_bottle": MeshObjectPreset(
        object_type="scanned_bottle",
        material_name="plastic",
        label="scanned_bottle",
        mesh_path="ScannedBottle/yibao_processed.ply",
        init_rot=(0.0, 0.0, 0.0),
        body_scale=(1.0, 1.0, 1.0),
        mass=0.05,
        initial_z=0.05,
        use_usd_properties=False,
    ),
}
COVERAGE_MESH_OBJECT_TYPES = ("sugar_box", "cube", "paper_cup")
FULL_MESH_OBJECT_TYPES = COVERAGE_MESH_OBJECT_TYPES
SMOKE_MESH_OBJECT_TYPES = ("sugar_box",)
PICKUP_APPROACH_CASES = ("top", "side")
SMOKE_PICKUP_APPROACH_CASES = ("top",)


def ensure_repo_root() -> None:
    """Add the repository root to sys.path for module execution."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def ensure_torch():
    """Import torch or raise a clear benchmark runtime error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Atomic action benchmark requires the EmbodiChain simulation runtime "
            f"and PyTorch. Missing module: {exc.name}."
        ) from exc
    return torch


def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common atomic-action benchmark CLI arguments."""
    add_profile_benchmark_args(parser)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats for every benchmark case.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --profile smoke.",
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
    add_video_benchmark_args(parser)


def add_video_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add optional benchmark video recording CLI arguments."""
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record trajectory replay videos for selected successful cases.",
    )
    parser.add_argument(
        "--record_failed_video",
        action="store_true",
        help=(
            "With --record_video, also record failed cases when a partial "
            "trajectory or static debug scene is available."
        ),
    )
    parser.add_argument(
        "--video_case_limit",
        type=int,
        default=DEFAULT_VIDEO_CASE_LIMIT,
        help="Maximum cases to record. Use 0 to record all selected cases.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory for benchmark replay videos.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help="Recorded video frames per second.",
    )
    parser.add_argument(
        "--video_max_memory",
        type=int,
        default=DEFAULT_VIDEO_MAX_MEMORY_MB,
        help="Maximum recorder frame-buffer memory in MB.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help="Recorded video width in pixels.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help="Recorded video height in pixels.",
    )
    parser.add_argument(
        "--video_hold_steps",
        type=int,
        default=DEFAULT_VIDEO_HOLD_STEPS,
        help="Extra simulation steps to hold the final replay pose.",
    )


def add_grasp_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common grasp-affordance setup arguments."""
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )


def add_profile_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add unified benchmark profile CLI arguments."""
    parser.add_argument(
        "--profile",
        choices=BENCHMARK_PROFILES,
        default=DEFAULT_BENCHMARK_PROFILE,
        help=(
            "Benchmark profile: smoke is one fast case, coverage is the default "
            "core object/position matrix, and full sweeps all configured cases."
        ),
    )


def add_object_position_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add object and initial-position selection arguments."""
    parser.add_argument(
        "--object_types",
        nargs="+",
        choices=(*MESH_OBJECT_PRESETS.keys(), "all"),
        default=None,
        help=(
            "Real mesh object presets to benchmark. Defaults are selected by "
            "--profile; use 'all' to include every default full preset."
        ),
    )
    parser.add_argument(
        "--position_cases",
        nargs="+",
        choices=(*POSITION_CASES.keys(), "all"),
        default=None,
        help=(
            "Initial object position cases to benchmark. Defaults are selected by "
            "--profile; use 'all' for all near/far cases."
        ),
    )


def add_pickup_approach_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common PickUp approach selection arguments."""
    parser.add_argument(
        "--approach_cases",
        nargs="+",
        choices=(*PICKUP_APPROACH_CASES, "all"),
        default=None,
        help=(
            "PickUp approach cases to benchmark. Defaults are selected by "
            "--profile; side uses the current object's initial XY direction."
        ),
    )


def resolve_profile(args: argparse.Namespace) -> str:
    """Resolve the effective benchmark profile from CLI arguments."""
    profile = getattr(args, "profile", DEFAULT_BENCHMARK_PROFILE)
    if getattr(args, "smoke", False):
        profile = "smoke"
    if profile not in BENCHMARK_PROFILES:
        raise ValueError(
            f"Unsupported benchmark profile {profile!r}. "
            f"Expected one of {BENCHMARK_PROFILES}."
        )
    return profile


def default_position_case_names_for_profile(profile: str) -> tuple[str, ...]:
    """Return default position case names for a profile."""
    if profile == "smoke":
        return SMOKE_POSITION_CASE_NAMES
    if profile == "coverage":
        return COVERAGE_POSITION_CASE_NAMES
    if profile == "full":
        return FULL_POSITION_CASE_NAMES
    raise ValueError(f"Unsupported benchmark profile: {profile}")


def select_position_cases(
    case_names: Sequence[str] | None,
    profile: str,
) -> list[PositionCase]:
    """Resolve requested position cases, falling back to profile defaults."""
    names = (
        default_position_case_names_for_profile(profile)
        if not case_names
        else tuple(case_names)
    )
    if "all" in names:
        names = FULL_POSITION_CASE_NAMES
    return [POSITION_CASES[name] for name in names]


def default_mesh_object_types_for_profile(profile: str) -> tuple[str, ...]:
    """Return default real mesh object preset names for a profile."""
    if profile == "smoke":
        return SMOKE_MESH_OBJECT_TYPES
    if profile == "coverage":
        return COVERAGE_MESH_OBJECT_TYPES
    if profile == "full":
        return FULL_MESH_OBJECT_TYPES
    raise ValueError(f"Unsupported benchmark profile: {profile}")


def select_mesh_object_presets(
    object_types: Sequence[str] | None,
    profile: str,
) -> list[MeshObjectPreset]:
    """Resolve requested mesh object presets, falling back to profile defaults."""
    names = (
        default_mesh_object_types_for_profile(profile)
        if not object_types
        else tuple(object_types)
    )
    if "all" in names:
        names = FULL_MESH_OBJECT_TYPES
    return [MESH_OBJECT_PRESETS[name] for name in names]


def default_pickup_approach_cases_for_profile(profile: str) -> tuple[str, ...]:
    """Return default PickUp approach case names for a profile."""
    if profile == "smoke":
        return SMOKE_PICKUP_APPROACH_CASES
    if profile in ("coverage", "full"):
        return PICKUP_APPROACH_CASES
    raise ValueError(f"Unsupported benchmark profile: {profile}")


def select_pickup_approaches(
    approach_cases: Sequence[str] | None,
    profile: str,
) -> list[str]:
    """Resolve PickUp approach cases, falling back to profile defaults."""
    names = (
        default_pickup_approach_cases_for_profile(profile)
        if not approach_cases
        else tuple(approach_cases)
    )
    if "all" in names:
        names = PICKUP_APPROACH_CASES
    return list(names)


def pickup_approach_direction_tuple(
    approach: str,
    position_case: PositionCase,
) -> tuple[float, float, float]:
    """Resolve a PickUp approach name into a normalized world-frame tuple."""
    if approach == "top":
        direction = (0.0, 0.0, -1.0)
    elif approach == "side":
        direction = (-position_case.xy[0], -position_case.xy[1], 0.0)
    else:
        raise ValueError(f"Unsupported PickUp approach case: {approach}")

    norm = math.sqrt(sum(value * value for value in direction))
    if norm < 1e-6:
        raise ValueError(f"PickUp approach direction is zero for {position_case.name}.")
    return tuple(value / norm for value in direction)


def resolve_pickup_approach_direction(
    approach: str,
    position_case: PositionCase,
    device,
):
    """Resolve a PickUp approach name into a normalized world-frame vector."""
    torch = ensure_torch()
    return torch.tensor(
        pickup_approach_direction_tuple(approach, position_case),
        dtype=torch.float32,
        device=device,
    )


def format_vector3(vector) -> str:
    """Format a 3D vector-like value for benchmark reports."""
    if hasattr(vector, "detach"):
        vector = vector.detach().to("cpu").tolist()
    return f"({float(vector[0]):.3f},{float(vector[1]):.3f},{float(vector[2]):.3f})"


def _is_horizontal_approach_direction(approach_direction) -> bool:
    """Return true when the requested approach is a side/horizontal approach."""
    if not hasattr(approach_direction, "detach"):
        return abs(float(approach_direction[2])) < 1e-4
    direction = approach_direction.detach()
    if direction.ndim > 1:
        direction = direction.reshape(-1, direction.shape[-1])[0]
    return abs(float(direction[2].to("cpu"))) < 1e-4


def create_benchmark_object(
    sim,
    preset: MeshObjectPreset,
    position_case: PositionCase,
    uid_suffix: str,
):
    """Create one benchmark object at a selected initial position."""
    from embodichain.data import get_data_path
    from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg, MeshCfg

    if preset.shape_type == "mesh":
        shape = MeshCfg(fpath=get_data_path(preset.mesh_path))
    elif preset.shape_type == "cube":
        if preset.cube_size is None:
            raise ValueError(f"Cube preset {preset.object_type!r} misses cube_size.")
        shape = CubeCfg(size=list(preset.cube_size))
    else:
        raise ValueError(
            f"Unsupported benchmark object shape_type {preset.shape_type!r}."
        )

    cfg = RigidObjectCfg(
        uid=f"benchmark_{preset.label}_{position_case.name}_{uid_suffix}",
        shape=shape,
        attrs=RigidBodyAttributesCfg(
            mass=preset.mass,
            dynamic_friction=preset.dynamic_friction,
            static_friction=preset.static_friction,
            restitution=preset.restitution,
            contact_offset=preset.contact_offset,
            rest_offset=preset.rest_offset,
            linear_damping=preset.linear_damping,
            angular_damping=preset.angular_damping,
            max_depenetration_velocity=preset.max_depenetration_velocity,
            min_position_iters=preset.min_position_iters,
            min_velocity_iters=preset.min_velocity_iters,
            max_linear_velocity=preset.max_linear_velocity,
            max_angular_velocity=preset.max_angular_velocity,
            enable_ccd=preset.enable_ccd,
        ),
        max_convex_hull_num=preset.max_convex_hull_num,
        init_pos=[position_case.xy[0], position_case.xy[1], preset.initial_z],
        init_rot=preset.init_rot,
        body_scale=preset.body_scale,
        use_usd_properties=preset.use_usd_properties,
    )
    obj = sim.add_rigid_object(cfg=cfg)
    sim.update(step=10)
    return obj


create_mesh_benchmark_object = create_benchmark_object


def _make_benchmark_antipodal_affordance_class():
    """Create an AntipodalAffordance subclass after project imports are available."""
    from embodichain.lab.sim.atomic_actions import AntipodalAffordance

    class BenchmarkAntipodalAffordance(AntipodalAffordance):
        """Benchmark affordance that biases side grasps to horizontal closing."""

        def get_valid_grasp_poses(
            self,
            obj_poses,
            approach_direction,
        ):
            results = super().get_valid_grasp_poses(
                obj_poses=obj_poses,
                approach_direction=approach_direction,
            )
            if not _is_horizontal_approach_direction(approach_direction):
                return results

            adjusted_results = []
            for grasp_poses, costs in results:
                if grasp_poses.ndim < 3 or grasp_poses.shape[0] == 0:
                    adjusted_results.append((grasp_poses, costs))
                    continue

                opening_axis_abs_z = grasp_poses[:, :3, 0].abs()[:, 2]
                keep = opening_axis_abs_z <= SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z
                if bool(keep.any()):
                    adjusted_results.append((grasp_poses[keep], costs[keep]))
                    continue

                adjusted_costs = costs + (
                    opening_axis_abs_z * SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT
                )
                adjusted_results.append((grasp_poses, adjusted_costs))
            return adjusted_results

    return BenchmarkAntipodalAffordance


def create_antipodal_object_semantics(
    obj,
    preset: MeshObjectPreset,
    args: argparse.Namespace,
    build_gripper_collision_cfg: Callable[[], object],
    build_grasp_generator_cfg: Callable[[argparse.Namespace], object],
):
    """Create object semantics with an antipodal grasp affordance."""
    from embodichain.lab.sim.atomic_actions import ObjectSemantics

    mesh_vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    mesh_triangles = obj.get_triangles(env_ids=[0])[0]
    affordance_cls = _make_benchmark_antipodal_affordance_class()
    return ObjectSemantics(
        label=preset.label,
        geometry={
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
        },
        affordance=affordance_cls(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            gripper_collision_cfg=build_gripper_collision_cfg(),
            generator_cfg=build_grasp_generator_cfg(args),
            force_reannotate=args.force_reannotate,
        ),
        entity=obj,
    )


def describe_object_preset(preset: MeshObjectPreset) -> str:
    """Describe an object preset for benchmark report notes."""
    if preset.shape_type == "cube":
        return (
            f"{preset.object_type}/{preset.material_name}/"
            f"CubeCfg(size={preset.cube_size})"
        )
    return f"{preset.object_type}/{preset.material_name}/{preset.mesh_path}"


def sync_cuda() -> None:
    """Synchronize CUDA stream when available."""
    torch = ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_gpu_memory() -> None:
    """Reset PyTorch peak GPU memory stats when CUDA is available."""
    torch = ensure_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated by PyTorch in MB."""
    torch = ensure_torch()
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def memory_snapshot() -> dict[str, float]:
    """Return current process memory usage snapshot in MB."""
    torch = ensure_torch()
    if psutil is not None:
        process = psutil.Process(os.getpid())
        cpu_mb = process.memory_info().rss / 1024**2
    else:
        cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def timed_call(
    callable_fn: Callable[[], object],
) -> tuple[float, dict[str, float], float, object]:
    """Time a callable and return elapsed seconds, memory deltas, peak GPU, result."""
    reset_peak_gpu_memory()
    before = memory_snapshot()
    sync_cuda()

    start = time.perf_counter()
    result = callable_fn()
    sync_cuda()
    elapsed = time.perf_counter() - start

    after = memory_snapshot()
    deltas = {
        "cpu_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu_mb": after["gpu_mb"] - before["gpu_mb"],
    }
    return elapsed, deltas, peak_gpu_memory_mb(), result


def reset_robot(robot, initial_qpos) -> None:
    """Reset current and target robot qpos to the benchmark initial posture."""
    for target in (False, True):
        robot.set_qpos(initial_qpos, target=target)
    robot.clear_dynamics()


def reset_rigid_object(obj, initial_pose) -> None:
    """Reset a rigid object pose and clear residual dynamics."""
    obj.set_local_pose(initial_pose)
    obj.clear_dynamics()


def reset_rigid_object_xy(
    obj,
    base_pose,
    xy: tuple[float, float],
    sim=None,
    settle_steps: int = 0,
):
    """Reset a rigid object to a new XY position while preserving base orientation."""
    pose = base_pose.clone()
    pose[:, 0, 3] = xy[0]
    pose[:, 1, 3] = xy[1]
    reset_rigid_object(obj, pose)
    if sim is not None and settle_steps > 0:
        sim.update(step=settle_steps)
    return pose


def park_rigid_object(
    obj,
    base_pose,
    index: int = 0,
    sim=None,
) -> None:
    """Move an inactive benchmark object outside the robot workspace."""
    pose = base_pose.clone()
    pose[:, 0, 3] = 8.0 + float(index)
    pose[:, 1, 3] = 8.0
    pose[:, 2, 3] = 1.0
    reset_rigid_object(obj, pose)
    if sim is not None:
        sim.update(step=1)


def should_record_case(
    args: argparse.Namespace,
    recorded_count: int,
    planning_success: bool,
) -> bool:
    """Return whether a benchmark case should emit a replay video."""
    if not getattr(args, "record_video", False):
        return False
    if not planning_success and not getattr(args, "record_failed_video", False):
        return False

    case_limit = getattr(args, "video_case_limit", DEFAULT_VIDEO_CASE_LIMIT)
    if case_limit < 0:
        raise ValueError("--video_case_limit must be non-negative.")
    return case_limit == 0 or recorded_count < case_limit


def build_video_output_path(
    args: argparse.Namespace,
    benchmark_name: str,
    case_id: str,
) -> Path:
    """Build a deterministic, timestamped output path for one replay video."""
    output_dir = Path(getattr(args, "video_dir", DEFAULT_VIDEO_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{benchmark_name}_{safe_case_id}_{timestamp}.mp4"


def summarize_video_recording(
    args: argparse.Namespace,
    results: Sequence[dict[str, object]],
    video_paths: Sequence[str],
) -> list[str]:
    """Build report notes that make video coverage explicit."""
    evaluated_count = len(results)
    success_count = sum(1 for result in results if bool(result.get("success")))
    failure_count = evaluated_count - success_count
    notes = [
        (
            "Case/video summary: "
            f"evaluated={evaluated_count}, success={success_count}, "
            f"failure={failure_count}, videos={len(video_paths)}"
        )
    ]
    if getattr(args, "record_video", False):
        if getattr(args, "record_failed_video", False):
            notes.append(
                "Video policy: records successful replays and failed-case debug "
                "videos when trajectory/static scene capture is available."
            )
        else:
            notes.append(
                "Video policy: records successful replays only; failed cases are "
                "reported in the tables but do not emit videos."
            )
    else:
        notes.append("Video policy: disabled.")
    notes.append(
        "Replay videos: " + (", ".join(video_paths) if video_paths else "disabled")
    )
    return notes


def _replay_trajectory_with_recording(
    sim,
    robot,
    traj,
    args: argparse.Namespace,
    video_path: Path,
    on_step: Callable[[int], None] | None = None,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_VIDEO_LOOK_AT,
) -> Path | None:
    """Replay a planned trajectory and record it with the simulation recorder."""
    if traj is None or getattr(traj, "ndim", 0) < 3 or traj.shape[1] == 0:
        return None

    video_fps = getattr(args, "video_fps", DEFAULT_VIDEO_FPS)
    video_max_memory = getattr(args, "video_max_memory", DEFAULT_VIDEO_MAX_MEMORY_MB)
    video_width = getattr(args, "video_width", DEFAULT_VIDEO_WIDTH)
    video_height = getattr(args, "video_height", DEFAULT_VIDEO_HEIGHT)
    video_hold_steps = getattr(args, "video_hold_steps", DEFAULT_VIDEO_HOLD_STEPS)

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    recording_started = False
    try:
        sim.sim_config.width = video_width
        sim.sim_config.height = video_height
        recording_started = sim.start_window_record(
            save_path=str(video_path),
            fps=video_fps,
            max_memory=video_max_memory,
            look_at=look_at,
            use_sim_time=True,
        )
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height

    if not recording_started:
        return None

    stop_success = False
    try:
        for waypoint_index in range(traj.shape[1]):
            robot.set_qpos(traj[:, waypoint_index, :])
            sim.update(step=4)
            if on_step is not None:
                on_step(waypoint_index)

        final_qpos = traj[:, -1, :]
        for _ in range(video_hold_steps):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
    finally:
        if sim.is_window_recording():
            stop_success = sim.stop_window_record()
        sim.wait_window_record_saves()

    return video_path if stop_success else None


def _record_static_scene_video(
    sim,
    args: argparse.Namespace,
    video_path: Path,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_VIDEO_LOOK_AT,
) -> Path | None:
    """Record the current scene without replaying a planned trajectory."""
    video_fps = getattr(args, "video_fps", DEFAULT_VIDEO_FPS)
    video_max_memory = getattr(args, "video_max_memory", DEFAULT_VIDEO_MAX_MEMORY_MB)
    video_width = getattr(args, "video_width", DEFAULT_VIDEO_WIDTH)
    video_height = getattr(args, "video_height", DEFAULT_VIDEO_HEIGHT)
    video_hold_steps = getattr(args, "video_hold_steps", DEFAULT_VIDEO_HOLD_STEPS)

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    recording_started = False
    try:
        sim.sim_config.width = video_width
        sim.sim_config.height = video_height
        recording_started = sim.start_window_record(
            save_path=str(video_path),
            fps=video_fps,
            max_memory=video_max_memory,
            look_at=look_at,
            use_sim_time=True,
        )
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height

    if not recording_started:
        return None

    stop_success = False
    try:
        for _ in range(video_hold_steps):
            sim.update(step=2)
    finally:
        if sim.is_window_recording():
            stop_success = sim.stop_window_record()
        sim.wait_window_record_saves()

    return video_path if stop_success else None


def replay_trajectory_with_recording(
    sim,
    robot,
    traj,
    args: argparse.Namespace,
    video_path: Path,
    on_step: Callable[[int], None] | None = None,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_VIDEO_LOOK_AT,
) -> Path | None:
    """Best-effort replay recording that never changes benchmark success."""
    try:
        return _replay_trajectory_with_recording(
            sim=sim,
            robot=robot,
            traj=traj,
            args=args,
            video_path=video_path,
            on_step=on_step,
            look_at=look_at,
        )
    except Exception as exc:
        try:
            if sim.is_window_recording():
                sim.stop_window_record()
            sim.wait_window_record_saves()
        except Exception:
            pass
        print(
            "Warning: failed to record benchmark replay video "
            f"{video_path}: {type(exc).__name__}: {exc}"
        )
        return None


def record_static_scene_video(
    sim,
    args: argparse.Namespace,
    video_path: Path,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_VIDEO_LOOK_AT,
) -> Path | None:
    """Best-effort static scene recording that never changes benchmark success."""
    try:
        return _record_static_scene_video(
            sim=sim,
            args=args,
            video_path=video_path,
            look_at=look_at,
        )
    except Exception as exc:
        try:
            if sim.is_window_recording():
                sim.stop_window_record()
            sim.wait_window_record_saves()
        except Exception:
            pass
        print(
            "Warning: failed to record benchmark static debug video "
            f"{video_path}: {type(exc).__name__}: {exc}"
        )
        return None


def format_float(value: float | None, precision: int = 6) -> str:
    """Format finite floats for tables and use N/A for missing values."""
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def format_markdown_table(rows: list[dict[str, object]]) -> list[str]:
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


def write_markdown_report(
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
    lines.extend(format_markdown_table(perf_rows))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(format_markdown_table(metric_rows))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(format_markdown_table(leaderboard_rows))

    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_single_action_leaderboard(
    action_name: str,
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate rows for one action by benchmark case."""
    if not metric_rows:
        return []

    success_sum = sum(float(row["success_rate"]) for row in metric_rows)
    count = len(metric_rows)
    return [
        {
            "rank": 1,
            "algorithm": action_name,
            "overall_success_rate": f"{success_sum / max(count, 1):.2%}",
            "evaluated_cases": count,
        }
    ]


__all__ = [
    "BENCHMARK_PROFILES",
    "CPU_MEMORY_BACKEND",
    "COVERAGE_MESH_OBJECT_TYPES",
    "COVERAGE_POSITION_CASE_NAMES",
    "DEFAULT_BENCHMARK_PROFILE",
    "add_common_benchmark_args",
    "add_grasp_benchmark_args",
    "add_object_position_benchmark_args",
    "add_profile_benchmark_args",
    "add_video_benchmark_args",
    "build_video_output_path",
    "build_single_action_leaderboard",
    "create_antipodal_object_semantics",
    "create_benchmark_object",
    "create_mesh_benchmark_object",
    "DEFAULT_VIDEO_DIR",
    "default_pickup_approach_cases_for_profile",
    "default_mesh_object_types_for_profile",
    "default_position_case_names_for_profile",
    "describe_object_preset",
    "ensure_repo_root",
    "ensure_torch",
    "format_float",
    "format_vector3",
    "FULL_MESH_OBJECT_TYPES",
    "FULL_POSITION_CASE_NAMES",
    "MESH_OBJECT_PRESETS",
    "MeshObjectPreset",
    "PICKUP_APPROACH_CASES",
    "POSITION_CASES",
    "PositionCase",
    "park_rigid_object",
    "pickup_approach_direction_tuple",
    "record_static_scene_video",
    "replay_trajectory_with_recording",
    "reset_rigid_object",
    "reset_rigid_object_xy",
    "reset_robot",
    "resolve_pickup_approach_direction",
    "resolve_profile",
    "select_mesh_object_presets",
    "select_pickup_approaches",
    "select_position_cases",
    "should_record_case",
    "SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z",
    "SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT",
    "SMOKE_PICKUP_APPROACH_CASES",
    "SMOKE_MESH_OBJECT_TYPES",
    "SMOKE_POSITION_CASE_NAMES",
    "timed_call",
    "summarize_video_recording",
    "write_markdown_report",
]
