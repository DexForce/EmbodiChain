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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

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
# Tolerance sets from BENCHMARK_STANDARD.md section 0. The split follows the
# binding contract rather than the skill: MoveEndEffector and MoveJoints bind
# only a motion endpoint and are scored after the terminal settle with the
# drive converged, while the skills that also bind a grasp endpoint are scored
# at the end of the actuating segment with the arm contact-loaded. Every value
# is 1.8x the worst error measured for its set, absolute, and never scaled by
# the commanded magnitude.
PRIMITIVE_POSITION_TOLERANCE_M = 0.003
PRIMITIVE_ROTATION_TOLERANCE_RAD = 0.00524
TASK_POSITION_TOLERANCE_M = 0.01
TASK_ROTATION_TOLERANCE_RAD = 0.08727
# Thresholds that are not an error against a commanded value.
PHYSICAL_PICK_MIN_LIFT_M = 0.04
PHYSICAL_PRESS_MIN_STROKE_RATIO = 0.80
PHYSICAL_DROP_MARGIN_M = 0.15
# Floor separating a target the skill actually engaged from physics noise. A
# joint the hand never touched holds its position to far less than this in the
# shipped assets, and it is two orders below every tolerance above. Radians for
# a revolute target, metres for a prismatic one.
ENGAGED_MIN_DISPLACEMENT = 1.0e-3
PHYSICAL_VALIDATION_HOLD_STEPS = 80
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
    asset_physics_mode: Literal["preserve", "overlay"] = "overlay"
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
    collision_approximation: Literal["convex_hull", "convex_decomposition"] = (
        "convex_decomposition"
    )
    max_hulls: int | None = 16
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
        asset_physics_mode="overlay",
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
        asset_physics_mode="overlay",
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
        asset_physics_mode="overlay",
        dynamic_friction=0.5,
        static_friction=0.5,
        contact_offset=0.003,
        rest_offset=0.001,
        max_depenetration_velocity=10.0,
        min_position_iters=32,
        min_velocity_iters=8,
        collision_approximation="convex_hull",
        max_hulls=None,
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
        asset_physics_mode="overlay",
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
        max_hulls=8,
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
        asset_physics_mode="overlay",
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
    from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg

    if preset.shape_type == "mesh":
        shape = MeshCfg(
            fpath=get_data_path(preset.mesh_path),
            collision=MeshCollisionCfg(
                approximation=preset.collision_approximation,
                max_hulls=preset.max_hulls,
            ),
        )
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
        attrs=RigidBodyPhysicsCfg.from_dict(
            {
                "mass_props": {"mass": preset.mass},
                "rigid_props": {
                    "linear_damping": preset.linear_damping,
                    "angular_damping": preset.angular_damping,
                    "max_depenetration_velocity": preset.max_depenetration_velocity,
                    "min_position_iters": preset.min_position_iters,
                    "min_velocity_iters": preset.min_velocity_iters,
                    "max_linear_velocity": preset.max_linear_velocity,
                    "max_angular_velocity": preset.max_angular_velocity,
                    "enable_ccd": preset.enable_ccd,
                },
                "collision_props": {
                    "contact_offset": preset.contact_offset,
                    "rest_offset": preset.rest_offset,
                },
                "material_props": {
                    "dynamic_friction": preset.dynamic_friction,
                    "static_friction": preset.static_friction,
                    "restitution": preset.restitution,
                },
            }
        ),
        init_pos=[position_case.xy[0], position_case.xy[1], preset.initial_z],
        init_rot=preset.init_rot,
        body_scale=preset.body_scale,
        asset_physics_mode=preset.asset_physics_mode,
    )
    obj = sim.add_rigid_object(cfg=cfg)
    # Adding a body changes the spawn topology; re-prepare so the new object's
    # body_data is bound before any benchmark reads its pose.
    sim.prepare()
    sim.update(step=10)
    return obj


create_mesh_benchmark_object = create_benchmark_object


def _make_benchmark_grasp_pose_generator_class():
    """Create the benchmark generator subclass after project imports are available."""
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

    class BenchmarkGraspPoseGenerator(AntipodalGraspPoseGenerator):
        """Antipodal service that biases side grasps to horizontal closing."""

        def get_valid_grasp_poses(
            self,
            **kwargs,
        ):
            results = super().get_valid_grasp_poses(**kwargs)
            approach_direction = kwargs["approach_direction"]
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

    return BenchmarkGraspPoseGenerator


def create_benchmark_grasp_pose_generator(
    args: argparse.Namespace,
):
    """Create the endpoint-owned parallel-jaw service used by benchmarks."""
    from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
        ParallelJawGraspCollisionCfg,
    )

    generator_type = _make_benchmark_grasp_pose_generator_class()
    return generator_type(
        ParallelJawGripperModelCfg(
            model_id="dh_pgi_140_80",
            min_opening_width=0.003,
            max_opening_width=0.1,
            finger_length=0.1,
            finger_width=0.04,
            finger_thickness=0.01,
            palm_depth=0.096,
        ),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(sample_count=args.n_sample),
        collision_cfg=ParallelJawGraspCollisionCfg(
            opening_margin=0.03,
            point_sample_density=0.012,
            filter_ground_collision=False,
        ),
        annotation_cfg=GraspAnnotationCfg(
            selection_mode="whole_mesh",
            viser_port=11801,
            force_refresh=args.force_reannotate,
        ),
    )


def create_antipodal_object_semantics(
    obj,
    preset: MeshObjectPreset,
):
    """Create pure object semantics with target-local antipodal geometry."""
    from embodichain.lab.sim.atomic_actions import AntipodalAffordance, ObjectSemantics

    mesh_vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    mesh_triangles = obj.get_triangles(env_ids=[0])[0]
    return ObjectSemantics(
        label=preset.label,
        geometry={
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
        },
        affordance=AntipodalAffordance(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
        ),
        entity_id=obj.uid,
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


def object_position_tuple(obj) -> tuple[float, float, float]:
    """Return the current object-frame origin position as a CPU tuple."""
    pose = obj.get_local_pose(to_matrix=True)
    xyz = pose[0, :3, 3]
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().to("cpu").tolist()
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def xy_distance_m(
    position: Sequence[float],
    target: Sequence[float],
) -> float:
    """Return XY Euclidean distance in meters."""
    return math.sqrt(
        (float(position[0]) - float(target[0])) ** 2
        + (float(position[1]) - float(target[1])) ** 2
    )


def xyz_distance_m(
    position: Sequence[float],
    target: Sequence[float],
) -> float:
    """Return XYZ Euclidean distance in meters."""
    return math.sqrt(
        (float(position[0]) - float(target[0])) ** 2
        + (float(position[1]) - float(target[1])) ** 2
        + (float(position[2]) - float(target[2])) ** 2
    )


def replay_trajectory_for_physical_validation(
    sim,
    robot,
    obj,
    traj,
    on_step: Callable[[int], None] | None = None,
    hold_steps: int = PHYSICAL_VALIDATION_HOLD_STEPS,
) -> tuple[float, float, float] | None:
    """Replay a trajectory in physics and return the final object position.

    The replay runs at :data:`DEFAULT_REPLAY_STEPS_PER_WAYPOINT`, the rate the
    standard calibrates every scene measurement at, so an object read here is
    comparable with one read through :func:`replay_and_track_channels`.
    """
    if traj is None or getattr(traj, "ndim", 0) < 3 or traj.shape[1] == 0:
        return None

    for waypoint_index in range(traj.shape[1]):
        robot.set_qpos(traj[:, waypoint_index, :])
        sim.update(step=DEFAULT_REPLAY_STEPS_PER_WAYPOINT)
        if on_step is not None:
            on_step(waypoint_index)

    final_qpos = traj[:, -1, :]
    for _ in range(hold_steps):
        robot.set_qpos(final_qpos)
        sim.update(step=2)
    return object_position_tuple(obj)


def should_record_case(
    args: argparse.Namespace,
    recorded_count: int,
    success: bool,
) -> bool:
    """Return whether a benchmark case should emit a replay video."""
    if not getattr(args, "record_video", False):
        return False
    if not success and not getattr(args, "record_failed_video", False):
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
                "Video policy: records report-success replays and failed-case "
                "debug videos when trajectory/static scene capture is available."
            )
        else:
            notes.append(
                "Video policy: records report-success replays only; failed "
                "cases are reported in the tables but do not emit videos."
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


# Task-level subset of the taxonomy in
# scripts/benchmark/motion_generation/BENCHMARK_DESIGN.md section 7. The
# trajectory-level reasons stay with that suite: these benchmarks measure
# skills, so neither a plan's validity nor the drive's tracking error fails one
# here. Both are reported as diagnostics instead.
FAILURE_TAXONOMY: tuple[str, ...] = (
    "invalid_case",
    "unsupported_capability",
    "planner_exception",
    "planner_reported_failure",
    "timeout",
    "object_not_grasped",
    "object_dropped",
    "release_failure",
    "task_goal_miss",
)
# Ordered stages per skill, from BENCHMARK_STANDARD.md section 1. Each stage is
# a scene measurement taken at the end of the segment where the skill stops
# acting on what the stage is about, and a stage is reached only when every
# earlier stage passed.
SKILL_STAGES: dict[str, tuple[str, ...]] = {
    "move_end_effector": ("reached",),
    "move_joints": ("reached",),
    "pick_up": ("grasped", "lifted", "held"),
    "axis_align": ("grasped", "aligned", "held"),
    "move_held_object": ("transported",),
    "place": ("released", "placed", "stable"),
    "pour": ("poured",),
    "push_object": ("contacted", "pushed", "on_support_plane"),
    "press": ("contacted", "pressed"),
    "slide": ("grasped", "slid", "released"),
    "open_door": ("grasped", "opened", "released"),
    "twist": ("grasped", "twisted", "released"),
    "coordinated_pickment": ("dual_grasped", "lifted", "moved", "held"),
    "coordinated_placement": ("aligned", "released", "stable"),
    "hand_over": ("grasped", "transferred", "handed_over", "placed"),
}
JOINT_LIMIT_TOLERANCE_RAD = 1e-3
# Physics steps per replayed waypoint. The planners emit one waypoint per
# physics step, but the simulated position drive cannot converge that fast: a
# measured sweep on OpenDoor gave max arm tracking errors of 1.38/0.92/0.44/
# 0.16/0.05/0.02 rad at 1/2/4/8/16/32 steps per waypoint, and the measured
# hinge error tracked it (0.23/0.13/0.07/0.05/0.03/0.02 rad). At 16 steps the
# target-joint measurement has converged to within 0.006 rad of the 32-step
# value, so the replay measures the skill rather than the drive.
DEFAULT_REPLAY_STEPS_PER_WAYPOINT = 16


@dataclass
class StageLadder:
    """Ordered per-skill stages for one benchmark case.

    The stages come from :data:`SKILL_STAGES`, which mirrors
    ``BENCHMARK_STANDARD.md`` section 1. A stage is reached only when every
    earlier stage passed, so a report localizes the failure instead of
    collapsing it into one boolean, and a closing stage counts: a skill that
    lifts an object and then drops it did not pick it up.

    Trajectory validity and replay tracking ride along as diagnostics.
    Neither fails a stage; they belong to
    ``scripts/benchmark/motion_generation/``.
    """

    stages: tuple[str, ...]
    passed: dict[str, bool] = field(default_factory=dict)
    failure_stage: str = ""
    failure_reason: str = ""
    motion_valid: bool | None = None
    motion_detail: str = ""
    max_tracking_error_rad: float | None = None

    def __post_init__(self) -> None:
        """Validate that the case declares at least one stage."""
        if not self.stages:
            raise ValueError("A skill must declare at least one stage.")

    @property
    def success(self) -> bool:
        """Whether every declared stage passed."""
        return all(self.passed.get(stage, False) for stage in self.stages)

    def reached(self, stage: str) -> bool:
        """Whether ``stage`` was reached, i.e. every earlier stage passed."""
        index = self.stages.index(stage)
        return all(self.passed.get(name, False) for name in self.stages[:index])

    def record(self, stage: str, passed: bool, reason: str = "") -> "StageLadder":
        """Record the next stage's outcome in declaration order.

        Calls after the first failure are ignored: those stages were never
        reached, and recording them would credit or blame a measurement the
        skill never got to make.

        Args:
            stage: Stage name, which must be the next unrecorded stage.
            passed: Whether the stage's scene measurement passed.
            reason: Failure reason from :data:`FAILURE_TAXONOMY`, required
                when ``passed`` is False.

        Returns:
            This ladder, to allow chained calls.
        """
        if stage not in self.stages:
            raise ValueError(f"Unknown stage {stage!r} for {self.stages}.")
        if self.failure_stage:
            return self
        expected = self.stages[len(self.passed)]
        if stage != expected:
            raise ValueError(
                f"Stage {stage!r} recorded out of order; expected {expected!r}."
            )
        self.passed[stage] = bool(passed)
        if not passed:
            if reason not in FAILURE_TAXONOMY:
                raise ValueError(f"Unknown failure reason {reason!r}.")
            self.failure_stage = stage
            self.failure_reason = reason
        return self

    def fail(self, stage: str, reason: str) -> "StageLadder":
        """Record ``stage`` as failed with a taxonomy reason."""
        return self.record(stage, False, reason)

    def unsupported(self) -> "StageLadder":
        """Record that this embodiment cannot serve the case.

        No stage is marked failed, because none was measured: the case leaves
        every denominator and lowers the suite's coverage instead.
        """
        self.failure_reason = "unsupported_capability"
        return self

    def as_row_fields(self) -> dict[str, object]:
        """Return this case's stage columns plus the shared diagnostics.

        An unreached stage is ``N/A`` rather than zero, so a stage rate is
        never diluted by cases that never got there.
        """
        fields: dict[str, object] = {}
        for stage in self.stages:
            if stage in self.passed:
                fields[stage] = f"{float(self.passed[stage]):.6f}"
            else:
                fields[stage] = "N/A"
        fields["success"] = f"{float(self.success):.6f}"
        fields["failure_stage"] = self.failure_stage or "N/A"
        fields["failure_reason"] = self.failure_reason or "N/A"
        fields["motion_valid"] = (
            "N/A" if self.motion_valid is None else f"{float(self.motion_valid):.6f}"
        )
        fields["max_tracking_error_rad"] = format_float(self.max_tracking_error_rad, 4)
        return fields


def dropped_below_support(min_height_m: float, support_height_m: float) -> bool:
    """Whether a held object fell out of the skill's control during a replay.

    Cross-cutting rule from ``BENCHMARK_STANDARD.md`` section 1: the minimum
    height over the whole replay is used, not the final one, because an object
    dropped mid-motion can still roll to rest near the target.

    Args:
        min_height_m: Lowest world z the object reached during the replay.
        support_height_m: Height of the support plane it was manipulated over.

    Returns:
        True when the object fell more than :data:`PHYSICAL_DROP_MARGIN_M`
        below the support plane at any point.
    """
    if not math.isfinite(min_height_m) or not math.isfinite(support_height_m):
        return False
    return (support_height_m - min_height_m) > PHYSICAL_DROP_MARGIN_M


def hand_is_released(robot, hand_control_part: str, open_qpos, close_qpos) -> bool:
    """Whether a gripper ended a replay nearer its open command than its closed one.

    Release is a scene measurement like any other stage: the standard asks
    whether the hand let go, not whether a release segment was planned. The
    comparison is against the skill's own two commands, so it holds for any
    gripper without a per-asset threshold.

    Args:
        robot: Robot holding the gripper.
        hand_control_part: Name of the gripper control part.
        open_qpos: Open command used by the skill for that part.
        close_qpos: Closed command used by the skill for that part.

    Returns:
        True when the achieved hand position is closer to the open command.
    """
    achieved = robot.get_qpos(name=hand_control_part, target=False)[0]
    to_open = float((achieved - open_qpos.to(achieved.device)).abs().max())
    to_close = float((achieved - close_qpos.to(achieved.device)).abs().max())
    return to_open < to_close


def release_simulation(sim=None) -> None:
    """Tear down a simulation so the next one can build its own.

    The simulator is a singleton keyed by instance id, and a planner resolves
    its robot through that instance. A benchmark that runs after another in the
    same process therefore has to start from a released singleton, or it builds
    its scene as a second instance and then fails to find its own robot.

    Args:
        sim: Simulation manager to release, or None to release the current
            instance if there is one.
    """
    from embodichain.lab.sim import SimulationManager

    if not SimulationManager.is_instantiated():
        return
    if sim is None:
        sim = SimulationManager.get_instance()
    if not getattr(sim, "_is_constructed", False):
        SimulationManager.reset(getattr(sim, "instance_id", 0))
        return
    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()


GRASP_ATTAINABILITY_MAX_CANDIDATES = 8
GRASP_ATTAINABILITY_TOLERANCE_RAD = 0.05
GRASP_ATTAINABILITY_HOLD_ITERATIONS = 40
GRASP_ATTAINABILITY_HOLD_STEPS = 4


def _drive_arm_to(sim, robot, qpos, control_part: str) -> float:
    """Command one arm configuration and return how far the arm stops short."""
    for _ in range(GRASP_ATTAINABILITY_HOLD_ITERATIONS):
        robot.set_qpos(qpos, name=control_part, target=True)
        sim.update(step=GRASP_ATTAINABILITY_HOLD_STEPS)
    achieved = robot.get_qpos(name=control_part, target=False)
    return float((achieved - qpos).abs().max())


def has_attainable_grasp_candidate(
    sim,
    robot,
    semantics,
    grasp_pose_generator,
    object_pose,
    approach_direction,
    pre_grasp_distance: float,
    control_part: str = "arm",
) -> bool:
    """Whether the robot can physically reach any sampled grasp on this object.

    An inverse-kinematics solution only says a joint configuration exists. The
    arm still has to get there: an object lying against the ground cannot be
    grasped by a gripper held horizontally at its height, and an object close
    to the base sits inside the arm's inner workspace, where every branch folds
    into something the drive cannot attain. Both produce plans that execute
    into thin air, which is a property of this embodiment and scene rather than
    a skill failure, so the caller reports those cases as
    ``unsupported_capability``.

    Each candidate is driven in physics from the current state, pre-grasp then
    grasp, and the first one that arrives ends the search. The robot state is
    restored before returning.

    Args:
        sim: Simulation manager to step.
        robot: Robot whose arm is tested.
        semantics: Object semantics carrying the antipodal affordance.
        grasp_pose_generator: Grasp pose service used by the benchmark.
        object_pose: Batched object pose with shape ``(num_envs, 4, 4)``.
        approach_direction: Unit approach vector with shape ``(3,)``.
        pre_grasp_distance: Stand-off distance along the approach direction.
        control_part: Name of the arm control part.

    Returns:
        True when at least one sampled grasp is physically attainable.
    """
    initial_qpos = robot.get_qpos().clone()
    start_qpos = robot.get_qpos(name=control_part).clone()
    candidates = semantics.affordance.get_grasp_candidates(
        grasp_pose_generator, object_pose, approach_direction
    )
    poses = candidates.poses[0]
    valid = candidates.valid[0]

    attainable = False
    tested = 0
    for index in range(poses.shape[0]):
        if tested >= GRASP_ATTAINABILITY_MAX_CANDIDATES:
            break
        if not bool(valid[index]):
            continue
        grasp_pose = poses[index : index + 1]
        pre_grasp_pose = grasp_pose.clone()
        pre_grasp_pose[..., :3, 3] -= approach_direction * pre_grasp_distance
        pre_grasp_ok, pre_grasp_qpos = robot.compute_ik(
            pose=pre_grasp_pose, joint_seed=start_qpos, name=control_part
        )
        if not bool(pre_grasp_ok.all()):
            continue
        grasp_ok, grasp_qpos = robot.compute_ik(
            pose=grasp_pose, joint_seed=pre_grasp_qpos, name=control_part
        )
        if not bool(grasp_ok.all()):
            continue
        tested += 1
        robot.set_qpos(initial_qpos, target=False)
        robot.set_qpos(initial_qpos, target=True)
        robot.clear_dynamics()
        sim.update(step=GRASP_ATTAINABILITY_HOLD_STEPS)
        _drive_arm_to(sim, robot, pre_grasp_qpos, control_part)
        error = _drive_arm_to(sim, robot, grasp_qpos, control_part)
        if error < GRASP_ATTAINABILITY_TOLERANCE_RAD:
            attainable = True
            break

    robot.set_qpos(initial_qpos, target=False)
    robot.set_qpos(initial_qpos, target=True)
    robot.clear_dynamics()
    sim.update(step=GRASP_ATTAINABILITY_HOLD_STEPS)
    return attainable


def check_motion_valid(traj, robot) -> tuple[bool, str]:
    """Check a planned trajectory as a diagnostic, not as a success gate.

    A trajectory is valid when it is non-empty, entirely finite, and stays
    inside the robot's joint position limits. Whether a plan is valid belongs
    to ``scripts/benchmark/motion_generation/``; here the answer is reported
    alongside the stages so a reader can judge a scene measurement, and only an
    unreplayable trajectory stops a case.

    Args:
        traj: Planned positions with shape ``(num_envs, num_waypoints, dof)``.
        robot: Robot providing ``get_qpos_limits``.

    Returns:
        ``(motion_valid, detail)``; the detail names the diagnostic that fired
        and is empty when the trajectory is valid.
    """
    torch = ensure_torch()
    if traj is None or getattr(traj, "ndim", 0) < 3 or traj.shape[1] == 0:
        return False, "non_finite_trajectory"
    if not bool(torch.isfinite(traj).all()):
        return False, "non_finite_trajectory"

    try:
        limits = robot.get_qpos_limits()
    except Exception:
        # Without limits the finite check above is the strongest available
        # statement; do not fabricate a limit violation.
        return True, ""

    dof = traj.shape[-1]
    if limits.shape[1] < dof:
        return True, ""
    lower = limits[0, :dof, 0].to(traj.device)
    upper = limits[0, :dof, 1].to(traj.device)
    # Unlimited joints are reported as non-finite bounds; ignore them.
    valid = torch.isfinite(lower) & torch.isfinite(upper)
    if not bool(valid.any()):
        return True, ""
    below = traj[..., valid] < (lower[valid] - JOINT_LIMIT_TOLERANCE_RAD)
    above = traj[..., valid] > (upper[valid] + JOINT_LIMIT_TOLERANCE_RAD)
    if bool(below.any()) or bool(above.any()):
        return False, "joint_limit_violation"
    return True, ""


@dataclass(frozen=True)
class JointDisplacementTrace:
    """Signed displacement of one articulation joint across a physical replay.

    Displacement is measured relative to the joint position recorded before the
    replay starts, so it is independent of the asset's zero convention.

    ``measured_position`` is the value at the caller-selected evaluation
    waypoint. Contact skills must be scored while the robot still controls the
    target joint: once the hand releases and the arm retracts, an undriven
    hinge, drawer, or knob is free to rebound, and the post-replay resting
    value describes the asset's dynamics rather than the skill's achievement.
    ``settled_position`` retains that resting value for diagnostics, and
    ``min_position`` and ``max_position`` bound the whole replay, which is what
    the cross-cutting drop rule needs: an object dropped mid-motion can still
    roll to rest near the target.
    """

    initial_position: float
    measured_position: float
    settled_position: float
    measured_displacement: float
    settled_displacement: float
    peak_signed_displacement: float
    min_position: float = 0.0
    max_position: float = 0.0
    max_tracking_error_rad: float = 0.0


def waypoint_step_counts(
    waypoint_dt,
    physics_dt: float,
    waypoint_count: int,
    fallback_steps: int = 4,
) -> list[int]:
    """Resolve physics steps per waypoint from a planned trajectory's timing.

    Contact skills are sensitive to replay rate: holding each commanded
    waypoint longer than the planner intended keeps pushing a position
    controlled arm into the contact, which over-actuates the manipulated
    joint. This mirrors the step derivation used by the tutorial replay so
    benchmark replays execute at the planned speed.

    Args:
        waypoint_dt: Per-waypoint arrival intervals for one environment, or
            None to fall back to a fixed rate.
        physics_dt: Simulation physics timestep in seconds.
        waypoint_count: Number of trajectory waypoints.
        fallback_steps: Steps per waypoint used when timing is unavailable.

    Returns:
        Physics step counts, one per waypoint, each at least one.
    """
    if waypoint_dt is None or physics_dt <= 0.0:
        return [fallback_steps] * waypoint_count

    counts: list[int] = []
    for index in range(waypoint_count):
        next_index = min(index + 1, waypoint_count - 1)
        try:
            duration = float(waypoint_dt[next_index])
        except (IndexError, TypeError, ValueError):
            counts.append(fallback_steps)
            continue
        ratio = duration / physics_dt
        nearest = round(ratio)
        if math.isclose(ratio, nearest, rel_tol=1.0e-6, abs_tol=1.0e-9):
            counts.append(max(1, int(nearest)))
        else:
            counts.append(max(1, math.ceil(ratio)))
    return counts


def replay_and_track_channels(
    sim,
    robot,
    traj,
    readers: dict[str, Callable[[int], float]],
    measure_waypoint: int | None = None,
    waypoint_dt=None,
    physics_dt: float = 0.0,
    steps_per_waypoint: int = DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    hold_steps: int = 60,
    hold_substeps: int = 2,
) -> dict[str, JointDisplacementTrace] | None:
    """Replay a trajectory once while sampling several scalar measurements.

    This is the shared physical-replay primitive for every skill benchmark. A
    skill that is scored on one signal and reports another as a diagnostic --
    Twist, scored on the executed end-effector rotation while the knob joint is
    only reported -- reads both from a single replay, so the two describe the
    same execution.

    Args:
        sim: Simulation manager to step.
        robot: Robot receiving the replayed joint positions.
        traj: Planned positions with shape ``(num_envs, num_waypoints, dof)``.
        readers: Named callables returning one scalar each at the current
            simulation state, given the waypoint index just executed. The index
            is ``-1`` before the first waypoint and ``waypoint_count`` during
            the hold, which lets a skill capture a reference at a segment
            boundary.
        measure_waypoint: Waypoint index scored as the skill's achievement.
            Defaults to the final waypoint.
        waypoint_dt: Planned per-waypoint arrival intervals for one
            environment. When given, the replay runs at the planned rate.
        physics_dt: Simulation physics timestep, required with ``waypoint_dt``.
        steps_per_waypoint: Fallback physics steps per waypoint when no
            planned timing is available.
        hold_steps: Terminal hold iterations after the last waypoint.
        hold_substeps: Physics steps per terminal hold iteration.

    Returns:
        One trace per reader, or None when the trajectory is unusable.
    """
    if traj is None or getattr(traj, "ndim", 0) < 3 or traj.shape[1] == 0:
        return None
    if not readers:
        raise ValueError("At least one reader is required.")

    waypoint_count = int(traj.shape[1])
    if measure_waypoint is None:
        measure_waypoint = waypoint_count - 1
    measure_waypoint = max(0, min(int(measure_waypoint), waypoint_count - 1))

    step_counts = waypoint_step_counts(
        waypoint_dt, physics_dt, waypoint_count, steps_per_waypoint
    )

    initial = {name: read(-1) for name, read in readers.items()}
    samples = {name: [value] for name, value in initial.items()}
    measured = dict(initial)
    max_tracking_error = 0.0
    for waypoint_index in range(waypoint_count):
        commanded = traj[:, waypoint_index, :]
        robot.set_qpos(commanded)
        sim.update(step=step_counts[waypoint_index])
        # Verify the arm actually reached the commanded waypoint. Scoring the
        # scene is only meaningful when the executed motion matches the planned
        # one; a large error here means the replay measured the controller.
        achieved = robot.get_qpos(target=False)
        error = float((achieved - commanded).abs().max())
        max_tracking_error = max(max_tracking_error, error)
        for name, read in readers.items():
            value = read(waypoint_index)
            samples[name].append(value)
            if waypoint_index == measure_waypoint:
                measured[name] = value

    final_qpos = traj[:, -1, :]
    for _ in range(hold_steps):
        robot.set_qpos(final_qpos)
        sim.update(step=hold_substeps)
        for name, read in readers.items():
            samples[name].append(read(waypoint_count))

    traces: dict[str, JointDisplacementTrace] = {}
    for name, values in samples.items():
        displacements = [value - initial[name] for value in values]
        traces[name] = JointDisplacementTrace(
            initial_position=initial[name],
            measured_position=measured[name],
            settled_position=values[-1],
            measured_displacement=measured[name] - initial[name],
            settled_displacement=values[-1] - initial[name],
            peak_signed_displacement=max(displacements, key=abs),
            min_position=min(values),
            max_position=max(values),
            max_tracking_error_rad=max_tracking_error,
        )
    return traces


def replay_and_track_scalar(
    sim,
    robot,
    traj,
    read_value: Callable[[int], float],
    measure_waypoint: int | None = None,
    waypoint_dt=None,
    physics_dt: float = 0.0,
    steps_per_waypoint: int = DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    hold_steps: int = 60,
    hold_substeps: int = 2,
) -> JointDisplacementTrace | None:
    """Replay a trajectory while sampling one scalar scene measurement.

    Args:
        sim: Simulation manager to step.
        robot: Robot receiving the replayed joint positions.
        traj: Planned positions with shape ``(num_envs, num_waypoints, dof)``.
        read_value: Returns the tracked scalar at the current simulation state,
            given the waypoint index just executed.
        measure_waypoint: Waypoint index scored as the skill's achievement.
            Defaults to the final waypoint.
        waypoint_dt: Planned per-waypoint arrival intervals for one
            environment. When given, the replay runs at the planned rate.
        physics_dt: Simulation physics timestep, required with ``waypoint_dt``.
        steps_per_waypoint: Fallback physics steps per waypoint when no
            planned timing is available.
        hold_steps: Terminal hold iterations after the last waypoint.
        hold_substeps: Physics steps per terminal hold iteration.

    Returns:
        The measurement trace, or None when the trajectory is unusable.
    """
    traces = replay_and_track_channels(
        sim=sim,
        robot=robot,
        traj=traj,
        readers={"value": read_value},
        measure_waypoint=measure_waypoint,
        waypoint_dt=waypoint_dt,
        physics_dt=physics_dt,
        steps_per_waypoint=steps_per_waypoint,
        hold_steps=hold_steps,
        hold_substeps=hold_substeps,
    )
    return None if traces is None else traces["value"]


def replay_and_track_joint(
    sim,
    robot,
    traj,
    articulation,
    joint_index: int,
    measure_waypoint: int | None = None,
    waypoint_dt=None,
    physics_dt: float = 0.0,
    steps_per_waypoint: int = DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    hold_steps: int = 60,
    hold_substeps: int = 2,
) -> JointDisplacementTrace | None:
    """Replay a trajectory in physics while tracking one articulation joint.

    Both the actuation waypoints and the terminal hold are sampled, so a joint
    that reaches its extreme mid-motion and then rebounds (a button, a
    free-swinging door) still reports the peak it actually attained.

    Args:
        sim: Simulation manager to step.
        robot: Robot receiving the replayed joint positions.
        traj: Planned positions with shape ``(num_envs, num_waypoints, dof)``.
        articulation: Articulation owning the tracked joint.
        joint_index: Index of the tracked joint in ``articulation``.
        measure_waypoint: Waypoint index scored as the skill's achievement.
            Defaults to the final waypoint.
        waypoint_dt: Planned per-waypoint arrival intervals for one
            environment. When given, the replay runs at the planned rate.
        physics_dt: Simulation physics timestep, required with ``waypoint_dt``.
        steps_per_waypoint: Fallback physics steps per waypoint when no
            planned timing is available.
        hold_steps: Terminal hold iterations after the last waypoint.
        hold_substeps: Physics steps per terminal hold iteration.

    Returns:
        The displacement trace, or None when the trajectory is unusable.
    """

    def read(waypoint_index: int) -> float:
        del waypoint_index
        return float(articulation.get_qpos(target=False)[0, joint_index])

    return replay_and_track_scalar(
        sim=sim,
        robot=robot,
        traj=traj,
        read_value=read,
        measure_waypoint=measure_waypoint,
        waypoint_dt=waypoint_dt,
        physics_dt=physics_dt,
        steps_per_waypoint=steps_per_waypoint,
        hold_steps=hold_steps,
        hold_substeps=hold_substeps,
    )


@dataclass
class ContactCaseOutcome:
    """Result of running one articulated-contact case through its stages."""

    ladder: StageLadder
    trace: JointDisplacementTrace | None = None
    channels: dict[str, JointDisplacementTrace] = field(default_factory=dict)
    elapsed_s: float = 0.0
    cpu_delta_mb: float = 0.0
    gpu_delta_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    trajectory = None

    @property
    def trajectory_waypoints(self) -> int:
        """Number of planned waypoints, or zero when planning failed."""
        traj = self.trajectory
        if traj is None or getattr(traj, "ndim", 0) < 3:
            return 0
        return int(traj.shape[1])


def run_articulated_contact_case(
    sim,
    robot,
    articulation,
    atomic_engine,
    build_invocation: Callable[[], tuple[object, object]],
    joint_index: int,
    actuation_segment: str,
    stages: tuple[str, ...],
    evaluate: Callable[[StageLadder, dict[str, JointDisplacementTrace]], None],
    build_extra_readers: (
        Callable[[object], dict[str, Callable[[int], float]]] | None
    ) = None,
    steps_per_waypoint: int = DEFAULT_REPLAY_STEPS_PER_WAYPOINT,
    hold_steps: int = 60,
) -> ContactCaseOutcome:
    """Run one contact skill through planning, physical replay, and its stages.

    The target joint is scored at the last waypoint of ``actuation_segment``,
    the point where the skill stops commanding the joint. Later segments open
    the hand and retract the arm, after which an undriven hinge, drawer, or
    knob moves under its own dynamics rather than the skill's control.

    Trajectory validity and the replay's peak tracking error are recorded on
    the ladder as diagnostics. Neither fails the skill: a skill that opened the
    door is not failed because the drive lagged the plan.

    Args:
        sim: Simulation manager to step.
        robot: Robot executing the trajectory.
        articulation: Articulation owning the manipulated joint.
        atomic_engine: Engine used to compile the invocation.
        build_invocation: Returns the ``(invocation, context)`` pair to compile.
        joint_index: Index of the manipulated joint in ``articulation``.
        actuation_segment: Segment name whose end is scored.
        stages: Ordered stage names for this skill, from :data:`SKILL_STAGES`.
        evaluate: Records each stage on the ladder from the replay traces,
            which arrive as a name-to-trace mapping including ``"joint"``.
        build_extra_readers: Given the compiled result, returns additional
            named scalars to sample during the same replay. A skill scored on
            one signal that reports another -- or one whose reference is a
            segment boundary it can only know after compiling -- builds its
            readers here.
        steps_per_waypoint: Physics steps per replayed waypoint.
        hold_steps: Terminal hold iterations after the last waypoint.

    Returns:
        The stage result, displacement trace, and cost metrics for the case.
    """
    ladder = StageLadder(stages=stages)
    outcome = ContactCaseOutcome(ladder=ladder)
    first_stage = stages[0]

    invocation, context = build_invocation()
    try:
        elapsed, mem_delta, peak_gpu, result = timed_call(
            lambda: atomic_engine.compile((invocation,), context)
        )
    except Exception as exc:
        print(f"    planner exception: {type(exc).__name__}: {exc}")
        ladder.fail(first_stage, "planner_exception")
        return outcome

    outcome.elapsed_s = elapsed
    outcome.cpu_delta_mb = mem_delta["cpu_mb"]
    outcome.gpu_delta_mb = mem_delta["gpu_mb"]
    outcome.peak_gpu_mb = peak_gpu

    if not bool(result.plan_success.all().item()):
        ladder.fail(first_stage, "planner_reported_failure")
        return outcome

    traj = result.trajectory.positions
    outcome.trajectory = traj
    motion_valid, motion_detail = check_motion_valid(traj, robot)
    ladder.motion_valid = motion_valid
    ladder.motion_detail = motion_detail
    if motion_detail == "non_finite_trajectory":
        # There is nothing to replay, so no stage can be measured. The planner
        # reported success and returned an unusable trajectory, which is the
        # closest task-level reason the taxonomy carries.
        ladder.fail(first_stage, "planner_reported_failure")
        return outcome

    try:
        measure_waypoint = result.segment(0, actuation_segment).stop - 1
    except (KeyError, AttributeError, IndexError) as exc:
        # Falling back to the final waypoint would score the target after the
        # hand released and the arm retracted, which is the rebound this
        # function exists to avoid. A missing segment is a case definition
        # error, not a skill failure.
        print(f"    missing segment {actuation_segment!r}: {type(exc).__name__}: {exc}")
        ladder.fail(first_stage, "invalid_case")
        return outcome

    def read_joint(waypoint_index: int) -> float:
        del waypoint_index
        return float(articulation.get_qpos(target=False)[0, joint_index])

    readers: dict[str, Callable[[int], float]] = {"joint": read_joint}
    if build_extra_readers is not None:
        readers.update(build_extra_readers(result))
    try:
        traces = replay_and_track_channels(
            sim=sim,
            robot=robot,
            traj=traj,
            readers=readers,
            measure_waypoint=measure_waypoint,
            steps_per_waypoint=steps_per_waypoint,
            hold_steps=hold_steps,
        )
    except Exception as exc:
        print(f"    replay exception: {type(exc).__name__}: {exc}")
        traces = None

    trace = None if traces is None else traces["joint"]
    outcome.trace = trace
    outcome.channels = dict(traces or {})
    if trace is None:
        ladder.fail(first_stage, "invalid_case")
        return outcome

    ladder.max_tracking_error_rad = trace.max_tracking_error_rad
    evaluate(ladder, outcome.channels)
    return outcome


def warmup_planning(callable_fn: Callable[[], object]) -> bool:
    """Run one discarded planning call so timings exclude first-call cost.

    The first compile in a process pays Warp/torch kernel compilation and
    allocator warm-up, which would otherwise be charged to whichever case
    happens to run first and make cases incomparable.

    Args:
        callable_fn: Planning call to execute and discard.

    Returns:
        True when the warm-up call completed without raising.
    """
    try:
        callable_fn()
        sync_cuda()
        return True
    except Exception as exc:
        print(f"Warning: benchmark warm-up call failed: {type(exc).__name__}: {exc}")
        return False


def build_stage_leaderboard(
    action_name: str,
    results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-stage rates for one action into a leaderboard row.

    Follows ``BENCHMARK_STANDARD.md`` section 1::

        stage_success_rate[i] = passed(i) / reached(i)   reached(i) = passed(i-1)
        success_rate          = passed(last) / measured

    A case the embodiment cannot serve is recorded with
    ``unsupported_capability`` and leaves every denominator, exactly as
    ``motion_generation/BENCHMARK_DESIGN.md`` section 5 prescribes; it lowers
    ``coverage_rate`` instead, so selective execution cannot improve a result.
    A stage nobody reached reports ``N/A`` rather than zero.

    Args:
        action_name: Atomic action identifier.
        results: Case results each carrying a ``ladder`` entry.

    Returns:
        A single-row leaderboard reporting coverage, every stage rate and the
        overall success rate.
    """
    if not results:
        return []

    ladders = [result["ladder"] for result in results]
    measured = [
        ladder
        for ladder in ladders
        if ladder.failure_reason != "unsupported_capability"
    ]
    stages = ladders[0].stages
    row: dict[str, object] = {"rank": 1, "algorithm": action_name}
    row["coverage_rate"] = f"{len(measured) / len(ladders):.2%}"
    for index, stage in enumerate(stages):
        earlier = stages[:index]
        reached = sum(
            1
            for ladder in measured
            if all(ladder.passed.get(name, False) for name in earlier)
        )
        passed = sum(1 for ladder in measured if ladder.passed.get(stage, False))
        row[f"{stage}_rate"] = f"{passed / reached:.2%}" if reached else "N/A"
    row["success_rate"] = (
        f"{sum(1 for l in measured if l.success) / len(measured):.2%}"
        if measured
        else "N/A"
    )
    row["evaluated_cases"] = len(measured)
    row["unsupported_cases"] = len(ladders) - len(measured)
    return [row]


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
    "build_stage_leaderboard",
    "build_single_action_leaderboard",
    "check_motion_valid",
    "dropped_below_support",
    "has_attainable_grasp_candidate",
    "hand_is_released",
    "ContactCaseOutcome",
    "create_antipodal_object_semantics",
    "create_benchmark_object",
    "create_mesh_benchmark_object",
    "DEFAULT_VIDEO_DIR",
    "default_pickup_approach_cases_for_profile",
    "default_mesh_object_types_for_profile",
    "default_position_case_names_for_profile",
    "describe_object_preset",
    "ensure_repo_root",
    "ENGAGED_MIN_DISPLACEMENT",
    "ensure_torch",
    "FAILURE_TAXONOMY",
    "format_float",
    "format_vector3",
    "FULL_MESH_OBJECT_TYPES",
    "FULL_POSITION_CASE_NAMES",
    "JointDisplacementTrace",
    "JOINT_LIMIT_TOLERANCE_RAD",
    "PRIMITIVE_POSITION_TOLERANCE_M",
    "PRIMITIVE_ROTATION_TOLERANCE_RAD",
    "MESH_OBJECT_PRESETS",
    "MeshObjectPreset",
    "PICKUP_APPROACH_CASES",
    "PHYSICAL_DROP_MARGIN_M",
    "PHYSICAL_PICK_MIN_LIFT_M",
    "PHYSICAL_PRESS_MIN_STROKE_RATIO",
    "PHYSICAL_VALIDATION_HOLD_STEPS",
    "POSITION_CASES",
    "PositionCase",
    "object_position_tuple",
    "park_rigid_object",
    "pickup_approach_direction_tuple",
    "record_static_scene_video",
    "release_simulation",
    "replay_and_track_channels",
    "replay_and_track_joint",
    "replay_and_track_scalar",
    "replay_trajectory_for_physical_validation",
    "replay_trajectory_with_recording",
    "reset_rigid_object",
    "reset_rigid_object_xy",
    "reset_robot",
    "resolve_pickup_approach_direction",
    "run_articulated_contact_case",
    "resolve_profile",
    "select_mesh_object_presets",
    "select_pickup_approaches",
    "select_position_cases",
    "should_record_case",
    "SIDE_GRASP_MAX_OPEN_AXIS_ABS_Z",
    "SKILL_STAGES",
    "SIDE_GRASP_OPEN_AXIS_Z_COST_WEIGHT",
    "SMOKE_PICKUP_APPROACH_CASES",
    "SMOKE_MESH_OBJECT_TYPES",
    "SMOKE_POSITION_CASE_NAMES",
    "StageLadder",
    "TASK_POSITION_TOLERANCE_M",
    "TASK_ROTATION_TOLERANCE_RAD",
    "timed_call",
    "warmup_planning",
    "waypoint_step_counts",
    "summarize_video_recording",
    "write_markdown_report",
    "xy_distance_m",
    "xyz_distance_m",
]
