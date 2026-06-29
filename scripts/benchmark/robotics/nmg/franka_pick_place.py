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

"""Benchmark Franka pick-place atomic actions with IK and NMG planners.

The benchmark separates planning-only trajectory quality from real physics
replay. It uses the original Franka PandaWithHand asset and reports gripper vs.
object feasibility instead of modifying the asset or forcing object motion.
Run: python -m scripts.benchmark.robotics.nmg.franka_pick_place
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import torch

from embodichain.data import get_data_path
from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectState,
    MoveEndEffector,
    MoveEndEffectorCfg,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenerator,
    NeuralPlannerCfg,
    ToppraPlannerCfg,
)
from embodichain.lab.sim.sensors import (
    ArticulationContactFilterCfg,
    CameraCfg,
    ContactSensor,
    ContactSensorCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix
from scripts.tutorials.atomic_action.tutorial_utils import (
    draw_axis_marker,
    start_auto_play_recording,
    stop_auto_play_recording,
)

SCRIPT_NAME = "franka_pick_place_nmg"
ARM_NAME = "main_arm"
HAND_NAME = "hand"
ROBOT_UID = "FrankaPanda"
TABLE_UID = "support_table"
TABLE_COLLIDER_UID = "support_table_collider"
TABLE_MESH_PATH = "ShopTableSimple/shop_table_simple.ply"
TABLE_MESH_TOP_Z = 0.8265
TABLE_TOP_Z = 0.1
TABLE_INIT_POS = (0.0, 0.0, TABLE_TOP_Z - TABLE_MESH_TOP_Z)
TABLE_SCALE = (1.0, 1.0, 1.0)
TABLE_COLLIDER_SIZE = (1.0, 1.0, 0.04)
TABLE_COLLIDER_INIT_POS = (
    0.0,
    0.0,
    TABLE_TOP_Z - TABLE_COLLIDER_SIZE[2] / 2.0,
)
OBJECT_SUPPORT_MARGIN = 0.002
TUTORIAL_PLACE_PROFILE = "tutorial_place"
BENCHMARK_PROFILE = "benchmark"
DEFAULT_DEMO_PROFILE = TUTORIAL_PLACE_PROFILE
TUTORIAL_OBJECT_NAME = "sugar_box"
TUTORIAL_OBJECT_XY = (-0.42, -0.08)
TUTORIAL_OBJECT_INIT_Z = 0.05
TUTORIAL_OBJECT_SCALE = (0.8, 0.8, 0.8)
TUTORIAL_PRE_GRASP_DISTANCE = 0.15
TUTORIAL_PICK_LIFT_HEIGHT = 0.16
TUTORIAL_PLACE_LIFT_HEIGHT = 0.14
BENCHMARK_PRE_GRASP_DISTANCE = 0.12
BENCHMARK_PICK_LIFT_HEIGHT = 0.14
BENCHMARK_PLACE_LIFT_HEIGHT = 0.14

PICK_SAMPLE_INTERVAL = 120
PLACE_SAMPLE_INTERVAL = 120
MOVE_SAMPLE_INTERVAL = 80
HAND_INTERP_STEPS = 12
POST_REPLAY_HOLD_STEPS = 80
POST_GRASP_HOLD_STEPS = 80
TUTORIAL_POST_TRAJECTORY_STEPS = 240

DEFAULT_POS_THRESHOLD = 1e-3
DEFAULT_ROT_THRESHOLD = 0.05
DEFAULT_OBJECT_POS_THRESHOLD = 0.05
DEFAULT_NMG_POS_THRESHOLD = 0.05
DEFAULT_NMG_ROT_THRESHOLD = 0.3
DEFAULT_PRE_PICK_Z = 0.36
DEFAULT_SEED = 0
DEFAULT_GRIPPER_CLOSE_MARGIN = 0.012
DEFAULT_GRASP_HOLD_STEPS = 80
DEFAULT_OBJECT_REPLAY_MODE = "physics"
DEFAULT_GRASP_AXIS = "auto"
DEFAULT_FRANKA_TCP_Z = 0.1034
DEFAULT_FRANKA_TCP_YAW = -math.pi / 4.0
DEFAULT_GRASP_DEPTH_OFFSET = 0.0
DEFAULT_MIN_PHYSICAL_GPU_FREE_MB = 2048.0
DEFAULT_ARENA_SPACE = 2.5
DEFAULT_OBJECT_SETTLE_STEPS = 10

FRANKA_FINGER_LINK_NAMES = ("leftfinger", "rightfinger")
FRANKA_GRIPPER_CLOSING_AXIS_INDEX = 1
FRANKA_FINGER_LENGTH = 0.058
FRANKA_ROOT_Z_WIDTH = 0.060
FRANKA_Y_THICKNESS = 0.030
FRANKA_START_QPOS = (
    0.0,
    -math.pi / 4.0,
    0.0,
    -3.0 * math.pi / 4.0,
    0.0,
    math.pi / 2.0,
    math.pi / 4.0,
)
FRANKA_REST_QPOS = (0.10, -0.65, 0.00, -2.10, 0.00, 1.55, 0.85)

OBJECT_PRESETS: dict[str, dict[str, Any]] = {
    "cube": {
        "label": "cube",
        "shape": "cube",
        "size": (0.045, 0.045, 0.045),
        "init_pos": (0.31, 0.00, 0.0),
        "init_rot": (0.0, 0.0, 0.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.03,
        "max_convex_hull_num": 1,
        "use_usd_properties": False,
        "mesh_min_z": -0.0225,
    },
    "sugar_box": {
        "label": "sugar_box",
        "mesh_path": "SugarBox/sugar_box_usd/sugar_box.usda",
        "init_pos": (0.31, 0.00, 0.0),
        "init_rot": (0.0, 0.0, 0.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.05,
        "max_convex_hull_num": 16,
        "use_usd_properties": False,
        "mesh_min_z": -0.02256747,
    },
    "cup": {
        "label": "cup",
        "mesh_path": "CoffeeCup/cup.ply",
        "init_pos": (0.31, 0.00, 0.0),
        "init_rot": (0.0, 0.0, -90.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.01,
        "max_convex_hull_num": 16,
        "use_usd_properties": False,
    },
}

APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


@dataclass(frozen=True)
class GraspPreflight:
    """Result of object-vs-gripper feasibility checks."""

    status: str
    object_grasp_width_m: float
    gripper_min_opening_m: float
    gripper_max_opening_m: float
    reason: str

    @property
    def failed(self) -> bool:
        """Return whether the preflight indicates an infeasible grasp."""
        return self.status != "ok"


@dataclass(frozen=True)
class FrankaHandOpeningSpec:
    """URDF-derived Franka finger geometry used to compute jaw opening."""

    left_origin_y: float
    left_axis_y: float
    left_inner_y: float
    right_origin_y: float
    right_axis_y: float
    right_inner_y: float


@dataclass(frozen=True)
class PlanOutcome:
    """Result returned by the action planner for one trial."""

    success: bool
    trajectory: torch.Tensor
    held_object: HeldObjectState | None
    place_eef_pose: torch.Tensor | None
    rest_eef_pose: torch.Tensor


@dataclass(frozen=True)
class ContactStats:
    """Summary of current finger-object contacts for one replay instant."""

    count: int
    max_impulse: float
    total_impulse: float
    min_distance: float | None


@dataclass(frozen=True)
class ReplayOutcome:
    """Physical replay metrics for one planned trajectory."""

    replay_time_sec: float
    replay_success: bool
    physical_success: bool
    object_lifted: bool
    object_moved: bool
    object_pos_error: float | None
    object_rot_error: float | None
    object_max_z: float | None
    object_displacement_m: float | None
    object_replay_mode: str
    close_end_index: int | None
    release_start_index: int | None
    min_hand_opening_m: float | None
    final_hand_opening_m: float | None
    actual_min_hand_opening_m: float | None
    actual_close_hand_opening_m: float | None
    actual_final_hand_opening_m: float | None
    close_contact_count: int | None
    path_contact_count: int | None
    hold_contact_count: int | None
    max_contact_count: int | None
    max_contact_impulse: float | None
    max_total_contact_impulse: float | None
    min_contact_distance_m: float | None
    close_tcp_object_delta_m: list[float] | None
    planned_grasp_object_delta_m: list[float] | None
    close_leftfinger_object_delta_m: list[float] | None
    close_rightfinger_object_delta_m: list[float] | None
    close_tcp_x_axis: list[float] | None
    close_tcp_y_axis: list[float] | None
    close_tcp_z_axis: list[float] | None


@dataclass(frozen=True)
class SummaryRow:
    """Aggregate measured rows for one planner and mode."""

    planner: str
    mode: str
    rows: list[dict[str, object]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Franka PickUp -> Place with IK and NMG planners."
    )
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(headless=True, arena_space=DEFAULT_ARENA_SPACE)
    parser.add_argument(
        "--planner",
        choices=["ik_interpolate", "neural", "neural_refine", "all"],
        default="all",
        help="Planner backend to evaluate. 'all' evaluates all supported planners.",
    )
    parser.add_argument(
        "--neural_checkpoint",
        type=str,
        default=None,
        help=(
            "Local Franka NMG checkpoint path. If omitted and a neural planner is "
            "selected, the project checkpoint downloader is used."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["planner", "physical", "both"],
        default="planner",
        help="Evaluation mode. 'both' runs separate planner and physical rows.",
    )
    parser.add_argument(
        "--run_kind",
        choices=["auto", "demo", "benchmark"],
        default="auto",
        help=(
            "Execution kind. 'demo' runs one tutorial-style PickUp -> Place "
            "sequence without the benchmark loop. 'auto' uses demo when "
            "--open_window is set, otherwise benchmark."
        ),
    )
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS),
        default="sugar_box",
        help="Object preset used for the pick-place scene.",
    )
    parser.add_argument(
        "--demo_profile",
        choices=[TUTORIAL_PLACE_PROFILE, BENCHMARK_PROFILE],
        default=DEFAULT_DEMO_PROFILE,
        help=(
            "'tutorial_place' mirrors scripts/tutorials/atomic_action/place.py "
            "as closely as possible while keeping the Franka robot. 'benchmark' "
            "uses the original object-target benchmark sequence."
        ),
    )
    parser.add_argument(
        "--object_scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Optional object body scale override for strict-physics experiments.",
    )
    parser.add_argument(
        "--object_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Optional world-frame object XY override. This affects both initial "
            "creation and per-trial object reset."
        ),
    )
    parser.add_argument(
        "--support_surface",
        choices=["ground", "table"],
        default="ground",
        help=(
            "Support surface for the object. 'ground' matches the atomic-action "
            "pickup tutorial and uses the SimulationManager default plane. "
            "'table' creates the benchmark's legacy hidden table collider and "
            "visual table mesh."
        ),
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=3,
        help="Measured repeats per planner and mode.",
    )
    parser.add_argument(
        "--warmup_repeats",
        type=int,
        default=1,
        help="Warmup repeats per planner and mode; excluded from summaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used before trial generation.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of antipodal samples used by grasp generation.",
    )
    parser.add_argument(
        "--n_deviated_approach_directions",
        type=int,
        default=1,
        help=(
            "Number of approach directions sampled around the requested approach. "
            "Use 1 for deterministic strict-physics checks."
        ),
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )
    parser.add_argument(
        "--open_window",
        action="store_true",
        help="Open the simulation viewer window for visual inspection.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help=(
            "Run the open-window tutorial profile without waiting for keyboard "
            "inspection prompts."
        ),
    )
    parser.add_argument(
        "--inspect_seconds",
        type=float,
        default=0.0,
        help="Optional seconds to keep updating the viewer before starting trials.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional Markdown report path. Defaults to outputs/benchmarks/.",
    )
    parser.add_argument(
        "--save_raw_jsonl",
        type=str,
        default=None,
        help="Optional JSONL path for raw per-trial rows.",
    )
    parser.add_argument(
        "--pre_pick_z",
        type=float,
        default=DEFAULT_PRE_PICK_Z,
        help="World z position used to seed the pre-pick TCP pose.",
    )
    parser.add_argument(
        "--tcp_z",
        type=float,
        default=DEFAULT_FRANKA_TCP_Z,
        help=(
            "Franka solver TCP z offset in ee_link frame. The default preserves "
            "the existing NMG comparator convention."
        ),
    )
    parser.add_argument(
        "--tcp_yaw",
        type=float,
        default=DEFAULT_FRANKA_TCP_YAW,
        help=(
            "Franka solver TCP yaw offset in radians. The default preserves "
            "the existing NMG comparator convention."
        ),
    )
    parser.add_argument(
        "--grasp_depth_offset",
        type=float,
        default=DEFAULT_GRASP_DEPTH_OFFSET,
        help=(
            "Meters to move the resolved grasp TCP backwards along its local z "
            "axis before planning. Negative values push the Franka fingers "
            "deeper into the object for physical replay experiments."
        ),
    )
    parser.add_argument(
        "--grasp_axis",
        choices=["auto", "short", "long"],
        default=DEFAULT_GRASP_AXIS,
        help=(
            "Preferred object horizontal axis for the Franka finger centerline. "
            "'short' favors closing across the object's short side."
        ),
    )
    parser.add_argument(
        "--approach",
        choices=["top", "side", "side_y", "custom"],
        default="top",
        help="Pick approach direction preset.",
    )
    parser.add_argument(
        "--custom_approach_direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame approach direction used when --approach custom.",
    )
    parser.add_argument(
        "--step_repeat",
        type=int,
        default=4,
        help="Simulation update steps per trajectory waypoint during physical replay.",
    )
    parser.add_argument(
        "--replay_control",
        choices=["target", "direct"],
        default="target",
        help=(
            "Replay mode. 'target' drives joint targets through physics; 'direct' "
            "sets current qpos at every waypoint."
        ),
    )
    parser.add_argument(
        "--object_replay_mode",
        choices=["physics", "attached"],
        default=DEFAULT_OBJECT_REPLAY_MODE,
        help=(
            "'physics' evaluates real object contact. 'attached' binds the object "
            "to the planned held-object transform after grasp for demo inspection."
        ),
    )
    parser.add_argument(
        "--demo_object_replay_mode",
        choices=["physics", "attached"],
        default="attached",
        help=(
            "Object replay mode used only by --run_kind demo. The benchmark loop "
            "continues to use --object_replay_mode."
        ),
    )
    parser.add_argument(
        "--grasp_hold_steps",
        type=int,
        default=DEFAULT_GRASP_HOLD_STEPS,
        help="Extra physics steps inserted after gripper close before lifting.",
    )
    parser.add_argument(
        "--gripper_close_margin",
        type=float,
        default=DEFAULT_GRIPPER_CLOSE_MARGIN,
        help="Meters subtracted from estimated object width when closing Franka fingers.",
    )
    parser.add_argument(
        "--screenshot_dir",
        type=str,
        default=None,
        help="Optional directory for diagnostic replay screenshots.",
    )
    parser.add_argument(
        "--hold_steps",
        type=int,
        default=40,
        help="Simulation update steps before planning and after replay.",
    )
    parser.add_argument(
        "--pos_success_threshold",
        type=float,
        default=DEFAULT_POS_THRESHOLD,
        help="Strict final TCP position threshold in meters.",
    )
    parser.add_argument(
        "--rot_success_threshold",
        type=float,
        default=DEFAULT_ROT_THRESHOLD,
        help="Strict final TCP rotation threshold in radians.",
    )
    parser.add_argument(
        "--object_pos_success_threshold",
        type=float,
        default=DEFAULT_OBJECT_POS_THRESHOLD,
        help="Placed-object position threshold in meters for physical success.",
    )
    parser.add_argument(
        "--nmg_pos_success_threshold",
        type=float,
        default=DEFAULT_NMG_POS_THRESHOLD,
        help="Loose position threshold passed to NMG planners.",
    )
    parser.add_argument(
        "--nmg_rot_success_threshold",
        type=float,
        default=DEFAULT_NMG_ROT_THRESHOLD,
        help="Loose rotation threshold passed to NMG planners.",
    )
    parser.add_argument(
        "--min_physical_gpu_free_mb",
        type=float,
        default=DEFAULT_MIN_PHYSICAL_GPU_FREE_MB,
        help=(
            "Minimum free GPU memory required before launching physical replay. "
            "Set <= 0 to disable the preflight skip."
        ),
    )
    args = parser.parse_args(argv)
    if args.open_window:
        args.headless = False
    if (
        args.demo_profile == TUTORIAL_PLACE_PROFILE
        and args.object != TUTORIAL_OBJECT_NAME
    ):
        logger.log_warning(
            f"{TUTORIAL_PLACE_PROFILE} is calibrated for {TUTORIAL_OBJECT_NAME}; "
            f"received --object {args.object}."
        )
    return args


def should_pause_for_tutorial_inspection(args: argparse.Namespace) -> bool:
    """Return whether the open-window tutorial profile should wait for inspection."""
    return (
        not args.headless
        and args.demo_profile == TUTORIAL_PLACE_PROFILE
        and not args.auto_play
    )


def pause_for_tutorial_inspection(args: argparse.Namespace) -> None:
    """Pause before planning when the tutorial viewer runs in an interactive shell."""
    if not should_pause_for_tutorial_inspection(args):
        return
    if not sys.stdin.isatty():
        logger.log_warning(
            f"{SCRIPT_NAME}: --open_window tutorial inspection prompt skipped "
            "because stdin is not interactive. Use --auto_play to suppress this "
            "warning explicitly."
        )
        return
    input("Inspect the object, then press Enter to plan PickUp -> Place...")


def inspect_viewer_before_trials(
    sim: SimulationManager, args: argparse.Namespace
) -> None:
    """Keep the viewer alive briefly before benchmark trials start."""
    if args.inspect_seconds <= 0.0:
        return
    steps = max(1, int(round(args.inspect_seconds / (1.0 / 100.0))))
    sim.update(step=steps)


def effective_replay_control(args: argparse.Namespace) -> str:
    """Return the replay control mode used for this profile."""
    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        return "direct"
    return args.replay_control


def effective_final_hold_steps(args: argparse.Namespace) -> int:
    """Return the number of final hold iterations used after replay."""
    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        return TUTORIAL_POST_TRAJECTORY_STEPS
    return args.hold_steps + POST_REPLAY_HOLD_STEPS


def effective_pre_plan_hold_steps(args: argparse.Namespace) -> int:
    """Return physics hold steps inserted before planning."""
    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        return 0
    return max(args.hold_steps, 1)


def validate_args(args: argparse.Namespace) -> None:
    """Validate benchmark arguments."""
    if args.num_repeats < 1:
        raise ValueError("--num_repeats must be >= 1.")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup_repeats must be >= 0.")
    if args.n_sample < 1:
        raise ValueError("--n_sample must be >= 1.")
    if args.n_deviated_approach_directions < 1:
        raise ValueError("--n_deviated_approach_directions must be >= 1.")
    if args.object_scale is not None and any(v <= 0.0 for v in args.object_scale):
        raise ValueError("--object_scale values must be > 0.")
    if args.pre_pick_z <= 0.0:
        raise ValueError("--pre_pick_z must be > 0.")
    if args.tcp_z <= 0.0:
        raise ValueError("--tcp_z must be > 0.")
    if args.step_repeat < 1:
        raise ValueError("--step_repeat must be >= 1.")
    if args.grasp_hold_steps < 0:
        raise ValueError("--grasp_hold_steps must be >= 0.")
    if args.gripper_close_margin < 0.0:
        raise ValueError("--gripper_close_margin must be >= 0.")
    if args.hold_steps < 0:
        raise ValueError("--hold_steps must be >= 0.")
    if args.inspect_seconds < 0.0:
        raise ValueError("--inspect_seconds must be >= 0.")
    if args.min_physical_gpu_free_mb < 0.0:
        raise ValueError("--min_physical_gpu_free_mb must be >= 0.")


def simulation_requires_cuda(args: argparse.Namespace) -> bool:
    """Return whether the selected simulation settings require CUDA."""
    if str(args.device).startswith("cuda"):
        return True
    return args.renderer in ("hybrid", "fast-rt", "rt")


def expand_planner_selection(planner: str) -> list[str]:
    """Expand planner aliases into concrete planner names."""
    if planner == "all":
        return ["ik_interpolate", "neural", "neural_refine"]
    return [planner]


def expand_mode_selection(mode: str) -> list[str]:
    """Expand mode aliases into concrete benchmark modes."""
    if mode == "both":
        return ["planner", "physical"]
    return [mode]


def free_gpu_memory_mb(gpu_id: int) -> float | None:
    """Return free GPU memory in MiB, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info(int(gpu_id))
    return free_bytes / 1024**2


def physical_gpu_memory_skip_reason(
    args: argparse.Namespace,
    modes: list[str],
) -> str | None:
    """Return a graceful-skip reason when physical replay is likely to OOM."""
    if "physical" not in modes or args.min_physical_gpu_free_mb <= 0.0:
        return None
    free_mb = free_gpu_memory_mb(args.gpu_id)
    if free_mb is None or free_mb >= args.min_physical_gpu_free_mb:
        return None
    return (
        "Physical replay was skipped before SimulationManager initialization "
        f"because free GPU memory on gpu_id={args.gpu_id} is {free_mb:.1f} MiB, "
        f"below --min_physical_gpu_free_mb={args.min_physical_gpu_free_mb:.1f} MiB. "
        "DexSim/Vulkan may otherwise terminate the process before Python can write "
        "benchmark reports."
    )


def _franka_tcp(
    tcp_z: float = DEFAULT_FRANKA_TCP_Z,
    tcp_yaw: float = DEFAULT_FRANKA_TCP_YAW,
) -> list[list[float]]:
    """Return the Franka TCP convention used by the existing NMG comparator."""
    c = math.cos(float(tcp_yaw))
    s = math.sin(float(tcp_yaw))
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, float(tcp_z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def resolve_object_body_scale(
    object_name: str,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    demo_profile: str = DEFAULT_DEMO_PROFILE,
) -> tuple[float, float, float]:
    """Return the benchmark scale for an object preset or CLI override."""
    if object_scale is not None:
        if len(object_scale) != 3:
            raise ValueError(f"object_scale must have 3 values, got {object_scale}")
        scale = tuple(float(v) for v in object_scale)
    elif demo_profile == TUTORIAL_PLACE_PROFILE and object_name == TUTORIAL_OBJECT_NAME:
        scale = TUTORIAL_OBJECT_SCALE
    else:
        scale = tuple(float(v) for v in OBJECT_PRESETS[object_name]["body_scale"])
    if any(v <= 0.0 for v in scale):
        raise ValueError(f"object scale values must be > 0, got {scale}")
    return scale


def object_body_scale(object_name: str) -> tuple[float, float, float]:
    """Return the fixed benchmark scale for an object preset."""
    return resolve_object_body_scale(object_name)


def object_supported_z(
    object_name: str,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    demo_profile: str = DEFAULT_DEMO_PROFILE,
) -> float:
    """Return the object origin z that places its scaled bottom on the support plane."""
    preset = OBJECT_PRESETS[object_name]
    scale = resolve_object_body_scale(object_name, object_scale, demo_profile)
    if preset.get("shape") == "cube":
        size_z = float(preset["size"][2]) * scale[2]
        return TABLE_TOP_Z + size_z / 2.0 + OBJECT_SUPPORT_MARGIN
    mesh_min_z = float(preset.get("mesh_min_z", 0.0))
    return TABLE_TOP_Z - mesh_min_z * scale[2] + OBJECT_SUPPORT_MARGIN


def object_initial_position(
    object_name: str,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    support_surface: str = "ground",
    demo_profile: str = DEFAULT_DEMO_PROFILE,
    object_xy: tuple[float, float] | list[float] | None = None,
) -> tuple[float, float, float]:
    """Return the benchmark object position."""
    if support_surface not in ("ground", "table"):
        raise ValueError(f"Unsupported support_surface: {support_surface}")
    pos = OBJECT_PRESETS[object_name]["init_pos"]
    xy = None if object_xy is None else (float(object_xy[0]), float(object_xy[1]))
    if demo_profile == TUTORIAL_PLACE_PROFILE and object_name == TUTORIAL_OBJECT_NAME:
        xy = xy or (float(TUTORIAL_OBJECT_XY[0]), float(TUTORIAL_OBJECT_XY[1]))
        if support_surface == "ground":
            return (
                xy[0],
                xy[1],
                TUTORIAL_OBJECT_INIT_Z,
            )
        pos = (xy[0], xy[1], pos[2])
    elif xy is not None:
        pos = (xy[0], xy[1], pos[2])
    return (
        float(pos[0]),
        float(pos[1]),
        object_supported_z(object_name, object_scale, demo_profile),
    )


def object_initial_pose(
    object_name: str,
    device: torch.device,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    support_surface: str = "ground",
    demo_profile: str = DEFAULT_DEMO_PROFILE,
    object_xy: tuple[float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Return the initial object pose as an unbatched 7D wxyz pose tensor."""
    preset = OBJECT_PRESETS[object_name]
    pos = object_initial_position(
        object_name,
        object_scale,
        support_surface,
        demo_profile,
        object_xy,
    )
    yaw = math.radians(float(preset["init_rot"][2]))
    return torch.tensor(
        [
            pos[0],
            pos[1],
            pos[2],
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ],
        dtype=torch.float32,
        device=device,
    )


def resolve_approach_direction(
    args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    """Resolve and normalize the pick approach direction."""
    if args.approach == "custom":
        if args.custom_approach_direction is None:
            raise ValueError(
                "--custom_approach_direction is required when --approach custom."
            )
        direction = args.custom_approach_direction
    else:
        direction = APPROACH_DIRECTIONS[args.approach]

    approach_direction = torch.tensor(direction, dtype=torch.float32, device=device)
    norm = torch.linalg.norm(approach_direction)
    if norm < 1e-6:
        raise ValueError("approach_direction must be non-zero.")
    return approach_direction / norm


def resolve_hand_open_close_qpos(
    limits: torch.Tensor,
    *,
    open_qpos: float = 0.04,
    close_qpos: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp scalar Panda finger qpos values to per-joint limits."""
    limits = limits.to(dtype=torch.float32)
    lower = limits[:, 0]
    upper = limits[:, 1]
    hand_open = torch.full_like(lower, float(open_qpos)).clamp(min=lower, max=upper)
    hand_close = torch.full_like(lower, float(close_qpos)).clamp(min=lower, max=upper)
    return hand_open, hand_close


def resolve_adaptive_hand_close_qpos(
    limits: torch.Tensor,
    *,
    object_width_m: float,
    margin_m: float,
    franka_urdf_path: str | None = None,
) -> torch.Tensor:
    """Return a Franka close target that fits the object without over-closing."""
    limits = limits.to(dtype=torch.float32)
    lower = limits[:, 0]
    upper = limits[:, 1]
    target_total_opening = max(float(object_width_m) - float(margin_m), 0.0)
    if franka_urdf_path is not None:
        spec = load_franka_hand_opening_spec(franka_urdf_path)
        closed_opening = float(
            franka_hand_opening_from_finger_qpos(
                torch.zeros(2, dtype=torch.float32),
                spec,
            ).item()
        )
        axis_gap = abs(spec.left_axis_y) + abs(spec.right_axis_y)
        per_finger = (target_total_opening - closed_opening) / max(axis_gap, 1e-6)
    else:
        per_finger = target_total_opening / float(limits.shape[0])
    return torch.full_like(lower, per_finger).clamp(min=lower, max=upper)


def get_hand_open_close_qpos(
    robot: Robot,
    preflight: GraspPreflight | None = None,
    *,
    close_margin_m: float = DEFAULT_GRIPPER_CLOSE_MARGIN,
    franka_urdf_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Franka hand open and adaptive close qpos values."""
    limits = robot.get_qpos_limits(name=HAND_NAME)[0].to(
        device=robot.device, dtype=torch.float32
    )
    hand_open, hand_close = resolve_hand_open_close_qpos(limits)
    if preflight is not None and preflight.object_grasp_width_m > 0.0:
        hand_close = resolve_adaptive_hand_close_qpos(
            limits,
            object_width_m=preflight.object_grasp_width_m,
            margin_m=close_margin_m,
            franka_urdf_path=franka_urdf_path,
        ).to(device=robot.device)
    return hand_open, hand_close


def _finger_joint_spec(urdf_root: ET.Element, joint_name: str) -> tuple[float, float]:
    """Return origin-y and axis-y for a Franka finger joint."""
    joint = urdf_root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Franka hand joint not found in URDF: {joint_name}")
    origin = joint.find("origin")
    axis = joint.find("axis")
    if origin is None or axis is None:
        raise ValueError(f"Franka hand joint is missing origin or axis: {joint_name}")
    origin_xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
    axis_xyz = [float(v) for v in axis.attrib.get("xyz", "0 0 0").split()]
    if len(origin_xyz) != 3 or len(axis_xyz) != 3:
        raise ValueError(f"Invalid Franka finger joint fields: {joint_name}")
    return origin_xyz[1], axis_xyz[1]


def _mesh_filename(urdf_root: ET.Element, link_name: str) -> str:
    """Return the first collision mesh filename for a URDF link."""
    link = urdf_root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Franka link not found in URDF: {link_name}")
    mesh = link.find("./collision/geometry/mesh")
    if mesh is None or "filename" not in mesh.attrib:
        raise ValueError(f"Franka link has no collision mesh: {link_name}")
    return mesh.attrib["filename"]


def _binary_stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    """Return axis-aligned bounds for a binary STL file."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Invalid binary STL file: {path}")
    n_triangles = struct.unpack_from("<I", data, 80)[0]
    expected_len = 84 + n_triangles * 50
    if len(data) < expected_len:
        raise ValueError(f"Truncated binary STL file: {path}")

    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    offset = 84
    for _ in range(n_triangles):
        # normal(12 bytes), 3 vertices(36 bytes), attribute count(2 bytes)
        vertices = struct.unpack_from("<9f", data, offset + 12)
        offset += 50
        for axis in range(3):
            coords = vertices[axis::3]
            mins[axis] = min(mins[axis], *coords)
            maxs[axis] = max(maxs[axis], *coords)
    return mins, maxs


def _finger_inner_y_from_mesh(
    urdf_path: str,
    urdf_root: ET.Element,
    *,
    link_name: str,
) -> float:
    """Estimate the inner finger collision surface y in the link frame."""
    mesh_path = Path(urdf_path).parent / _mesh_filename(urdf_root, link_name)
    mins, maxs = _binary_stl_bounds(mesh_path)
    if link_name == "leftfinger":
        return mins[1]
    if link_name == "rightfinger":
        # The right finger collision mesh is rotated by pi around z. Its mesh
        # minimum-y face maps to positive link-y, which is the inner side.
        return -mins[1]
    raise ValueError(f"Unsupported Franka finger link: {link_name}")


def load_franka_hand_opening_spec(urdf_path: str) -> FrankaHandOpeningSpec:
    """Load Franka finger joint geometry needed for jaw opening estimates."""
    root = ET.parse(urdf_path).getroot()
    left_origin_y, left_axis_y = _finger_joint_spec(root, "finger_joint1")
    right_origin_y, right_axis_y = _finger_joint_spec(root, "finger_joint2")
    return FrankaHandOpeningSpec(
        left_origin_y=left_origin_y,
        left_axis_y=left_axis_y,
        left_inner_y=_finger_inner_y_from_mesh(
            urdf_path,
            root,
            link_name="leftfinger",
        ),
        right_origin_y=right_origin_y,
        right_axis_y=right_axis_y,
        right_inner_y=_finger_inner_y_from_mesh(
            urdf_path,
            root,
            link_name="rightfinger",
        ),
    )


def franka_hand_opening_from_finger_qpos(
    finger_qpos: torch.Tensor,
    spec: FrankaHandOpeningSpec,
) -> torch.Tensor:
    """Return physical jaw opening from Franka finger qpos values."""
    if finger_qpos.dim() == 1:
        finger_qpos = finger_qpos.unsqueeze(0)
    if finger_qpos.shape[-1] != 2:
        raise ValueError(f"finger_qpos must end with 2 values, got {finger_qpos.shape}")
    left_y = spec.left_origin_y + spec.left_axis_y * finger_qpos[..., 0]
    left_y = left_y + spec.left_inner_y
    right_y = spec.right_origin_y + spec.right_axis_y * finger_qpos[..., 1]
    right_y = right_y + spec.right_inner_y
    return torch.abs(left_y - right_y)


def estimate_gripper_opening_range(
    urdf_path: str, hand_limits: torch.Tensor
) -> tuple[float, float]:
    """Estimate min and max Franka finger opening from joint travel.

    Panda finger joints are prismatic and move symmetrically. The physical
    opening is the distance between the two child-link joint origins after
    applying each prismatic displacement.
    """
    spec = load_franka_hand_opening_spec(urdf_path)
    limits = torch.as_tensor(hand_limits, dtype=torch.float32).detach().cpu()
    if limits.shape != (2, 2):
        raise ValueError(
            f"Franka hand_limits must have shape (2, 2), got {limits.shape}"
        )

    openings = []
    for left_q in (float(limits[0, 0]), float(limits[0, 1])):
        for right_q in (float(limits[1, 0]), float(limits[1, 1])):
            opening = franka_hand_opening_from_finger_qpos(
                torch.tensor([left_q, right_q], dtype=torch.float32),
                spec,
            )
            openings.append(float(opening.item()))
    return min(openings), max(openings)


def estimate_object_grasp_width(vertices: torch.Tensor) -> float:
    """Estimate the narrowest horizontal grasp width from scaled object vertices."""
    verts = torch.as_tensor(vertices, dtype=torch.float32)
    if verts.numel() == 0:
        return 0.0
    if verts.dim() != 2 or verts.shape[1] < 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    extents = verts[:, :2].max(dim=0).values - verts[:, :2].min(dim=0).values
    positive = extents[extents > 1e-6]
    if positive.numel() == 0:
        return 0.0
    return float(torch.min(positive).item())


def horizontal_bbox_axis(vertices: torch.Tensor, axis_preference: str) -> torch.Tensor:
    """Return the world horizontal short or long bbox axis for object vertices."""
    if axis_preference not in ("short", "long"):
        raise ValueError(f"Unsupported axis_preference: {axis_preference}")
    verts = torch.as_tensor(vertices, dtype=torch.float32)
    if verts.dim() != 2 or verts.shape[1] < 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    extents = verts[:, :2].max(dim=0).values - verts[:, :2].min(dim=0).values
    axis_idx = int(torch.argmin(extents).item())
    if axis_preference == "long":
        axis_idx = 1 - axis_idx
    axis = torch.zeros(3, dtype=torch.float32, device=verts.device)
    axis[axis_idx] = 1.0
    return axis


def grasp_axis_alignment_cost(
    candidate_axis: torch.Tensor, preferred_axis: torch.Tensor
) -> torch.Tensor:
    """Return 0 for candidate axes aligned with the preferred object axis."""
    candidate_axis = torch.as_tensor(candidate_axis, dtype=torch.float32)
    preferred_axis = torch.as_tensor(
        preferred_axis,
        dtype=torch.float32,
        device=candidate_axis.device,
    )
    preferred_axis = preferred_axis / torch.clamp(
        torch.linalg.norm(preferred_axis),
        min=1e-6,
    )
    candidate_axis = candidate_axis / torch.clamp(
        torch.linalg.norm(candidate_axis, dim=-1, keepdim=True), min=1e-6
    )
    return 1.0 - torch.abs((candidate_axis * preferred_axis).sum(dim=-1))


def rerank_grasp_costs_by_axis(
    costs: torch.Tensor,
    grasp_poses: torch.Tensor,
    preferred_axis: torch.Tensor,
    *,
    weight: float = 2.0,
    axis_index: int = FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
) -> torch.Tensor:
    """Add an axis-alignment penalty to grasp candidate costs."""
    if grasp_poses.dim() != 3 or grasp_poses.shape[-2:] != (4, 4):
        raise ValueError(
            f"grasp_poses must have shape (N, 4, 4), got {grasp_poses.shape}"
        )
    if axis_index not in (0, 1, 2):
        raise ValueError(f"axis_index must be 0, 1, or 2, got {axis_index}")
    costs = torch.as_tensor(costs, dtype=torch.float32, device=grasp_poses.device)
    return costs + float(weight) * grasp_axis_alignment_cost(
        grasp_poses[:, :3, axis_index],
        preferred_axis,
    )


def evaluate_grasp_preflight(
    *,
    object_grasp_width_m: float,
    gripper_min_opening_m: float,
    gripper_max_opening_m: float,
    margin_m: float = 0.002,
) -> GraspPreflight:
    """Check whether the object width fits the gripper opening range."""
    if object_grasp_width_m <= 0.0:
        return GraspPreflight(
            status="unknown_width",
            object_grasp_width_m=object_grasp_width_m,
            gripper_min_opening_m=gripper_min_opening_m,
            gripper_max_opening_m=gripper_max_opening_m,
            reason="object_grasp_width_m is not positive",
        )
    if object_grasp_width_m < gripper_min_opening_m - margin_m:
        return GraspPreflight(
            status="too_small",
            object_grasp_width_m=object_grasp_width_m,
            gripper_min_opening_m=gripper_min_opening_m,
            gripper_max_opening_m=gripper_max_opening_m,
            reason="object narrower than gripper minimum opening",
        )
    if object_grasp_width_m > gripper_max_opening_m + margin_m:
        return GraspPreflight(
            status="too_large",
            object_grasp_width_m=object_grasp_width_m,
            gripper_min_opening_m=gripper_min_opening_m,
            gripper_max_opening_m=gripper_max_opening_m,
            reason="object wider than gripper maximum opening",
        )
    return GraspPreflight(
        status="ok",
        object_grasp_width_m=object_grasp_width_m,
        gripper_min_opening_m=gripper_min_opening_m,
        gripper_max_opening_m=gripper_max_opening_m,
        reason="object width lies inside gripper opening range",
    )


def build_grasp_preflight(
    obj: RigidObject,
    robot: Robot,
    *,
    franka_urdf_path: str,
) -> GraspPreflight:
    """Build a preflight result from live object vertices and robot limits."""
    vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    hand_limits = robot.get_qpos_limits(name=HAND_NAME)[0]
    min_opening, max_opening = estimate_gripper_opening_range(
        franka_urdf_path, hand_limits
    )
    return evaluate_grasp_preflight(
        object_grasp_width_m=estimate_object_grasp_width(vertices),
        gripper_min_opening_m=min_opening,
        gripper_max_opening_m=max_opening,
    )


def make_sim(args: argparse.Namespace) -> SimulationManager:
    """Create a simulation manager for benchmark trials."""
    sim = SimulationManager(
        SimulationManagerCfg(
            width=1280,
            height=720,
            headless=args.headless,
            sim_device=args.device,
            gpu_id=args.gpu_id,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_dt=1.0 / 100.0,
            num_envs=1,
            arena_space=args.arena_space,
        )
    )
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0.0, 3.0),
        )
    )
    return sim


def create_franka(
    sim: SimulationManager,
    *,
    tcp_z: float = DEFAULT_FRANKA_TCP_Z,
    tcp_yaw: float = DEFAULT_FRANKA_TCP_YAW,
) -> tuple[Robot, str]:
    """Create the Franka PandaWithHand robot from the original asset."""
    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    if not os.path.isfile(urdf):
        raise FileNotFoundError(f"Franka URDF not found: {urdf}")

    cfg = RobotCfg(
        uid=ROBOT_UID,
        fpath=urdf,
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"Joint[1-7]": 1e4, "finger_joint[1-2]": 1e3},
            damping={"Joint[1-7]": 1e3, "finger_joint[1-2]": 1e2},
            max_effort={"Joint[1-7]": 1e5, "finger_joint[1-2]": 1e3},
            drive_type="force",
        ),
        control_parts={
            ARM_NAME: [
                "Joint1",
                "Joint2",
                "Joint3",
                "Joint4",
                "Joint5",
                "Joint6",
                "Joint7",
            ],
            HAND_NAME: ["finger_joint1", "finger_joint2"],
        },
        solver_cfg={
            ARM_NAME: PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=_franka_tcp(tcp_z, tcp_yaw),
                num_samples=30,
            ),
        },
        init_qpos=[*FRANKA_START_QPOS, 0.04, 0.04],
    )
    return sim.add_robot(cfg=cfg), urdf


def create_support_table(sim: SimulationManager) -> RigidObject:
    """Create a stable static support collider for the benchmark object."""
    table_cfg = RigidObjectCfg(
        uid=TABLE_COLLIDER_UID,
        shape=CubeCfg(size=list(TABLE_COLLIDER_SIZE)),
        attrs=RigidBodyAttributesCfg(
            mass=10.0,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        body_type="static",
        init_pos=TABLE_COLLIDER_INIT_POS,
    )
    table = sim.add_rigid_object(cfg=table_cfg)
    if hasattr(table, "set_visible"):
        table.set_visible(False)
    return table


def create_visual_support_table(sim: SimulationManager) -> RigidObject:
    """Create the visual table mesh below the hidden support collider."""
    table_cfg = RigidObjectCfg(
        uid=TABLE_UID,
        shape=MeshCfg(fpath=get_data_path(TABLE_MESH_PATH)),
        attrs=RigidBodyAttributesCfg(
            mass=10.0,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        body_scale=TABLE_SCALE,
        body_type="static",
        init_pos=TABLE_INIT_POS,
    )
    return sim.add_rigid_object(cfg=table_cfg)


def create_object(
    sim: SimulationManager,
    object_name: str,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    support_surface: str = "ground",
    demo_profile: str = DEFAULT_DEMO_PROFILE,
    object_xy: tuple[float, float] | list[float] | None = None,
) -> RigidObject:
    """Create the benchmark object."""
    preset = OBJECT_PRESETS[object_name]
    scale = resolve_object_body_scale(object_name, object_scale, demo_profile)
    shape = (
        CubeCfg(size=list(preset["size"]))
        if preset.get("shape") == "cube"
        else MeshCfg(fpath=get_data_path(preset["mesh_path"]))
    )
    attrs_kwargs = {
        "mass": preset["mass"],
        "dynamic_friction": 0.97,
        "static_friction": 0.99,
        "enable_ccd": True,
    }
    if demo_profile != TUTORIAL_PLACE_PROFILE:
        attrs_kwargs["contact_offset"] = 0.004
    cfg = RigidObjectCfg(
        uid=preset["label"],
        shape=shape,
        attrs=RigidBodyAttributesCfg(**attrs_kwargs),
        max_convex_hull_num=preset["max_convex_hull_num"],
        init_pos=object_initial_position(
            object_name,
            scale,
            support_surface,
            demo_profile,
            object_xy,
        ),
        init_rot=preset["init_rot"],
        body_scale=scale,
        use_usd_properties=preset["use_usd_properties"],
    )
    obj = sim.add_rigid_object(cfg=cfg)
    sim.update(step=DEFAULT_OBJECT_SETTLE_STEPS)
    return obj


def reset_object_pose(
    obj: RigidObject,
    object_name: str,
    object_scale: tuple[float, float, float] | list[float] | None = None,
    support_surface: str = "ground",
    demo_profile: str = DEFAULT_DEMO_PROFILE,
    object_xy: tuple[float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reset the benchmark object to its initial pose and return it as a matrix."""
    pose = object_initial_pose(
        object_name,
        obj.device,
        object_scale,
        support_surface,
        demo_profile,
        object_xy,
    ).unsqueeze(0)
    obj.set_local_pose(pose, env_ids=[0])
    obj.clear_dynamics(env_ids=[0])
    return obj.get_local_pose(to_matrix=True)[0].clone()


def make_arm_qpos(values: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Create a float32 arm qpos tensor."""
    return torch.tensor(values, dtype=torch.float32, device=device)


def set_robot_start_qpos(robot: Robot, hand_open: torch.Tensor) -> None:
    """Reset arm and hand joints to the benchmark start configuration."""
    start_arm = make_arm_qpos(FRANKA_START_QPOS, robot.device).unsqueeze(0)
    hand_open = hand_open.unsqueeze(0)
    for target in (False, True):
        robot.set_qpos(start_arm, name=ARM_NAME, target=target)
        robot.set_qpos(hand_open, name=HAND_NAME, target=target)
    robot.clear_dynamics()


def compute_fk_pose(robot: Robot, qpos_values: tuple[float, ...]) -> torch.Tensor:
    """Compute an unbatched TCP pose from a known arm qpos."""
    return robot.compute_fk(
        qpos=make_arm_qpos(qpos_values, robot.device).unsqueeze(0),
        name=ARM_NAME,
        to_matrix=True,
    )[0]


def make_rest_pose(robot: Robot) -> torch.Tensor:
    """Return a reachable final rest EEF pose generated through Franka FK."""
    return compute_fk_pose(robot, FRANKA_REST_QPOS)


def make_tutorial_place_eef_pose(device: torch.device) -> torch.Tensor:
    """Return the fixed Place target pose used by the atomic-action tutorial."""
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022],
            [-0.9977, 0.0540, -0.0401],
            [0.0401, 0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor([-0.20, 0.28, 0.1], dtype=torch.float32, device=device)
    return pose


def make_object_target_pose(obj: RigidObject) -> torch.Tensor:
    """Return a desired object placement pose derived from the live object pose."""
    target = obj.get_local_pose(to_matrix=True)[0].clone()
    target[:3, 3] += torch.tensor(
        [-0.10, 0.24, 0.0],
        dtype=torch.float32,
        device=target.device,
    )
    return target


def make_tutorial_retract_eef_pose(place_eef_pose: torch.Tensor) -> torch.Tensor:
    """Return the final retract pose produced by the tutorial Place action."""
    retract_pose = place_eef_pose.clone()
    retract_pose[:3, 3] += torch.tensor(
        [0.0, 0.0, TUTORIAL_PLACE_LIFT_HEIGHT],
        dtype=torch.float32,
        device=place_eef_pose.device,
    )
    return retract_pose


def make_pre_pick_eef_pose(robot: Robot, position: torch.Tensor) -> torch.Tensor:
    """Return a pre-pick TCP pose using the current TCP orientation."""
    pose = robot.compute_fk(
        qpos=robot.get_qpos(name=ARM_NAME),
        name=ARM_NAME,
        to_matrix=True,
    ).clone()
    pose[:, :3, 3] = position
    return pose


def initialize_pre_pick_robot_pose(
    robot: Robot,
    obj: RigidObject,
    hand_open: torch.Tensor,
    *,
    pre_pick_z: float,
    reference_pose: torch.Tensor | None = None,
) -> None:
    """Move Franka to a pre-pick pose above the object."""
    obj_pose = (
        obj.get_local_pose(to_matrix=True)
        if reference_pose is None
        else reference_pose.to(device=robot.device, dtype=torch.float32).unsqueeze(0)
    )
    move_position = obj_pose[:, :3, 3].clone()
    move_position[:, 2] = pre_pick_z
    pre_pick_pose = make_pre_pick_eef_pose(robot, move_position)
    ik_success, arm_qpos = robot.compute_ik(
        pose=pre_pick_pose,
        joint_seed=robot.get_qpos(name=ARM_NAME),
        name=ARM_NAME,
    )
    if not torch.all(ik_success):
        obj_xyz = obj_pose[0, :3, 3].detach().cpu().tolist()
        target_xyz = move_position[0].detach().cpu().tolist()
        raise RuntimeError(
            "Failed to initialize Franka at the pre-pick pose "
            f"(object_xyz={obj_xyz}, target_xyz={target_xyz})."
        )

    n_envs = robot.get_qpos().shape[0]
    hand_qpos = hand_open.unsqueeze(0).repeat(n_envs, 1)
    for target in (False, True):
        robot.set_qpos(arm_qpos, name=ARM_NAME, target=target)
        robot.set_qpos(hand_qpos, name=HAND_NAME, target=target)
    robot.clear_dynamics()


def build_gripper_collision_cfg(preflight: GraspPreflight) -> GripperCollisionCfg:
    """Build collision geometry approximating the original Franka hand."""
    return GripperCollisionCfg(
        max_open_length=preflight.gripper_max_opening_m,
        finger_length=FRANKA_FINGER_LENGTH,
        y_thickness=FRANKA_Y_THICKNESS,
        root_z_width=FRANKA_ROOT_Z_WIDTH,
        open_check_margin=0.002,
        point_sample_dense=0.012,
    )


def build_grasp_generator_cfg(
    args: argparse.Namespace, preflight: GraspPreflight
) -> GraspGeneratorCfg:
    """Build grasp annotation config."""
    cfg = GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=preflight.gripper_max_opening_m,
            min_length=(
                0.003
                if args.demo_profile == TUTORIAL_PLACE_PROFILE
                else max(0.003, preflight.gripper_min_opening_m)
            ),
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=args.support_surface == "table",
    )
    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        return cfg
    return GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=preflight.gripper_max_opening_m,
            min_length=max(0.003, preflight.gripper_min_opening_m),
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=args.support_surface == "table",
        n_deviated_approach_directions=args.n_deviated_approach_directions,
        n_top_grasps=30,
    )


def create_object_semantics(
    obj: RigidObject,
    args: argparse.Namespace,
    preflight: GraspPreflight,
) -> ObjectSemantics:
    """Create grasp semantics for the benchmark object."""
    label = OBJECT_PRESETS[args.object]["label"]
    vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    triangles = obj.get_triangles(env_ids=[0])[0]
    return ObjectSemantics(
        label=label,
        geometry={
            "mesh_vertices": vertices,
            "mesh_triangles": triangles,
        },
        affordance=AntipodalAffordance(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            gripper_collision_cfg=build_gripper_collision_cfg(preflight),
            generator_cfg=build_grasp_generator_cfg(args, preflight),
            force_reannotate=args.force_reannotate,
        ),
        entity=obj,
    )


def resolve_checkpoint(args: argparse.Namespace, planners: list[str]) -> str | None:
    """Resolve the NMG checkpoint path if any selected planner needs it."""
    if not any(planner in ("neural", "neural_refine") for planner in planners):
        return None
    if args.neural_checkpoint:
        return args.neural_checkpoint
    return download_neural_planner_checkpoint()


def build_motion_generator(
    robot: Robot,
    planner_name: str,
    checkpoint_path: str | None,
    args: argparse.Namespace,
) -> MotionGenerator:
    """Create a motion generator for one planner backend."""
    if planner_name == "ik_interpolate":
        planner_cfg = ToppraPlannerCfg(robot_uid=robot.uid)
    elif planner_name in ("neural", "neural_refine"):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for neural planners.")
        planner_cfg = NeuralPlannerCfg(
            robot_uid=robot.uid,
            planner_type=planner_name,
            checkpoint_path=checkpoint_path,
            control_part=ARM_NAME,
            num_arm_joints=len(robot.get_joint_ids(ARM_NAME)),
            pos_eps=args.nmg_pos_success_threshold,
            rot_eps=args.nmg_rot_success_threshold,
        )
    else:
        raise ValueError(f"Unsupported planner: {planner_name}")
    return MotionGenerator(cfg=MotionGenCfg(planner_cfg=planner_cfg))


class PhysicalPickUp(PickUp):
    """Benchmark-local PickUp variant with an optional grasp-depth offset."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PickUpCfg,
        *,
        grasp_depth_offset: float,
        grasp_axis: str,
    ) -> None:
        super().__init__(motion_generator, cfg=cfg)
        self.grasp_depth_offset = float(grasp_depth_offset)
        self.grasp_axis = str(grasp_axis)

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grasp_axis == "auto":
            is_success, grasp_xpos = super()._resolve_grasp_pose(semantics)
        else:
            is_success, grasp_xpos = self._resolve_axis_preferred_grasp_pose(
                semantics,
            )
        if abs(self.grasp_depth_offset) < 1e-9:
            return is_success, grasp_xpos
        grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            grasp_xpos[:, :3, 2] * self.grasp_depth_offset,
        )
        return is_success, grasp_xpos

    def _resolve_axis_preferred_grasp_pose(
        self,
        semantics: ObjectSemantics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses,
            approach_direction=self.approach_direction,
        )
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        is_success_list = []
        best_grasp_xpos_list = []
        vertices = semantics.entity.get_vertices(env_ids=[0], scale=True)[0]
        preferred_axis = horizontal_bbox_axis(vertices, self.grasp_axis)
        for env_id, (candidate_xpos, candidate_costs) in enumerate(grasp_poses_result):
            if candidate_xpos.shape == (4, 4):
                candidate_xpos = candidate_xpos.unsqueeze(0)
            if candidate_costs.dim() == 0:
                candidate_costs = candidate_costs.unsqueeze(0)
            candidate_xpos = candidate_xpos.to(self.device)
            candidate_costs = candidate_costs.to(self.device)
            joint_seed = init_qpos[env_id : env_id + 1, None, :].repeat(
                1, candidate_xpos.shape[0], 1
            )
            ik_success, _ = self.robot.compute_batch_ik(
                pose=candidate_xpos.unsqueeze(0),
                name=self.cfg.control_part,
                joint_seed=joint_seed,
            )
            reranked_costs = rerank_grasp_costs_by_axis(
                candidate_costs,
                candidate_xpos,
                preferred_axis.to(candidate_xpos.device),
                axis_index=FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
            )
            masked_costs = torch.where(
                ik_success[0],
                reranked_costs,
                torch.full_like(reranked_costs, 10000.0),
            )
            best_cost, best_idx = masked_costs.min(dim=0)
            is_success_list.append(best_cost < 9999.0)
            best_grasp_xpos_list.append(candidate_xpos[best_idx].unsqueeze(0))
        is_success = torch.stack(is_success_list).to(device=self.device)
        best_grasp_xpos = torch.cat(best_grasp_xpos_list, dim=0).to(self.device)
        return is_success, best_grasp_xpos


def build_engine(
    robot: Robot,
    motion_gen: MotionGenerator,
    args: argparse.Namespace,
    preflight: GraspPreflight | None = None,
    franka_urdf_path: str | None = None,
) -> AtomicActionEngine:
    """Register the complete benchmark action sequence."""
    hand_open, hand_close = get_hand_open_close_qpos(
        robot,
        preflight,
        close_margin_m=args.gripper_close_margin,
        franka_urdf_path=franka_urdf_path,
    )
    tutorial_profile = args.demo_profile == TUTORIAL_PLACE_PROFILE
    pickup_cfg = PickUpCfg(
        control_part=ARM_NAME,
        hand_control_part=HAND_NAME,
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=resolve_approach_direction(args, robot.device),
        pre_grasp_distance=(
            TUTORIAL_PRE_GRASP_DISTANCE
            if tutorial_profile
            else BENCHMARK_PRE_GRASP_DISTANCE
        ),
        lift_height=(
            TUTORIAL_PICK_LIFT_HEIGHT
            if tutorial_profile
            else BENCHMARK_PICK_LIFT_HEIGHT
        ),
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    place_cfg = PlaceCfg(
        control_part=ARM_NAME,
        hand_control_part=HAND_NAME,
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        lift_height=(
            TUTORIAL_PLACE_LIFT_HEIGHT
            if tutorial_profile
            else BENCHMARK_PLACE_LIFT_HEIGHT
        ),
        sample_interval=PLACE_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    move_cfg = MoveEndEffectorCfg(
        control_part=ARM_NAME,
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )

    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        PhysicalPickUp(
            motion_gen,
            cfg=pickup_cfg,
            grasp_depth_offset=args.grasp_depth_offset,
            grasp_axis=args.grasp_axis,
        )
    )
    engine.register(Place(motion_gen, cfg=place_cfg))
    engine.register(MoveEndEffector(motion_gen, cfg=move_cfg))
    return engine


def plan_sequence(
    engine: AtomicActionEngine,
    semantics: ObjectSemantics,
    target_pose: torch.Tensor,
    rest_pose: torch.Tensor,
    *,
    demo_profile: str,
) -> PlanOutcome:
    """Plan the requested PickUp/Place sequence."""
    pick_success, pick_traj, pick_state = engine.run(
        steps=[("pick_up", GraspTarget(semantics=semantics))]
    )
    if not pick_success or pick_state.held_object is None:
        return PlanOutcome(
            success=False,
            trajectory=pick_traj,
            held_object=pick_state.held_object,
            place_eef_pose=None,
            rest_eef_pose=rest_pose,
        )

    if demo_profile == TUTORIAL_PLACE_PROFILE:
        place_pose = target_pose
        finish_steps = [("place", EndEffectorPoseTarget(xpos=place_pose))]
    else:
        object_to_eef = pick_state.held_object.object_to_eef.to(
            device=target_pose.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0)
        place_pose = torch.bmm(target_pose.unsqueeze(0), object_to_eef)[0]
        finish_steps = [
            ("place", EndEffectorPoseTarget(xpos=place_pose)),
            ("move_end_effector", EndEffectorPoseTarget(xpos=rest_pose)),
        ]

    finish_success, finish_traj, _ = engine.run(steps=finish_steps, state=pick_state)
    trajectory = torch.cat([pick_traj, finish_traj], dim=1)
    return PlanOutcome(
        success=finish_success,
        trajectory=trajectory,
        held_object=pick_state.held_object,
        place_eef_pose=place_pose,
        rest_eef_pose=rest_pose,
    )


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
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / 1024**2
    gpu_mb = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return {"cpu_mb": cpu_mb, "gpu_mb": gpu_mb}


def timed_plan_sequence(
    engine: AtomicActionEngine,
    semantics: ObjectSemantics,
    target_pose: torch.Tensor,
    rest_pose: torch.Tensor,
    *,
    demo_profile: str,
) -> tuple[PlanOutcome, float, dict[str, float], float]:
    """Plan once and return time, memory delta, and peak GPU memory."""
    _reset_peak_gpu_memory()
    before = _memory_snapshot()
    _sync_cuda()
    start = time.perf_counter()
    outcome = plan_sequence(
        engine,
        semantics,
        target_pose,
        rest_pose,
        demo_profile=demo_profile,
    )
    _sync_cuda()
    elapsed = time.perf_counter() - start
    after = _memory_snapshot()
    mem_delta = {
        "cpu_mb": after["cpu_mb"] - before["cpu_mb"],
        "gpu_mb": after["gpu_mb"] - before["gpu_mb"],
    }
    return outcome, elapsed, mem_delta, _peak_gpu_memory_mb()


def _set_replay_qpos(robot: Robot, qpos: torch.Tensor, *, mode: str) -> None:
    """Apply a replay qpos either as a target or as a direct state reset."""
    if mode == "target":
        robot.set_qpos(qpos, target=True)
    elif mode == "direct":
        robot.set_qpos(qpos, target=False)
        robot.set_qpos(qpos, target=True)
    else:
        raise ValueError(f"Unsupported replay control mode: {mode}")


def make_finger_object_contact_sensor(
    sim: SimulationManager,
    obj: RigidObject,
    *,
    uid_suffix: str,
) -> ContactSensor:
    """Create a sensor that only reports Franka finger contacts with the object."""
    contact_cfg = ContactSensorCfg(
        uid=f"franka_pick_contact_{uid_suffix}_{time.time_ns()}",
        rigid_uid_list=[obj.uid],
        articulation_cfg_list=[
            ArticulationContactFilterCfg(
                articulation_uid=ROBOT_UID,
                link_name_list=list(FRANKA_FINGER_LINK_NAMES),
            )
        ],
        filter_need_both_actor=True,
        max_contacts_per_env=128,
    )
    return sim.add_sensor(sensor_cfg=contact_cfg)


def summarize_contact_sensor(contact_sensor: ContactSensor | None) -> ContactStats:
    """Return current filtered contact count and impulse summary."""
    if contact_sensor is None:
        return ContactStats(0, 0.0, 0.0, None)
    contact_sensor.update()
    report = contact_sensor.get_data()
    valid = report["is_valid"][0]
    count = int(valid.sum().item())
    if count == 0:
        return ContactStats(0, 0.0, 0.0, None)
    impulses = report["impulse"][0][valid]
    distances = report["distance"][0][valid]
    return ContactStats(
        count=count,
        max_impulse=float(impulses.max().item()),
        total_impulse=float(impulses.sum().item()),
        min_distance=float(distances.min().item()),
    )


def _max_optional_float(current: float | None, candidate: float | None) -> float | None:
    """Return the max of two optional floats."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _min_optional_float(current: float | None, candidate: float | None) -> float | None:
    """Return the min of two optional floats."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _record_contact_stats(
    stats: ContactStats,
    *,
    max_count: int,
    max_impulse: float | None,
    max_total_impulse: float | None,
    min_distance: float | None,
) -> tuple[int, float | None, float | None, float | None]:
    """Update aggregate contact diagnostics with one sample."""
    return (
        max(max_count, stats.count),
        _max_optional_float(max_impulse, stats.max_impulse),
        _max_optional_float(max_total_impulse, stats.total_impulse),
        _min_optional_float(min_distance, stats.min_distance),
    )


def hand_opening_from_qpos(
    robot: Robot,
    qpos: torch.Tensor,
    spec: FrankaHandOpeningSpec,
) -> torch.Tensor:
    """Return physical Franka jaw opening from a full-DoF qpos tensor."""
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0)
    hand_joint_ids = robot.get_joint_ids(HAND_NAME)
    return franka_hand_opening_from_finger_qpos(qpos[:, hand_joint_ids], spec)


def current_hand_opening(robot: Robot, spec: FrankaHandOpeningSpec) -> float:
    """Return the actual current Franka jaw opening in simulation."""
    opening = franka_hand_opening_from_finger_qpos(
        robot.get_qpos(name=HAND_NAME)[0],
        spec,
    )
    return float(opening.item())


def object_delta_from_tcp(
    robot: Robot,
    obj: RigidObject,
    qpos: torch.Tensor,
) -> list[float]:
    """Return object-position minus TCP-position for one full-DoF qpos."""
    if qpos.dim() == 2:
        qpos = qpos[0]
    arm_joint_ids = robot.get_joint_ids(ARM_NAME)
    tcp_pose = compute_tcp_pose(robot, qpos[arm_joint_ids], control_part=ARM_NAME)
    obj_pos = obj.get_local_pose(to_matrix=True)[0, :3, 3]
    return [float(v) for v in (obj_pos - tcp_pose[:3, 3]).detach().cpu().tolist()]


def link_delta_from_object(
    robot: Robot, obj: RigidObject, link_name: str
) -> list[float]:
    """Return link-position minus object-position for one live link pose."""
    link_pos = robot.get_link_pose(link_name, to_matrix=True)[0, :3, 3]
    obj_pos = obj.get_local_pose(to_matrix=True)[0, :3, 3]
    return [float(v) for v in (link_pos - obj_pos).detach().cpu().tolist()]


def tcp_axes_from_qpos(robot: Robot, qpos: torch.Tensor) -> tuple[list[float], ...]:
    """Return TCP x/y/z axes for one full-DoF qpos."""
    if qpos.dim() == 2:
        qpos = qpos[0]
    arm_joint_ids = robot.get_joint_ids(ARM_NAME)
    tcp_pose = compute_tcp_pose(robot, qpos[arm_joint_ids], control_part=ARM_NAME)
    return tuple(
        [float(v) for v in tcp_pose[:3, axis].detach().cpu().tolist()]
        for axis in range(3)
    )


def _find_hand_close_end_index(
    robot: Robot,
    traj: torch.Tensor,
    spec: FrankaHandOpeningSpec,
) -> int | None:
    """Return the first index where the gripper reaches its minimum opening."""
    if traj.numel() == 0:
        return None
    opening = hand_opening_from_qpos(robot, traj[0], spec).reshape(-1)
    min_opening = torch.min(opening)
    close_indices = torch.nonzero(opening <= min_opening + 1e-4, as_tuple=False)
    if close_indices.numel() == 0:
        return None
    return int(close_indices[0, 0].item())


def _find_hand_release_start_index(
    robot: Robot,
    traj: torch.Tensor,
    spec: FrankaHandOpeningSpec,
) -> int | None:
    """Return the first index after closing where the gripper opens again."""
    close_end_idx = _find_hand_close_end_index(robot, traj, spec)
    if close_end_idx is None:
        return None
    opening = hand_opening_from_qpos(robot, traj[0], spec).reshape(-1)
    min_opening = torch.min(opening)
    release_indices = torch.nonzero(
        opening[close_end_idx:] > min_opening + 1e-4,
        as_tuple=False,
    )
    if release_indices.numel() == 0:
        return None
    return close_end_idx + int(release_indices[0, 0].item())


def compute_attached_object_pose(
    eef_pose: torch.Tensor,
    object_to_eef: torch.Tensor,
) -> torch.Tensor:
    """Return object pose from an EEF pose and object-to-EEF transform."""
    eef_pose = _as_unbatched_pose(eef_pose).to(dtype=torch.float32)
    object_to_eef = _as_unbatched_pose(object_to_eef).to(
        device=eef_pose.device, dtype=torch.float32
    )
    return torch.mm(eef_pose, torch.inverse(object_to_eef))


def update_attached_object_pose(
    robot: Robot,
    obj: RigidObject,
    qpos: torch.Tensor,
    object_to_eef: torch.Tensor | None,
) -> None:
    """Move the object with the planned TCP pose in attached replay mode."""
    if object_to_eef is None:
        return
    arm_joint_ids = robot.get_joint_ids(ARM_NAME)
    eef_pose = compute_tcp_pose(robot, qpos[0, arm_joint_ids], control_part=ARM_NAME)
    obj_pose = compute_attached_object_pose(eef_pose, object_to_eef)
    obj.set_local_pose(obj_pose.unsqueeze(0), env_ids=[0])
    obj.clear_dynamics(env_ids=[0])


def save_replay_screenshot(
    sim: SimulationManager,
    *,
    screenshot_dir: str | None,
    planner: str,
    mode: str,
    repeat_id: int,
    label: str,
) -> None:
    """Save a diagnostic screenshot if a screenshot directory is configured."""
    if not screenshot_dir:
        return
    from PIL import Image
    import numpy as np

    screenshot_path = Path(screenshot_dir)
    screenshot_path.mkdir(parents=True, exist_ok=True)
    camera = sim.add_sensor(
        sensor_cfg=CameraCfg(
            uid=f"debug_camera_{planner}_{mode}_{repeat_id}_{label}",
            width=640,
            height=480,
            intrinsics=(520.0, 520.0, 320.0, 240.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(0.82, -0.72, 0.58),
                target=(0.24, 0.04, 0.10),
                up=(0.0, 0.0, 1.0),
            ),
            near=0.01,
            far=4.0,
            enable_color=True,
        )
    )
    sim.update(step=1)
    camera.update()
    color = camera.get_data()["color"][0, :, :, :3].detach().cpu().numpy()
    out_path = (
        screenshot_path / f"{SCRIPT_NAME}_{planner}_{mode}_{repeat_id}_{label}.png"
    )
    Image.fromarray(np.ascontiguousarray(color)).save(out_path)


def replay_trajectory(
    sim: SimulationManager,
    robot: Robot,
    obj: RigidObject,
    traj: torch.Tensor,
    object_target_pose: torch.Tensor,
    args: argparse.Namespace,
    *,
    held_object: HeldObjectState | None,
    planner: str,
    mode: str,
    repeat_id: int,
    hand_opening_spec: FrankaHandOpeningSpec,
) -> ReplayOutcome:
    """Replay a planned full-DoF trajectory with real object physics."""
    if traj.numel() == 0:
        return ReplayOutcome(
            0.0,
            False,
            False,
            False,
            False,
            None,
            None,
            None,
            None,
            args.object_replay_mode,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    start_pose = obj.get_local_pose(to_matrix=True)[0].clone()
    start_pos = start_pose[:3, 3].clone()
    max_z = float(start_pos[2].item())
    start = time.perf_counter()
    close_end_idx = _find_hand_close_end_index(robot, traj, hand_opening_spec)
    release_start_idx = _find_hand_release_start_index(
        robot,
        traj,
        hand_opening_spec,
    )
    object_to_eef = (
        None
        if held_object is None
        else _as_unbatched_pose(held_object.object_to_eef).to(robot.device)
    )
    opening = hand_opening_from_qpos(robot, traj[0], hand_opening_spec).reshape(-1)
    min_hand_opening = float(torch.min(opening).item())
    final_hand_opening = float(opening[-1].item())
    contact_sensor = (
        None
        if args.object_replay_mode != "physics"
        else make_finger_object_contact_sensor(
            sim,
            obj,
            uid_suffix=f"{planner}_{mode}_{repeat_id}",
        )
    )
    actual_min_hand_opening = current_hand_opening(robot, hand_opening_spec)
    actual_close_hand_opening = None
    close_contact_count = None
    path_contact_count = None
    hold_contact_count = None
    max_contact_count = 0
    max_contact_impulse = None
    max_total_contact_impulse = None
    min_contact_distance = None
    close_tcp_object_delta = None
    close_leftfinger_object_delta = None
    close_rightfinger_object_delta = None
    close_tcp_x_axis = None
    close_tcp_y_axis = None
    close_tcp_z_axis = None
    planned_grasp_object_delta = (
        None
        if close_end_idx is None
        else object_delta_from_tcp(robot, obj, traj[:, close_end_idx, :])
    )
    should_clear_object_dynamics = args.demo_profile == TUTORIAL_PLACE_PROFILE
    replay_control = effective_replay_control(args)
    try:
        for idx in range(traj.shape[1]):
            _set_replay_qpos(robot, traj[:, idx, :], mode=replay_control)
            if (
                args.object_replay_mode == "attached"
                and close_end_idx is not None
                and idx >= close_end_idx
                and (release_start_idx is None or idx < release_start_idx)
            ):
                update_attached_object_pose(robot, obj, traj[:, idx, :], object_to_eef)
            sim.update(step=max(args.step_repeat, 1))
            actual_min_hand_opening = min(
                actual_min_hand_opening,
                current_hand_opening(robot, hand_opening_spec),
            )
            contact_stats = summarize_contact_sensor(contact_sensor)
            if contact_sensor is not None:
                (
                    max_contact_count,
                    max_contact_impulse,
                    max_total_contact_impulse,
                    min_contact_distance,
                ) = _record_contact_stats(
                    contact_stats,
                    max_count=max_contact_count,
                    max_impulse=max_contact_impulse,
                    max_total_impulse=max_total_contact_impulse,
                    min_distance=min_contact_distance,
                )
            if close_end_idx is not None and idx == close_end_idx:
                close_contact_count = (
                    None if contact_sensor is None else contact_stats.count
                )
                actual_close_hand_opening = current_hand_opening(
                    robot,
                    hand_opening_spec,
                )
                close_tcp_object_delta = object_delta_from_tcp(
                    robot,
                    obj,
                    traj[:, idx, :],
                )
                close_leftfinger_object_delta = link_delta_from_object(
                    robot,
                    obj,
                    "leftfinger",
                )
                close_rightfinger_object_delta = link_delta_from_object(
                    robot,
                    obj,
                    "rightfinger",
                )
                (
                    close_tcp_x_axis,
                    close_tcp_y_axis,
                    close_tcp_z_axis,
                ) = tcp_axes_from_qpos(robot, traj[:, idx, :])
                save_replay_screenshot(
                    sim,
                    screenshot_dir=args.screenshot_dir,
                    planner=planner,
                    mode=mode,
                    repeat_id=repeat_id,
                    label="after_close",
                )
                if should_clear_object_dynamics:
                    obj.clear_dynamics(env_ids=[0])
                    should_clear_object_dynamics = False
                close_hold_steps = (
                    0
                    if args.demo_profile == TUTORIAL_PLACE_PROFILE
                    else args.grasp_hold_steps + POST_GRASP_HOLD_STEPS
                )
                if close_hold_steps > 0:
                    sim.update(step=close_hold_steps)
                if args.object_replay_mode == "attached":
                    update_attached_object_pose(
                        robot, obj, traj[:, idx, :], object_to_eef
                    )
                actual_min_hand_opening = min(
                    actual_min_hand_opening,
                    current_hand_opening(robot, hand_opening_spec),
                )
                contact_stats = summarize_contact_sensor(contact_sensor)
                if contact_sensor is not None:
                    close_contact_count = max(
                        close_contact_count or 0,
                        contact_stats.count,
                    )
                    (
                        max_contact_count,
                        max_contact_impulse,
                        max_total_contact_impulse,
                        min_contact_distance,
                    ) = _record_contact_stats(
                        contact_stats,
                        max_count=max_contact_count,
                        max_impulse=max_contact_impulse,
                        max_total_impulse=max_total_contact_impulse,
                        min_distance=min_contact_distance,
                    )
            obj_pos = obj.get_local_pose(to_matrix=True)[0, :3, 3]
            max_z = max(max_z, float(obj_pos[2].item()))

        path_contact_stats = summarize_contact_sensor(contact_sensor)
        path_contact_count = (
            None if contact_sensor is None else path_contact_stats.count
        )
        if contact_sensor is not None:
            (
                max_contact_count,
                max_contact_impulse,
                max_total_contact_impulse,
                min_contact_distance,
            ) = _record_contact_stats(
                path_contact_stats,
                max_count=max_contact_count,
                max_impulse=max_contact_impulse,
                max_total_impulse=max_total_contact_impulse,
                min_distance=min_contact_distance,
            )
        save_replay_screenshot(
            sim,
            screenshot_dir=args.screenshot_dir,
            planner=planner,
            mode=mode,
            repeat_id=repeat_id,
            label="after_path",
        )
        final_hold_steps = effective_final_hold_steps(args)
        if args.demo_profile == TUTORIAL_PLACE_PROFILE:
            for _ in range(final_hold_steps):
                _set_replay_qpos(robot, traj[:, -1, :], mode=replay_control)
                sim.update(step=2)
        else:
            _set_replay_qpos(robot, traj[:, -1, :], mode=replay_control)
            sim.update(step=final_hold_steps)
        actual_min_hand_opening = min(
            actual_min_hand_opening,
            current_hand_opening(robot, hand_opening_spec),
        )
        actual_final_hand_opening = current_hand_opening(robot, hand_opening_spec)
        hold_contact_stats = summarize_contact_sensor(contact_sensor)
        hold_contact_count = (
            None if contact_sensor is None else hold_contact_stats.count
        )
        if contact_sensor is not None:
            (
                max_contact_count,
                max_contact_impulse,
                max_total_contact_impulse,
                min_contact_distance,
            ) = _record_contact_stats(
                hold_contact_stats,
                max_count=max_contact_count,
                max_impulse=max_contact_impulse,
                max_total_impulse=max_total_contact_impulse,
                min_distance=min_contact_distance,
            )
        save_replay_screenshot(
            sim,
            screenshot_dir=args.screenshot_dir,
            planner=planner,
            mode=mode,
            repeat_id=repeat_id,
            label="after_hold",
        )
    finally:
        # SimulationManager currently keeps sensor assets alive for the lifetime
        # of the simulation, so each replay uses a unique contact sensor UID.
        pass
    elapsed = time.perf_counter() - start
    object_pos_error, object_rot_error = compute_object_pose_error(
        obj, object_target_pose
    )
    final_pos = obj.get_local_pose(to_matrix=True)[0, :3, 3]
    displacement = float(torch.linalg.norm(final_pos - start_pos).item())
    object_lifted = max_z - float(start_pos[2].item()) >= 0.03
    object_moved = displacement >= 0.03
    replay_success = (
        object_lifted
        and object_moved
        and object_pos_error <= args.object_pos_success_threshold
    )
    physical_success = replay_success and args.object_replay_mode == "physics"
    return ReplayOutcome(
        replay_time_sec=elapsed,
        replay_success=replay_success,
        physical_success=physical_success,
        object_lifted=object_lifted,
        object_moved=object_moved,
        object_pos_error=object_pos_error,
        object_rot_error=object_rot_error,
        object_max_z=max_z,
        object_displacement_m=displacement,
        object_replay_mode=args.object_replay_mode,
        close_end_index=close_end_idx,
        release_start_index=release_start_idx,
        min_hand_opening_m=min_hand_opening,
        final_hand_opening_m=final_hand_opening,
        actual_min_hand_opening_m=actual_min_hand_opening,
        actual_close_hand_opening_m=actual_close_hand_opening,
        actual_final_hand_opening_m=actual_final_hand_opening,
        close_contact_count=close_contact_count,
        path_contact_count=path_contact_count,
        hold_contact_count=hold_contact_count,
        max_contact_count=None if contact_sensor is None else max_contact_count,
        max_contact_impulse=max_contact_impulse,
        max_total_contact_impulse=max_total_contact_impulse,
        min_contact_distance_m=min_contact_distance,
        close_tcp_object_delta_m=close_tcp_object_delta,
        planned_grasp_object_delta_m=planned_grasp_object_delta,
        close_leftfinger_object_delta_m=close_leftfinger_object_delta,
        close_rightfinger_object_delta_m=close_rightfinger_object_delta,
        close_tcp_x_axis=close_tcp_x_axis,
        close_tcp_y_axis=close_tcp_y_axis,
        close_tcp_z_axis=close_tcp_z_axis,
    )


def trajectory_stats(robot: Robot, traj: torch.Tensor) -> dict[str, object]:
    """Compute joint path statistics for a full-DoF trajectory."""
    if traj.numel() == 0:
        return {
            "trajectory_steps": 0,
            "joint_path_length": 0.0,
            "max_joint_step": 0.0,
            "final_qpos": None,
        }
    arm_joint_ids = robot.get_joint_ids(ARM_NAME)
    arm_traj = traj[0, :, arm_joint_ids]
    deltas = torch.diff(arm_traj, dim=0)
    step_norms = (
        torch.linalg.norm(deltas, dim=-1)
        if deltas.numel()
        else torch.zeros(1, device=robot.device)
    )
    return {
        "trajectory_steps": int(traj.shape[1]),
        "joint_path_length": float(step_norms.sum().item()),
        "max_joint_step": float(step_norms.max().item()),
        "final_qpos": [float(v) for v in traj[0, -1, :].detach().cpu().tolist()],
    }


def _as_unbatched_pose(pose: torch.Tensor) -> torch.Tensor:
    """Return the first pose when a batched pose tensor is provided."""
    if pose.dim() == 3:
        return pose[0]
    return pose


def pose_error(
    actual_pose: torch.Tensor, target_pose: torch.Tensor
) -> tuple[float, float]:
    """Return position and rotation error between two pose matrices."""
    actual_pose = _as_unbatched_pose(actual_pose)
    target_pose = _as_unbatched_pose(target_pose)
    actual_quat = quat_from_matrix(actual_pose[:3, :3].unsqueeze(0))[0]
    target_quat = quat_from_matrix(target_pose[:3, :3].unsqueeze(0))[0]
    pos_error = float(torch.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3]).item())
    rot_error = float(
        quat_error_magnitude(
            target_quat.unsqueeze(0),
            actual_quat.unsqueeze(0),
        )[0].item()
    )
    return pos_error, rot_error


def compute_tcp_pose(
    robot: Robot, arm_qpos: torch.Tensor, *, control_part: str
) -> torch.Tensor:
    """Compute one unbatched TCP pose matrix from arm qpos."""
    return robot.compute_fk(
        qpos=arm_qpos.unsqueeze(0),
        name=control_part,
        to_matrix=True,
    )[0]


def compute_object_pose_error(
    obj: RigidObject,
    target_pose: torch.Tensor,
) -> tuple[float, float]:
    """Return object pose error to a target pose for env 0."""
    object_pose = obj.get_local_pose(to_matrix=True)[0]
    return pose_error(object_pose, target_pose)


def build_trial_row(
    *,
    planner: str,
    mode: str,
    repeat_id: int,
    warmup: bool,
    outcome: PlanOutcome,
    planning_time_sec: float,
    mem_delta: dict[str, float],
    peak_gpu_mb: float,
    robot: Robot,
    preflight: GraspPreflight,
    replay: ReplayOutcome | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Build one raw metrics row."""
    stats = trajectory_stats(robot, outcome.trajectory)
    final_tcp_pos_error = None
    final_tcp_rot_error = None
    strict_tcp_success = False
    if outcome.trajectory.numel() > 0:
        arm_joint_ids = robot.get_joint_ids(ARM_NAME)
        final_arm_qpos = outcome.trajectory[0, -1, arm_joint_ids]
        final_pose = compute_tcp_pose(robot, final_arm_qpos, control_part=ARM_NAME)
        final_tcp_pos_error, final_tcp_rot_error = pose_error(
            final_pose, outcome.rest_eef_pose
        )
        strict_tcp_success = (
            final_tcp_pos_error <= args.pos_success_threshold
            and final_tcp_rot_error <= args.rot_success_threshold
        )

    planner_success = bool(outcome.success and strict_tcp_success)
    physical_evaluated = replay is not None
    physical_success = bool(replay.physical_success) if replay is not None else None
    replay_success = bool(replay.replay_success) if replay is not None else None
    task_success = physical_success if mode == "physical" else planner_success

    row: dict[str, object] = {
        "script": SCRIPT_NAME,
        "planner": planner,
        "mode": mode,
        "demo_profile": args.demo_profile,
        "object": args.object,
        "support_surface": args.support_surface,
        "object_scale": list(
            resolve_object_body_scale(
                args.object,
                args.object_scale,
                args.demo_profile,
            )
        ),
        "gripper_closing_axis_index": FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
        "tcp_z": float(args.tcp_z),
        "tcp_yaw": float(args.tcp_yaw),
        "grasp_depth_offset": float(args.grasp_depth_offset),
        "gripper_close_margin": float(args.gripper_close_margin),
        "grasp_axis": args.grasp_axis,
        "repeat_id": repeat_id,
        "warmup": warmup,
        "action_success": bool(outcome.success),
        "planner_success": planner_success,
        "physical_evaluated": physical_evaluated,
        "physical_success": physical_success,
        "replay_success": replay_success,
        "task_success": bool(task_success),
        "planning_time_sec": float(planning_time_sec),
        "replay_time_sec": float(replay.replay_time_sec) if replay else 0.0,
        "object_replay_mode": replay.object_replay_mode if replay else None,
        "replay_control": effective_replay_control(args) if replay else None,
        "cpu_delta_mb": float(mem_delta["cpu_mb"]),
        "gpu_delta_mb": float(mem_delta["gpu_mb"]),
        "peak_gpu_mb": float(peak_gpu_mb),
        "final_tcp_pos_error": final_tcp_pos_error,
        "final_tcp_rot_error": final_tcp_rot_error,
        "strict_tcp_success": strict_tcp_success,
        "preflight_status": preflight.status,
        "preflight_failed": preflight.failed,
        "preflight_reason": preflight.reason,
        "object_grasp_width_m": preflight.object_grasp_width_m,
        "gripper_min_opening_m": preflight.gripper_min_opening_m,
        "gripper_max_opening_m": preflight.gripper_max_opening_m,
        "object_lifted": replay.object_lifted if replay else None,
        "object_moved": replay.object_moved if replay else None,
        "object_pos_error": replay.object_pos_error if replay else None,
        "object_rot_error": replay.object_rot_error if replay else None,
        "object_max_z": replay.object_max_z if replay else None,
        "object_displacement_m": replay.object_displacement_m if replay else None,
        "close_end_index": replay.close_end_index if replay else None,
        "release_start_index": replay.release_start_index if replay else None,
        "min_hand_opening_m": replay.min_hand_opening_m if replay else None,
        "final_hand_opening_m": replay.final_hand_opening_m if replay else None,
        "actual_min_hand_opening_m": (
            replay.actual_min_hand_opening_m if replay else None
        ),
        "actual_close_hand_opening_m": (
            replay.actual_close_hand_opening_m if replay else None
        ),
        "actual_final_hand_opening_m": (
            replay.actual_final_hand_opening_m if replay else None
        ),
        "close_contact_count": replay.close_contact_count if replay else None,
        "path_contact_count": replay.path_contact_count if replay else None,
        "hold_contact_count": replay.hold_contact_count if replay else None,
        "max_contact_count": replay.max_contact_count if replay else None,
        "max_contact_impulse": replay.max_contact_impulse if replay else None,
        "max_total_contact_impulse": (
            replay.max_total_contact_impulse if replay else None
        ),
        "min_contact_distance_m": replay.min_contact_distance_m if replay else None,
        "close_tcp_object_delta_m": replay.close_tcp_object_delta_m if replay else None,
        "planned_grasp_object_delta_m": (
            replay.planned_grasp_object_delta_m if replay else None
        ),
        "close_leftfinger_object_delta_m": (
            replay.close_leftfinger_object_delta_m if replay else None
        ),
        "close_rightfinger_object_delta_m": (
            replay.close_rightfinger_object_delta_m if replay else None
        ),
        "close_tcp_x_axis": replay.close_tcp_x_axis if replay else None,
        "close_tcp_y_axis": replay.close_tcp_y_axis if replay else None,
        "close_tcp_z_axis": replay.close_tcp_z_axis if replay else None,
    }
    row.update(stats)
    return row


def run_one_trial(
    *,
    sim: SimulationManager,
    robot: Robot,
    obj: RigidObject,
    franka_urdf_path: str,
    planner: str,
    mode: str,
    repeat_id: int,
    warmup: bool,
    checkpoint_path: str | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run one independent benchmark trial."""
    object_start_pose = reset_object_pose(
        obj,
        args.object,
        args.object_scale,
        args.support_surface,
        args.demo_profile,
        args.object_xy,
    )
    hand_open, _ = get_hand_open_close_qpos(robot)
    set_robot_start_qpos(robot, hand_open)
    try:
        initialize_pre_pick_robot_pose(
            robot,
            obj,
            hand_open,
            pre_pick_z=args.pre_pick_z,
            reference_pose=object_start_pose,
        )
    except RuntimeError as exc:
        reason = str(exc)
        logger.log_warning(
            f"{SCRIPT_NAME}: planner={planner} mode={mode} repeat={repeat_id} "
            f"warmup={warmup} setup_failed={reason}"
        )
        return make_failed_trial_row(
            planner=planner,
            mode=mode,
            object_name=args.object,
            repeat_id=repeat_id,
            warmup=warmup,
            reason=reason,
            support_surface=args.support_surface,
            demo_profile=args.demo_profile,
        )
    pre_plan_hold_steps = effective_pre_plan_hold_steps(args)
    if pre_plan_hold_steps > 0:
        sim.update(step=pre_plan_hold_steps)

    preflight = build_grasp_preflight(obj, robot, franka_urdf_path=franka_urdf_path)
    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        target_pose = make_tutorial_place_eef_pose(robot.device)
        rest_pose = make_tutorial_retract_eef_pose(target_pose)
    else:
        target_pose = make_object_target_pose(obj)
        rest_pose = make_rest_pose(robot)
    motion_gen = build_motion_generator(robot, planner, checkpoint_path, args)
    engine = build_engine(
        robot,
        motion_gen,
        args,
        preflight,
        franka_urdf_path=franka_urdf_path,
    )
    semantics = create_object_semantics(obj, args, preflight)

    outcome, planning_time_sec, mem_delta, peak_gpu_mb = timed_plan_sequence(
        engine,
        semantics,
        target_pose,
        rest_pose,
        demo_profile=args.demo_profile,
    )
    object_target_pose = target_pose
    if (
        args.demo_profile == TUTORIAL_PLACE_PROFILE
        and outcome.held_object is not None
        and outcome.place_eef_pose is not None
    ):
        object_target_pose = compute_attached_object_pose(
            outcome.place_eef_pose,
            outcome.held_object.object_to_eef,
        )

    replay = None
    can_replay = (
        mode == "physical"
        and outcome.success
        and (
            not preflight.failed
            or args.object_replay_mode == "attached"
            or args.demo_profile == TUTORIAL_PLACE_PROFILE
        )
    )
    if can_replay:
        hand_opening_spec = load_franka_hand_opening_spec(franka_urdf_path)
        replay = replay_trajectory(
            sim,
            robot,
            obj,
            outcome.trajectory,
            object_target_pose,
            args,
            held_object=outcome.held_object,
            planner=planner,
            mode=mode,
            repeat_id=repeat_id,
            hand_opening_spec=hand_opening_spec,
        )

    row = build_trial_row(
        planner=planner,
        mode=mode,
        repeat_id=repeat_id,
        warmup=warmup,
        outcome=outcome,
        planning_time_sec=planning_time_sec,
        mem_delta=mem_delta,
        peak_gpu_mb=peak_gpu_mb,
        robot=robot,
        preflight=preflight,
        replay=replay,
        args=args,
    )

    logger.log_info(
        f"{SCRIPT_NAME}: planner={planner} mode={mode} repeat={repeat_id} "
        f"warmup={warmup} planner_success={row['planner_success']} "
        f"physical_success={row['physical_success']} preflight={preflight.status}"
    )
    return row


def _jsonable(value: Any) -> Any:
    """Convert tensors and dataclasses to JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def empty_replay_diagnostics() -> dict[str, object]:
    """Return replay diagnostic fields for rows without physical replay."""
    return {
        "actual_min_hand_opening_m": None,
        "actual_close_hand_opening_m": None,
        "actual_final_hand_opening_m": None,
        "close_contact_count": None,
        "path_contact_count": None,
        "hold_contact_count": None,
        "max_contact_count": None,
        "max_contact_impulse": None,
        "max_total_contact_impulse": None,
        "min_contact_distance_m": None,
        "close_tcp_object_delta_m": None,
        "planned_grasp_object_delta_m": None,
        "close_leftfinger_object_delta_m": None,
        "close_rightfinger_object_delta_m": None,
        "close_tcp_x_axis": None,
        "close_tcp_y_axis": None,
        "close_tcp_z_axis": None,
    }


def make_skipped_rows(
    planners: list[str],
    modes: list[str],
    *,
    object_name: str,
    reason: str,
    support_surface: str | None = None,
    demo_profile: str | None = None,
) -> list[dict[str, object]]:
    """Build measured rows for a gracefully skipped live-simulation benchmark."""
    rows: list[dict[str, object]] = []
    for planner in planners:
        for mode in modes:
            rows.append(
                {
                    "script": SCRIPT_NAME,
                    "planner": planner,
                    "mode": mode,
                    "demo_profile": demo_profile,
                    "object": object_name,
                    "support_surface": support_surface,
                    "object_scale": None,
                    "gripper_closing_axis_index": FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
                    "tcp_z": None,
                    "tcp_yaw": None,
                    "grasp_depth_offset": None,
                    "gripper_close_margin": None,
                    "grasp_axis": None,
                    "repeat_id": 0,
                    "warmup": False,
                    "action_success": False,
                    "planner_success": False,
                    "physical_evaluated": False,
                    "physical_success": None,
                    "replay_success": None,
                    "task_success": False,
                    "planning_time_sec": 0.0,
                    "replay_time_sec": 0.0,
                    "object_replay_mode": None,
                    "replay_control": None,
                    "cpu_delta_mb": 0.0,
                    "gpu_delta_mb": 0.0,
                    "peak_gpu_mb": 0.0,
                    "final_tcp_pos_error": None,
                    "final_tcp_rot_error": None,
                    "strict_tcp_success": False,
                    "preflight_status": "skipped",
                    "preflight_failed": True,
                    "preflight_reason": reason,
                    "object_grasp_width_m": 0.0,
                    "gripper_min_opening_m": 0.0,
                    "gripper_max_opening_m": 0.0,
                    "object_lifted": None,
                    "object_moved": None,
                    "object_pos_error": None,
                    "object_rot_error": None,
                    "object_max_z": None,
                    "object_displacement_m": None,
                    "close_end_index": None,
                    "release_start_index": None,
                    "min_hand_opening_m": None,
                    "final_hand_opening_m": None,
                    **empty_replay_diagnostics(),
                    "trajectory_steps": 0,
                    "joint_path_length": 0.0,
                    "max_joint_step": 0.0,
                    "final_qpos": None,
                }
            )
    return rows


def make_failed_trial_row(
    *,
    planner: str,
    mode: str,
    object_name: str,
    repeat_id: int,
    warmup: bool,
    reason: str,
    support_surface: str | None = None,
    demo_profile: str | None = None,
) -> dict[str, object]:
    """Build one failed row when setup cannot reach the planner call."""
    return {
        "script": SCRIPT_NAME,
        "planner": planner,
        "mode": mode,
        "demo_profile": demo_profile,
        "object": object_name,
        "support_surface": support_surface,
        "object_scale": None,
        "gripper_closing_axis_index": FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
        "tcp_z": None,
        "tcp_yaw": None,
        "grasp_depth_offset": None,
        "gripper_close_margin": None,
        "grasp_axis": None,
        "repeat_id": repeat_id,
        "warmup": warmup,
        "action_success": False,
        "planner_success": False,
        "physical_evaluated": False,
        "physical_success": None,
        "replay_success": None,
        "task_success": False,
        "planning_time_sec": 0.0,
        "replay_time_sec": 0.0,
        "object_replay_mode": None,
        "replay_control": None,
        "cpu_delta_mb": 0.0,
        "gpu_delta_mb": 0.0,
        "peak_gpu_mb": 0.0,
        "final_tcp_pos_error": None,
        "final_tcp_rot_error": None,
        "strict_tcp_success": False,
        "preflight_status": "setup_failed",
        "preflight_failed": True,
        "preflight_reason": reason,
        "object_grasp_width_m": 0.0,
        "gripper_min_opening_m": 0.0,
        "gripper_max_opening_m": 0.0,
        "object_lifted": None,
        "object_moved": None,
        "object_pos_error": None,
        "object_rot_error": None,
        "object_max_z": None,
        "object_displacement_m": None,
        "close_end_index": None,
        "release_start_index": None,
        "min_hand_opening_m": None,
        "final_hand_opening_m": None,
        **empty_replay_diagnostics(),
        "trajectory_steps": 0,
        "joint_path_length": 0.0,
        "max_joint_step": 0.0,
        "final_qpos": None,
    }


def write_raw_jsonl(path: str | None, row: dict[str, object]) -> None:
    """Append one raw metrics row when requested."""
    if not path:
        return
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    """Return numeric values for a row key."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean of values."""
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float | None:
    """Return a linear-interpolated quantile."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(rows: list[dict[str, object]], key: str) -> float | None:
    """Return the true-rate for a boolean row key."""
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def _fmt_float(value: float | None, digits: int = 3) -> str:
    """Format a possibly-missing float."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_rate(value: float | None) -> str:
    """Format a possibly-missing rate."""
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def summarize_rows(rows: list[dict[str, object]]) -> list[SummaryRow]:
    """Group measured rows by planner and mode."""
    measured = [row for row in rows if not row.get("warmup", False)]
    keys = sorted({(str(row["planner"]), str(row["mode"])) for row in measured})
    return [
        SummaryRow(
            planner=planner,
            mode=mode,
            rows=[
                row
                for row in measured
                if row["planner"] == planner and row["mode"] == mode
            ],
        )
        for planner, mode in keys
    ]


def _preflight_status(rows: list[dict[str, object]]) -> str:
    """Return a compact aggregate preflight status."""
    statuses = sorted({str(row.get("preflight_status", "-")) for row in rows})
    if len(statuses) == 1:
        return statuses[0]
    return "mixed"


def _single_status(rows: list[dict[str, object]], key: str) -> str:
    """Return a compact aggregate string for a row key."""
    values = sorted({str(row.get(key, "-")) for row in rows})
    if len(values) == 1:
        return values[0]
    return "mixed"


def make_perf_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build Time & Memory report rows."""
    perf_rows = []
    for summary in summarize_rows(rows):
        times = _numeric_values(summary.rows, "planning_time_sec")
        perf_rows.append(
            {
                "planner": summary.planner,
                "mode": summary.mode,
                "repeat_count": len(summary.rows),
                "cost_time_ms_mean": _fmt_float(
                    None if _mean(times) is None else _mean(times) * 1000.0, 2
                ),
                "cost_time_ms_p50": _fmt_float(
                    (
                        None
                        if _quantile(times, 0.5) is None
                        else _quantile(times, 0.5) * 1000.0
                    ),
                    2,
                ),
                "cost_time_ms_p95": _fmt_float(
                    (
                        None
                        if _quantile(times, 0.95) is None
                        else _quantile(times, 0.95) * 1000.0
                    ),
                    2,
                ),
                "cpu_delta_mb": _fmt_float(
                    _mean(_numeric_values(summary.rows, "cpu_delta_mb")), 2
                ),
                "gpu_delta_mb": _fmt_float(
                    _mean(_numeric_values(summary.rows, "gpu_delta_mb")), 2
                ),
                "peak_gpu_mb": _fmt_float(
                    _mean(_numeric_values(summary.rows, "peak_gpu_mb")), 2
                ),
            }
        )
    return perf_rows


def make_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build Success & Other Metrics report rows."""
    metric_rows = []
    for summary in summarize_rows(rows):
        final_pos = _mean(_numeric_values(summary.rows, "final_tcp_pos_error"))
        final_rot = _mean(_numeric_values(summary.rows, "final_tcp_rot_error"))
        width = _mean(_numeric_values(summary.rows, "object_grasp_width_m"))
        min_open = _mean(_numeric_values(summary.rows, "gripper_min_opening_m"))
        max_open = _mean(_numeric_values(summary.rows, "gripper_max_opening_m"))
        replay_displacement = _mean(
            _numeric_values(summary.rows, "object_displacement_m")
        )
        actual_min_open = _mean(
            _numeric_values(summary.rows, "actual_min_hand_opening_m")
        )
        actual_close_open = _mean(
            _numeric_values(summary.rows, "actual_close_hand_opening_m")
        )
        max_contact_count = _mean(_numeric_values(summary.rows, "max_contact_count"))
        max_contact_impulse = _mean(
            _numeric_values(summary.rows, "max_contact_impulse")
        )
        metric_rows.append(
            {
                "planner": summary.planner,
                "mode": summary.mode,
                "demo_profile": _single_status(summary.rows, "demo_profile"),
                "support_surface": _single_status(summary.rows, "support_surface"),
                "object_replay_mode": _single_status(
                    summary.rows, "object_replay_mode"
                ),
                "action_success_rate": _fmt_rate(_rate(summary.rows, "action_success")),
                "planner_success_rate": _fmt_rate(
                    _rate(summary.rows, "planner_success")
                ),
                "replay_success_rate": _fmt_rate(_rate(summary.rows, "replay_success")),
                "physical_success_rate": _fmt_rate(
                    _rate(summary.rows, "physical_success")
                ),
                "preflight_failed_rate": _fmt_rate(
                    _rate(summary.rows, "preflight_failed")
                ),
                "preflight_status": _preflight_status(summary.rows),
                "object_grasp_width_m": _fmt_float(width, 4),
                "gripper_min_opening_m": _fmt_float(min_open, 4),
                "gripper_max_opening_m": _fmt_float(max_open, 4),
                "min_hand_opening_m": _fmt_float(
                    _mean(_numeric_values(summary.rows, "min_hand_opening_m")), 4
                ),
                "actual_min_hand_opening_m": _fmt_float(actual_min_open, 4),
                "actual_close_hand_opening_m": _fmt_float(actual_close_open, 4),
                "max_contact_count": _fmt_float(max_contact_count, 1),
                "max_contact_impulse": _fmt_float(max_contact_impulse, 4),
                "object_displacement_m": _fmt_float(replay_displacement, 4),
                "final_tcp_pos_err_mm": _fmt_float(
                    None if final_pos is None else final_pos * 1000.0, 3
                ),
                "final_tcp_rot_err_deg": _fmt_float(
                    None if final_rot is None else final_rot * 180.0 / math.pi, 3
                ),
                "joint_path_length": _fmt_float(
                    _mean(_numeric_values(summary.rows, "joint_path_length")), 4
                ),
                "max_joint_step": _fmt_float(
                    _mean(_numeric_values(summary.rows, "max_joint_step")), 4
                ),
            }
        )
    return metric_rows


def make_leaderboard_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build leaderboard rows sorted by the mode-specific primary metric."""
    leaderboard = []
    for summary in summarize_rows(rows):
        planner_rate = _rate(summary.rows, "planner_success") or 0.0
        task_rate = _rate(summary.rows, "task_success") or 0.0
        physical_rate = _rate(summary.rows, "physical_success")
        replay_rate = _rate(summary.rows, "replay_success")
        overall = task_rate if summary.mode == "physical" else planner_rate
        avg_time = _mean(_numeric_values(summary.rows, "planning_time_sec")) or 0.0
        avg_pos = _mean(_numeric_values(summary.rows, "final_tcp_pos_error"))
        leaderboard.append(
            {
                "planner": summary.planner,
                "mode": summary.mode,
                "overall_success_rate": overall,
                "planner_success_rate": planner_rate,
                "physical_success_rate": physical_rate,
                "replay_success_rate": replay_rate,
                "avg_cost_time_ms": avg_time * 1000.0,
                "avg_final_tcp_pos_err_mm": (
                    None if avg_pos is None else avg_pos * 1000.0
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -float(row["overall_success_rate"]),
            float(row["avg_cost_time_ms"]),
        )
    )
    return [
        {
            "rank": rank,
            "planner": row["planner"],
            "mode": row["mode"],
            "overall_success_rate": _fmt_rate(float(row["overall_success_rate"])),
            "planner_success_rate": _fmt_rate(float(row["planner_success_rate"])),
            "replay_success_rate": _fmt_rate(row["replay_success_rate"]),
            "physical_success_rate": _fmt_rate(row["physical_success_rate"]),
            "avg_cost_time_ms": _fmt_float(float(row["avg_cost_time_ms"]), 2),
            "avg_final_tcp_pos_err_mm": _fmt_float(row["avg_final_tcp_pos_err_mm"], 3),
        }
        for rank, row in enumerate(leaderboard, start=1)
    ]


def _markdown_table(rows: list[dict[str, object]]) -> list[str]:
    """Format rows into a Markdown table."""
    if not rows:
        rows = [{"status": "No rows were produced."}]
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return lines


def default_report_path() -> Path:
    """Return the default timestamped Markdown report path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/benchmarks") / f"{SCRIPT_NAME}_{timestamp}.md"


def write_markdown_report(
    rows: list[dict[str, object]],
    report_path: str | None = None,
) -> Path:
    """Write the benchmark Markdown report with exactly three tables."""
    path = Path(report_path) if report_path else default_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# {SCRIPT_NAME} Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Time & Memory",
        "",
        *_markdown_table(make_perf_rows(rows)),
        "",
        "## Success & Other Metrics",
        "",
        *_markdown_table(make_metric_rows(rows)),
        "",
        "## Leaderboard",
        "",
        *_markdown_table(make_leaderboard_rows(rows)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def selected_run_kind(args: argparse.Namespace) -> str:
    """Resolve the effective run kind from CLI args."""
    if args.run_kind != "auto":
        return args.run_kind
    if args.open_window:
        return "demo"
    return "benchmark"


def run_demo(args: argparse.Namespace) -> dict[str, object]:
    """Run one tutorial-style PickUp -> Place sequence without benchmark repeats."""
    validate_args(args)
    torch.manual_seed(args.seed)
    planner = expand_planner_selection(args.planner)[0]
    checkpoint_path = resolve_checkpoint(args, [planner])

    sim = make_sim(args)
    robot, franka_urdf_path = create_franka(
        sim,
        tcp_z=args.tcp_z,
        tcp_yaw=args.tcp_yaw,
    )
    if args.support_surface == "table":
        create_support_table(sim)
        create_visual_support_table(sim)
    obj = create_object(
        sim,
        args.object,
        args.object_scale,
        args.support_surface,
        args.demo_profile,
        args.object_xy,
    )

    if not args.headless:
        sim.open_window()

    hand_open, _ = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(
        robot,
        obj,
        hand_open,
        pre_pick_z=args.pre_pick_z,
    )

    preflight = build_grasp_preflight(obj, robot, franka_urdf_path=franka_urdf_path)
    motion_gen = build_motion_generator(robot, planner, checkpoint_path, args)
    engine = build_engine(
        robot,
        motion_gen,
        args,
        preflight,
        franka_urdf_path=franka_urdf_path,
    )
    semantics = create_object_semantics(obj, args, preflight)

    if args.demo_profile == TUTORIAL_PLACE_PROFILE:
        target_pose = make_tutorial_place_eef_pose(robot.device)
        rest_pose = make_tutorial_retract_eef_pose(target_pose)
    else:
        target_pose = make_object_target_pose(obj)
        rest_pose = make_rest_pose(robot)

    if not args.headless:
        draw_axis_marker(sim, "franka_pick_place_target_axis", target_pose)
    pause_for_tutorial_inspection(args)
    inspect_viewer_before_trials(sim, args)

    logger.log_info(
        f"{SCRIPT_NAME}: running demo planner={planner} "
        f"support_surface={args.support_surface} replay_mode={args.demo_object_replay_mode}"
    )
    outcome, planning_time_sec, mem_delta, peak_gpu_mb = timed_plan_sequence(
        engine,
        semantics,
        target_pose,
        rest_pose,
        demo_profile=args.demo_profile,
    )
    if not outcome.success:
        logger.log_warning(f"{SCRIPT_NAME}: demo planning failed.")
        return build_trial_row(
            planner=planner,
            mode="demo",
            repeat_id=0,
            warmup=False,
            outcome=outcome,
            planning_time_sec=planning_time_sec,
            mem_delta=mem_delta,
            peak_gpu_mb=peak_gpu_mb,
            robot=robot,
            preflight=preflight,
            replay=None,
            args=args,
        )

    if not args.auto_play and sys.stdin.isatty():
        input("Press Enter to replay the Franka PickUp -> Place demo...")

    object_target_pose = target_pose
    if (
        args.demo_profile == TUTORIAL_PLACE_PROFILE
        and outcome.held_object is not None
        and outcome.place_eef_pose is not None
    ):
        object_target_pose = compute_attached_object_pose(
            outcome.place_eef_pose,
            outcome.held_object.object_to_eef,
        )

    replay_args = argparse.Namespace(**vars(args))
    replay_args.object_replay_mode = args.demo_object_replay_mode
    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix=f"{SCRIPT_NAME}_demo",
    )
    try:
        replay = replay_trajectory(
            sim,
            robot,
            obj,
            outcome.trajectory,
            object_target_pose,
            replay_args,
            held_object=outcome.held_object,
            planner=planner,
            mode="demo",
            repeat_id=0,
            hand_opening_spec=load_franka_hand_opening_spec(franka_urdf_path),
        )
    finally:
        stop_auto_play_recording(sim, recording_started)

    row = build_trial_row(
        planner=planner,
        mode="demo",
        repeat_id=0,
        warmup=False,
        outcome=outcome,
        planning_time_sec=planning_time_sec,
        mem_delta=mem_delta,
        peak_gpu_mb=peak_gpu_mb,
        robot=robot,
        preflight=preflight,
        replay=replay,
        args=replay_args,
    )
    logger.log_info(
        f"{SCRIPT_NAME}: demo planner_success={row['planner_success']} "
        f"replay_success={row['replay_success']} preflight={preflight.status}"
    )
    if not args.auto_play and sys.stdin.isatty():
        input("Press Enter to exit the simulation...")
    return row


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the selected planner benchmark trials."""
    validate_args(args)
    torch.manual_seed(args.seed)
    planners = expand_planner_selection(args.planner)
    modes = expand_mode_selection(args.mode)
    rows: list[dict[str, object]] = []

    if args.save_raw_jsonl:
        raw_path = Path(args.save_raw_jsonl)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("", encoding="utf-8")

    if simulation_requires_cuda(args) and not torch.cuda.is_available():
        reason = (
            "CUDA is unavailable to PyTorch, so the live DexSim benchmark was "
            "skipped before SimulationManager initialization."
        )
        logger.log_warning(reason)
        rows = make_skipped_rows(
            planners,
            modes,
            object_name=args.object,
            reason=reason,
            support_surface=args.support_surface,
            demo_profile=args.demo_profile,
        )
        for row in rows:
            write_raw_jsonl(args.save_raw_jsonl, row)
        report_path = write_markdown_report(rows, args.report_path)
        logger.log_info(f"Markdown benchmark report saved: {report_path}")
        return rows

    skip_reason = physical_gpu_memory_skip_reason(args, modes)
    if skip_reason is not None:
        logger.log_warning(skip_reason)
        rows = make_skipped_rows(
            planners,
            modes,
            object_name=args.object,
            reason=skip_reason,
            support_surface=args.support_surface,
            demo_profile=args.demo_profile,
        )
        for row in rows:
            write_raw_jsonl(args.save_raw_jsonl, row)
        report_path = write_markdown_report(rows, args.report_path)
        logger.log_info(f"Markdown benchmark report saved: {report_path}")
        return rows

    checkpoint_path = resolve_checkpoint(args, planners)

    sim = make_sim(args)
    robot, franka_urdf_path = create_franka(
        sim,
        tcp_z=args.tcp_z,
        tcp_yaw=args.tcp_yaw,
    )
    if args.support_surface == "table":
        create_support_table(sim)
        create_visual_support_table(sim)
    obj = create_object(
        sim,
        args.object,
        args.object_scale,
        args.support_surface,
        args.demo_profile,
        args.object_xy,
    )
    if not args.headless:
        sim.open_window()
    pause_for_tutorial_inspection(args)
    inspect_viewer_before_trials(sim, args)

    for planner in planners:
        for mode in modes:
            for warmup in (True, False):
                repeat_count = args.warmup_repeats if warmup else args.num_repeats
                for repeat_id in range(repeat_count):
                    row = run_one_trial(
                        sim=sim,
                        robot=robot,
                        obj=obj,
                        franka_urdf_path=franka_urdf_path,
                        planner=planner,
                        mode=mode,
                        repeat_id=repeat_id,
                        warmup=warmup,
                        checkpoint_path=checkpoint_path,
                        args=args,
                    )
                    rows.append(row)
                    write_raw_jsonl(args.save_raw_jsonl, row)

    report_path = write_markdown_report(rows, args.report_path)
    logger.log_info(f"Markdown benchmark report saved: {report_path}")
    for row in make_leaderboard_rows(rows):
        logger.log_info(json.dumps(row, sort_keys=True))
    return rows


def run_all_benchmarks() -> None:
    """Parse CLI args and run the selected Franka pick-place path."""
    args = parse_args()
    if selected_run_kind(args) == "demo":
        run_demo(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    run_all_benchmarks()
