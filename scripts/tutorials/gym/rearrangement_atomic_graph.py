#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from embodichain.agents.hierarchy import llm as agent_llm
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.lab.scripts.run_env import generate_and_execute_action_list
from embodichain.utils import logger
from embodichain.utils.utility import load_json

DEFAULT_GYM_CONFIG = (
    ROOT_DIR / "configs/gym/agent/rearrangement_agent/fast_gym_config.json"
)
DEFAULT_AGENT_CONFIG = (
    ROOT_DIR / "configs/gym/agent/rearrangement_agent/agent_config.json"
)
DEFAULT_DATABASE_ARTIFACT_ROOT = (
    ROOT_DIR / "embodichain/database/agent_generated_content/Rearrangement"
)
TASK_NAME = "Rearrangement"
FORCED_ERROR_PRESETS: dict[str, dict[str, Any]] = {
    "hold_lost_move_regrasp": {
        "edge_index": 3,
        "blind": False,
        "blind_obj_name": "fork",
        "error_type": "misplaced_object",
        "relative_error_xyz": [0.10, 0.0, 0.0],
        "expected_monitor": "monitor_object_held",
        "require_precompiled_recovery": True,
        "require_task_success": True,
        "require_layout_success": True,
    },
    "plate_moved_retry": {
        "edge_index": 3,
        "blind": True,
        "blind_obj_name": "plate",
        "error_type": "misplaced_object",
        "relative_error_xyz": [0.08, 0.0, 0.0],
        "require_precompiled_recovery": True,
        "require_task_success": True,
        "require_layout_success": True,
    },
    "legacy_monitor_index": {
        "edge_index": 1,
        "blind": False,
        "blind_obj_name": "fork",
        "error_type": "misplaced_object",
        "relative_error_xyz": [0.0, 0.1, 0.0],
        "require_precompiled_recovery": False,
        "require_task_success": False,
        "require_layout_success": False,
    },
}


class _OfflineLLM:
    """Fail fast if cached artifacts are missing and a prompt would call an LLM."""

    def invoke(self, prompt):
        raise RuntimeError(
            "Offline Rearrangement demo expected cached agent artifacts. "
            "Use --runtime_llm_recovery or provide fresh artifacts if LLM calls are needed."
        )


def _ensure_offline_llms(*, force: bool = False) -> None:
    offline_llm = _OfflineLLM()
    if force or agent_llm.task_llm is None:
        agent_llm.task_llm = offline_llm
    if force or agent_llm.recovery_llm is None:
        agent_llm.recovery_llm = offline_llm
    if force or agent_llm.compile_llm is None:
        agent_llm.compile_llm = offline_llm
    if force or agent_llm.failure_anticipation_llm is None:
        agent_llm.failure_anticipation_llm = agent_llm.recovery_llm
    if agent_llm.code_llm is None:
        agent_llm.code_llm = agent_llm.compile_llm


def _timestamped_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return ROOT_DIR / "outputs" / f"{timestamp}_rearrangement_atomic_graph"


def _parse_xyz(value: str) -> list[float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    return parts


def _parse_xyz_list(value: str) -> list[list[float]]:
    directions = [_parse_xyz(part.strip()) for part in value.split(";") if part.strip()]
    if not directions:
        raise argparse.ArgumentTypeError(
            "Expected one or more xyz vectors separated by semicolons."
        )
    return directions


def _parse_optional_bool(value: str | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _disable_video_events(env_cfg) -> None:
    env_cfg.filter_dataset_saving = True
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for event_name in ("record_camera", "validation_cameras"):
        if hasattr(events, event_name):
            setattr(events, event_name, None)


def _disable_validation_camera_events(env_cfg) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if hasattr(events, "validation_cameras"):
        setattr(events, "validation_cameras", None)


def _set_record_video_output_dir(env_cfg, save_path: Path) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    record_camera = getattr(events, "record_camera", None)
    if record_camera is None:
        return
    params = getattr(record_camera, "params", None)
    if isinstance(params, dict):
        params["save_path"] = str(save_path)


def _flush_recorded_videos(env) -> int:
    raw_env = getattr(env, "unwrapped", env)
    event_manager = getattr(raw_env, "event_manager", None)
    if event_manager is None:
        return 0

    try:
        from embodichain.lab.gym.envs.managers.record import record_camera_data
    except Exception:
        return 0

    flushed = 0
    for mode_cfgs in getattr(event_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            if not isinstance(functor, record_camera_data):
                continue
            if not getattr(functor, "_frames", []):
                continue
            functor.save_and_clear()
            flushed += 1
    return flushed


def _copy_database_artifacts(artifact_dir: Path, regenerate: bool) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if regenerate:
        return
    if not DEFAULT_DATABASE_ARTIFACT_ROOT.exists():
        raise FileNotFoundError(
            f"Rearrangement database artifacts not found: {DEFAULT_DATABASE_ARTIFACT_ROOT}"
        )
    shutil.copytree(DEFAULT_DATABASE_ARTIFACT_ROOT, artifact_dir, dirs_exist_ok=True)
    logger.log_info(
        f"Copied Rearrangement agent artifacts from {DEFAULT_DATABASE_ARTIFACT_ROOT} "
        f"to {artifact_dir}.",
        color="cyan",
    )


def _build_forced_error_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.forced_error_injection:
        return None
    preset = dict(FORCED_ERROR_PRESETS[args.forced_error_preset])
    edge_index = (
        int(args.error_injection_edge_index)
        if args.error_injection_edge_index is not None
        else int(preset.get("edge_index", 1))
    )
    error_type = (
        preset.get("error_type", "misplaced_object")
        if args.error_injection_type == "auto"
        else args.error_injection_type
    )
    relative_error_xyz = (
        args.error_injection_offset
        if args.error_injection_offset is not None
        else list(preset.get("relative_error_xyz", [0.0, 0.1, 0.0]))
    )
    blind_obj_name = args.error_injection_object or preset.get("blind_obj_name", "fork")
    return {
        "enabled": True,
        "preset": args.forced_error_preset,
        "edge_index": edge_index,
        "step_index": args.error_injection_step,
        "relative_error_xyz": relative_error_xyz,
        "error_type": error_type,
        "require_precompiled_recovery": bool(
            preset.get("require_precompiled_recovery", False)
        ),
        "require_task_success": bool(preset.get("require_task_success", False)),
        "require_layout_success": bool(preset.get("require_layout_success", False)),
        "expected_monitor": preset.get("expected_monitor"),
        "blind": bool(preset.get("blind", False)),
        "blind_obj_name": blind_obj_name,
    }


def _compiled_graph_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    action_names: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            kind = value.get("kind")
            if kind is not None:
                counts[kind] = counts.get(kind, 0) + 1
                if kind in {"atomic_action", "atomic_sequence"}:
                    action_names.append(str(value.get("name", kind)))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return {
        "exists": True,
        "schema_version": data.get("metadata", {}).get("schema_version"),
        "kind_counts": counts,
        "atomic_action_names": action_names,
    }


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def _write_result(
    *,
    artifact_dir: Path,
    program_success: bool,
    task_success: bool | None,
    forced_error_config: dict[str, Any] | None = None,
    env=None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    graph_path = artifact_dir / "agent_compiled_graph.json"
    layout = _rearrangement_layout_summary(env)
    result = {
        "task": TASK_NAME,
        "program_success": program_success,
        "task_success": task_success,
        "layout": layout,
        "compiled_graph": _compiled_graph_stats(graph_path),
        "forced_error": _forced_error_summary(forced_error_config),
        "exception": (
            None
            if exception is None
            else {"type": type(exception).__name__, "message": str(exception)}
        ),
    }
    result_path = artifact_dir / "case_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.log_info(f"Rearrangement result written to {result_path}", color="green")
    return result


def _forced_error_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "enabled": bool(config.get("enabled", False)),
        "preset": config.get("preset"),
        "edge_index": config.get("edge_index"),
        "blind": bool(config.get("blind", False)),
        "blind_obj_name": config.get("blind_obj_name"),
        "error_type": config.get("error_type"),
        "relative_error_xyz": config.get("relative_error_xyz"),
        "require_precompiled_recovery": bool(
            config.get("require_precompiled_recovery", False)
        ),
        "require_task_success": bool(config.get("require_task_success", False)),
        "require_layout_success": bool(config.get("require_layout_success", False)),
        "expected_monitor": config.get("expected_monitor"),
        "injected": bool(config.get("_injected", False)),
        "triggered": bool(config.get("_triggered", False)),
        "injected_at_step": config.get("_injected_at_step"),
        "triggered_step": config.get("_triggered_step"),
        "injected_monitor": config.get("_injected_monitor"),
        "triggered_monitor": config.get("_triggered_monitor_name"),
        "validation": config.get("_validation"),
    }


def _validate_forced_recovery_demo(
    *,
    forced_error_config: dict[str, Any],
    task_success: bool | None,
    layout: dict[str, Any] | None,
) -> None:
    validation = {
        "monitor_triggered": bool(forced_error_config.get("_triggered", False)),
        "task_success": task_success,
        "layout_success": None if layout is None else layout.get("success_xy"),
        "passed": True,
        "failure_reasons": [],
    }
    if not validation["monitor_triggered"]:
        validation["failure_reasons"].append("forced monitor did not trigger")
    expected_monitor = forced_error_config.get("expected_monitor")
    triggered_monitor = forced_error_config.get("_triggered_monitor_name")
    if expected_monitor and triggered_monitor != expected_monitor:
        validation["failure_reasons"].append(
            f"expected monitor {expected_monitor}, got {triggered_monitor}"
        )
    if (
        forced_error_config.get("require_task_success", False)
        and task_success is not True
    ):
        validation["failure_reasons"].append("task did not succeed after recovery")
    if (
        forced_error_config.get("require_layout_success", False)
        and validation["layout_success"] is not True
    ):
        validation["failure_reasons"].append(
            "final fork/spoon xy layout is not aligned with the moved plate"
        )

    validation["passed"] = not validation["failure_reasons"]
    forced_error_config["_validation"] = validation
    if not validation["passed"]:
        raise RuntimeError(
            "Forced recovery demo validation failed: "
            + "; ".join(validation["failure_reasons"])
        )


def _rearrangement_layout_summary(env) -> dict[str, Any] | None:
    if env is None:
        return None
    raw_env = getattr(env, "unwrapped", env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return None
    try:
        plate_xy = _object_xy(raw_env, "plate")
        fork_xy = _object_xy(raw_env, "fork")
        spoon_xy = _object_xy(raw_env, "spoon")
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    tolerance = float(
        getattr(raw_env, "metadata", {})
        .get("success_params", {})
        .get("tolerance", 0.02)
    )
    expected = {
        "fork": [plate_xy[0], plate_xy[1] + 0.16],
        "spoon": [plate_xy[0], plate_xy[1] - 0.16],
    }
    errors = {
        "fork": _xy_error(fork_xy, expected["fork"]),
        "spoon": _xy_error(spoon_xy, expected["spoon"]),
    }
    success_xy = all(
        abs(error["dx"]) <= tolerance and abs(error["dy"]) <= tolerance
        for error in errors.values()
    )
    success_y_only = all(abs(error["dy"]) <= tolerance for error in errors.values())
    return {
        "available": True,
        "tolerance": tolerance,
        "positions_xy": {
            "plate": plate_xy,
            "fork": fork_xy,
            "spoon": spoon_xy,
        },
        "expected_xy": expected,
        "errors": errors,
        "success_xy": success_xy,
        "success_y_only": success_y_only,
    }


def _object_xy(env, obj_name: str) -> list[float]:
    pose = env.sim.get_rigid_object(obj_name).get_local_pose(to_matrix=True)
    tensor = torch.as_tensor(pose).detach().cpu()
    return [float(tensor[0, 0, 3].item()), float(tensor[0, 1, 3].item())]


def _xy_error(actual: list[float], expected: list[float]) -> dict[str, float]:
    dx = float(actual[0] - expected[0])
    dy = float(actual[1] - expected[1])
    return {
        "dx": dx,
        "dy": dy,
        "distance_xy": float((dx * dx + dy * dy) ** 0.5),
    }


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RearrangementAgent demo from cached agent JSON artifacts and "
            "recompile recovery actions through the atomic graph runtime."
        )
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "outputs/YYYYMMDD_HHMM_rearrangement_atomic_graph."
        ),
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default=str(DEFAULT_AGENT_CONFIG),
        help="Path to the Rearrangement agent config JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for env.reset().",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        default=False,
        help="Do not copy cached database artifacts; allow agents to regenerate JSON.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        default=False,
        help="Only generate/recompile agent graph artifacts without executing actions.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Enable record/validation camera events. Disabled by default for low memory.",
    )
    parser.add_argument(
        "--open_window",
        action="store_true",
        default=False,
        help="Open the simulator window.",
    )
    parser.add_argument(
        "--runtime_llm_recovery",
        action="store_true",
        default=False,
        help="Allow runtime recovery to call the RecoveryAgent instead of using only cached branches.",
    )
    parser.add_argument(
        "--prefer_runtime_llm_recovery",
        action="store_true",
        default=False,
        help=(
            "Prefer runtime LLM recovery when a monitor triggers. "
            "Enabled automatically by --runtime_llm_recovery."
        ),
    )
    parser.add_argument(
        "--recovery_max_monitor_attempts",
        type=int,
        default=2,
        help=(
            "Maximum static recovery attempts for the same edge/monitor pair. "
            "Use 0 to keep retrying until graph max_transitions."
        ),
    )
    parser.add_argument(
        "--recovery_max_total_attempts",
        type=int,
        default=8,
        help=(
            "Maximum total static recovery attempts in one graph execution. "
            "Use 0 to keep retrying until graph max_transitions."
        ),
    )
    parser.add_argument(
        "--disable_atomic_action_graph",
        action="store_true",
        default=False,
        help="Disable atomic_action/atomic_sequence graph execution.",
    )
    parser.add_argument(
        "--require_atomic_action_graph",
        action="store_true",
        default=False,
        help="Fail instead of falling back if an atomic graph action cannot execute.",
    )
    parser.add_argument(
        "--disable_public_grasp_candidate_generation",
        action="store_true",
        default=False,
        help=(
            "Require an existing public grasp annotation cache instead of "
            "generating candidates offline."
        ),
    )
    parser.add_argument(
        "--recovery_public_grasp_strategy",
        choices=[
            "none",
            "top_down",
            "bottle_lateral",
            "lateral_down",
            "legacy_guided",
            "auto_try_all",
            "auto_general",
        ],
        default="auto_general",
        help=(
            "Recovery-only semantic grasp strategy for compiled atomic pick_up "
            "actions."
        ),
    )
    parser.add_argument(
        "--recovery_public_grasp_candidate_num",
        type=int,
        default=64,
        help="Recovery-only semantic grasp candidate count override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pre_grasp_distance",
        type=float,
        default=None,
        help="Recovery-only public PickUpActionCfg pre_grasp_distance override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_lift_height",
        type=float,
        default=None,
        help="Recovery-only lift_height passed to compiled atomic pick_up actions.",
    )
    parser.add_argument(
        "--recovery_public_grasp_auto_approach_direction",
        nargs="?",
        const="true",
        default=None,
        type=_parse_optional_bool,
        help="Recovery-only auto approach direction override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_try_approach_directions",
        nargs="?",
        const="true",
        default=None,
        type=_parse_optional_bool,
        help="Recovery-only multi-direction approach override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_approach_direction",
        type=_parse_xyz,
        default=None,
        help="Recovery-only explicit semantic grasp approach direction.",
    )
    parser.add_argument(
        "--recovery_public_grasp_approach_directions",
        type=_parse_xyz_list,
        default=None,
        help=(
            "Recovery-only approach directions, e.g. 0,0,-1;1,0,0. "
            "Overrides strategy-generated directions."
        ),
    )
    parser.add_argument(
        "--recovery_public_grasp_pose_offset_world",
        type=_parse_xyz,
        default=None,
        help="Recovery-only world-frame xyz offset for semantic grasp poses.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pose_offset_along_approach",
        type=float,
        default=None,
        help="Recovery-only offset along semantic grasp approach direction.",
    )
    parser.add_argument(
        "--recovery_validate_public_grasp_after_action",
        nargs="?",
        const="true",
        default=False,
        type=_parse_optional_bool,
        help="Recovery-only physical validation override after atomic pick_up.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_min_object_lift",
        type=float,
        default=None,
        help="Recovery-only minimum object z displacement for validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_max_object_lift",
        type=float,
        default=None,
        help="Recovery-only maximum object z displacement for validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=None,
        help="Recovery-only maximum object xy displacement for validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_rank_by_legacy_pose",
        nargs="?",
        const="true",
        default=False,
        type=_parse_optional_bool,
        help="Recovery-only legacy-pose candidate ranking override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_use_legacy_orientation",
        nargs="?",
        const="true",
        default=False,
        type=_parse_optional_bool,
        help="Recovery-only legacy-orientation candidate override.",
    )
    parser.add_argument("--grasp_max_open_length", type=float, default=0.088)
    parser.add_argument("--grasp_min_open_length", type=float, default=0.003)
    parser.add_argument("--grasp_finger_length", type=float, default=0.078)
    parser.add_argument("--grasp_x_thickness", type=float, default=0.01)
    parser.add_argument("--grasp_y_thickness", type=float, default=0.03)
    parser.add_argument("--grasp_root_z_width", type=float, default=0.08)
    parser.add_argument("--grasp_open_check_margin", type=float, default=0.01)
    parser.add_argument("--grasp_point_sample_dense", type=float, default=0.012)
    parser.add_argument("--grasp_antipodal_n_sample", type=int, default=20000)
    parser.add_argument("--grasp_collision_query_batch_size", type=int, default=512)
    parser.add_argument(
        "--forced_error_injection",
        action="store_true",
        default=False,
        help="Inject a deterministic error to exercise recovery branches.",
    )
    parser.add_argument(
        "--forced_error_preset",
        choices=sorted(FORCED_ERROR_PRESETS),
        default="hold_lost_move_regrasp",
        help=(
            "Deterministic recovery demo preset. hold_lost_move_regrasp "
            "displaces the held fork on monitored edge #3 so the precompiled "
            "e03 hold-lost branch can regrasp, replay e02, and retry e03; "
            "plate_moved_retry keeps the earlier blind plate displacement "
            "retry scenario; "
            "legacy_monitor_index keeps the old monitored-edge index behavior."
        ),
    )
    parser.add_argument(
        "--error_injection_edge_index",
        type=int,
        default=None,
        help=(
            "Override the preset 1-based edge index. For blind presets this counts "
            "all graph edges; otherwise it counts monitored edges."
        ),
    )
    parser.add_argument(
        "--error_injection_step",
        type=int,
        default=-1,
        help="Step index within the selected edge; negative means the final step.",
    )
    parser.add_argument(
        "--error_injection_offset",
        type=_parse_xyz,
        default=None,
        help="Injected xyz displacement, for example 0,0.1,0.",
    )
    parser.add_argument(
        "--error_injection_type",
        choices=["auto", "misplaced_object", "fallen_object"],
        default="auto",
        help="Error primitive used for forced recovery injection.",
    )
    parser.add_argument(
        "--error_injection_object",
        default=None,
        help="Object used by blind forced recovery injection.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Optional max episode steps override.",
    )

    for action in parser._actions:
        if action.dest == "gym_config":
            action.required = False
            action.default = str(DEFAULT_GYM_CONFIG)
            break

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.set_defaults(
        device=default_device,
        headless=True,
        filter_visual_rand=True,
        filter_dataset_saving=True,
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    if args.open_window:
        args.headless = False
    _ensure_offline_llms(force=not args.runtime_llm_recovery)

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _timestamped_output_root()
    )
    case_dir = output_root / "rearrangement"
    artifact_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    _copy_database_artifacts(artifact_dir, args.regenerate)

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    if not args.record_video:
        _disable_video_events(env_cfg)
    else:
        _disable_validation_camera_events(env_cfg)
        _set_record_video_output_dir(env_cfg, case_dir / "outputs" / "videos")
    if args.max_episode_steps is not None:
        env_cfg.max_episode_steps = int(args.max_episode_steps)
        gym_config["max_episode_steps"] = int(args.max_episode_steps)

    agent_config = load_json(args.agent_config)
    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=TASK_NAME,
        **action_config,
    )

    forced_error_config = _build_forced_error_config(args)
    task_success = None
    try:
        if (
            forced_error_config is not None
            and forced_error_config.get("require_precompiled_recovery", False)
            and args.runtime_llm_recovery
        ):
            raise ValueError(
                "--runtime_llm_recovery conflicts with static forced recovery presets. "
                "Use --forced_error_preset legacy_monitor_index for runtime recovery tests."
            )
        env.reset(seed=args.seed)
        common_kwargs = {
            "regenerate": args.regenerate,
            "recovery": True,
            "runtime_llm_recovery": args.runtime_llm_recovery,
            "prefer_runtime_llm_recovery": (
                args.prefer_runtime_llm_recovery or args.runtime_llm_recovery
            ),
            "recovery_max_monitor_attempts": args.recovery_max_monitor_attempts,
            "recovery_max_total_attempts": args.recovery_max_total_attempts,
            "runtime_recovery_max_monitor_attempts": (
                args.recovery_max_monitor_attempts
            ),
            "runtime_recovery_max_total_attempts": (args.recovery_max_total_attempts),
            "forced_recovery_injection": forced_error_config,
            "use_atomic_action_graph": not args.disable_atomic_action_graph,
            "require_atomic_action_graph": args.require_atomic_action_graph,
            "generate_public_grasp_candidates": (
                not args.disable_public_grasp_candidate_generation
            ),
            "recovery_public_grasp_strategy": args.recovery_public_grasp_strategy,
            "recovery_public_grasp_candidate_num": (
                args.recovery_public_grasp_candidate_num
            ),
            "recovery_public_grasp_pre_grasp_distance": (
                args.recovery_public_grasp_pre_grasp_distance
            ),
            "recovery_public_grasp_lift_height": (
                args.recovery_public_grasp_lift_height
            ),
            "recovery_public_grasp_auto_approach_direction": (
                args.recovery_public_grasp_auto_approach_direction
            ),
            "recovery_public_grasp_try_approach_directions": (
                args.recovery_public_grasp_try_approach_directions
            ),
            "recovery_public_grasp_approach_direction": (
                args.recovery_public_grasp_approach_direction
            ),
            "recovery_public_grasp_approach_directions": (
                args.recovery_public_grasp_approach_directions
            ),
            "recovery_public_grasp_pose_offset_world": (
                args.recovery_public_grasp_pose_offset_world
            ),
            "recovery_public_grasp_pose_offset_along_approach": (
                args.recovery_public_grasp_pose_offset_along_approach
            ),
            "recovery_validate_public_grasp_after_action": (
                args.recovery_validate_public_grasp_after_action
            ),
            "recovery_public_grasp_validation_min_object_lift": (
                args.recovery_public_grasp_validation_min_object_lift
            ),
            "recovery_public_grasp_validation_max_object_lift": (
                args.recovery_public_grasp_validation_max_object_lift
            ),
            "recovery_public_grasp_validation_max_object_xy_displacement": (
                args.recovery_public_grasp_validation_max_object_xy_displacement
            ),
            "recovery_public_grasp_rank_by_legacy_pose": (
                args.recovery_public_grasp_rank_by_legacy_pose
            ),
            "recovery_public_grasp_use_legacy_orientation": (
                args.recovery_public_grasp_use_legacy_orientation
            ),
            "grasp_max_open_length": args.grasp_max_open_length,
            "grasp_min_open_length": args.grasp_min_open_length,
            "grasp_finger_length": args.grasp_finger_length,
            "grasp_x_thickness": args.grasp_x_thickness,
            "grasp_y_thickness": args.grasp_y_thickness,
            "grasp_root_z_width": args.grasp_root_z_width,
            "grasp_open_check_margin": args.grasp_open_check_margin,
            "grasp_point_sample_dense": args.grasp_point_sample_dense,
            "grasp_antipodal_n_sample": args.grasp_antipodal_n_sample,
            "grasp_collision_query_batch_size": args.grasp_collision_query_batch_size,
            "log_dir": str(artifact_dir),
        }
        if args.compile_only:
            env.unwrapped.generate_graph_for_actions(**common_kwargs)
            task_success = None
        else:
            valid = generate_and_execute_action_list(env, 0, False, **common_kwargs)
            if not valid:
                raise RuntimeError("Rearrangement demo produced no valid actions.")
            task_success = _as_bool(env.unwrapped.is_task_success())
            layout = _rearrangement_layout_summary(env)
            if forced_error_config is not None:
                _validate_forced_recovery_demo(
                    forced_error_config=forced_error_config,
                    task_success=task_success,
                    layout=layout,
                )
        _write_result(
            artifact_dir=artifact_dir,
            program_success=True,
            task_success=task_success,
            forced_error_config=forced_error_config,
            env=env,
        )
    except Exception as exc:
        _write_result(
            artifact_dir=artifact_dir,
            program_success=False,
            task_success=task_success,
            forced_error_config=forced_error_config,
            env=env,
            exception=exc,
        )
        raise
    finally:
        if args.record_video:
            flushed_videos = _flush_recorded_videos(env)
            if flushed_videos:
                logger.log_info(
                    f"Flushed {flushed_videos} recorded video stream(s) to "
                    f"{case_dir / 'outputs' / 'videos'}.",
                    color="green",
                )
        env.close()

    logger.log_info(
        f"Rearrangement atomic graph demo output: {case_dir}", color="green"
    )


if __name__ == "__main__":
    main()
