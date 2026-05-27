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
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
DEFAULT_APPLE_ASSET = ROOT_DIR / "obj/asset_simready.obj"
TASK_NAME = "Rearrangement"
DEFAULT_APPLE_OBJECTS = ("fork", "spoon")


@dataclass(frozen=True)
class CaseSpec:
    label: str
    compile_recovery: bool
    runtime_recovery: bool
    forced_error_injection: bool
    expected_success: bool
    output_dir: str


CASE_SPECS: dict[str, CaseSpec] = {
    "apple_clean_no_recovery": CaseSpec(
        label="apple_clean_no_recovery",
        compile_recovery=False,
        runtime_recovery=False,
        forced_error_injection=False,
        expected_success=True,
        output_dir="apple_clean_no_recovery",
    ),
    "apple_error_no_recovery": CaseSpec(
        label="apple_error_no_recovery",
        compile_recovery=True,
        runtime_recovery=False,
        forced_error_injection=True,
        expected_success=False,
        output_dir="apple_error_no_recovery",
    ),
    "apple_clean_with_recovery": CaseSpec(
        label="apple_clean_with_recovery",
        compile_recovery=True,
        runtime_recovery=True,
        forced_error_injection=False,
        expected_success=True,
        output_dir="apple_clean_with_recovery",
    ),
    "apple_error_with_recovery": CaseSpec(
        label="apple_error_with_recovery",
        compile_recovery=True,
        runtime_recovery=True,
        forced_error_injection=True,
        expected_success=True,
        output_dir="apple_error_with_recovery",
    ),
}

CASE_ALIASES: dict[str, str] = {
    "clean": "apple_clean_with_recovery",
    "error": "apple_error_with_recovery",
    "no_recovery": "apple_error_no_recovery",
}


class _OfflineLLM:
    """Fail fast if cached artifacts are missing and a prompt would call an LLM."""

    def invoke(self, *args, **kwargs):
        raise RuntimeError(
            "Offline apple rearrangement demo expected cached agent artifacts. "
            "Use --regenerate or --runtime_llm_recovery if LLM calls are needed."
        )


def _configure_llms_for_args(args: argparse.Namespace) -> None:
    """Allow only the agent phase explicitly requested by the CLI flags."""

    offline_llm = _OfflineLLM()
    if not args.regenerate or agent_llm.task_llm is None:
        agent_llm.task_llm = offline_llm
    if not args.runtime_llm_recovery or agent_llm.recovery_llm is None:
        agent_llm.recovery_llm = offline_llm
    if (
        not (args.regenerate or args.runtime_llm_recovery)
        or agent_llm.compile_llm is None
    ):
        agent_llm.compile_llm = offline_llm
    agent_llm.failure_anticipation_llm = agent_llm.recovery_llm
    agent_llm.code_llm = agent_llm.compile_llm


def _agent_regenerate_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    task_regenerate = bool(args.regenerate)
    recovery_regenerate = bool(args.runtime_llm_recovery)
    return {
        "task_regenerate": task_regenerate,
        "recovery_regenerate": recovery_regenerate,
        "compile_regenerate": task_regenerate or recovery_regenerate,
    }


def _timestamped_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return ROOT_DIR / "outputs" / f"{timestamp}_apple_rearrangement_generalization"


def _parse_xyz(value: str) -> list[float]:
    parts = [
        part.strip() for part in value.replace("[", "").replace("]", "").split(",")
    ]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated numbers, for example 0.08,0.0,0.0."
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated numbers, for example 0.08,0.0,0.0."
        ) from exc


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


def _parse_csv_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected at least one object uid.")
    return names


def _argv_without(args: list[str], names: set[str]) -> list[str]:
    filtered = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in names:
            skip_next = True
            continue
        if any(arg.startswith(f"{name}=") for name in names):
            continue
        filtered.append(arg)
    return filtered


def _run_all_cases_in_subprocesses(
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.tsv"
    base_args = _argv_without(sys.argv[1:], {"--case", "--output_root"})

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(
            "case\texit_code\tprogram_success\ttask_success\tlayout_success\t"
            "forced_injected\tforced_triggered\texpectation_matched\tcase_result\n"
        )
        for case_name in CASE_SPECS:
            log_path = log_dir / f"{case_name}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case_name,
                "--output_root",
                str(output_root),
                *base_args,
            ]
            logger.log_info(
                f"Running case '{case_name}' in a subprocess. Log: {log_path}",
                color="cyan",
            )
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            case_result_path = (
                output_root
                / CASE_SPECS[case_name].output_dir
                / "artifacts"
                / "case_result.json"
            )
            case_result = _load_case_result(case_result_path)
            forced_error = (case_result or {}).get("forced_error") or {}
            layout = (case_result or {}).get("layout") or {}
            summary_file.write(
                f"{case_name}\t{result.returncode}\t"
                f"{_summary_value(case_result, 'program_success')}\t"
                f"{_summary_value(case_result, 'task_success')}\t"
                f"{layout.get('success_xy', '')}\t"
                f"{forced_error.get('injected', '')}\t"
                f"{forced_error.get('triggered', '')}\t"
                f"{_summary_value(case_result, 'expectation_matched')}\t"
                f"{case_result_path if case_result_path.exists() else ''}\n"
            )
            summary_file.flush()
            if result.returncode != 0:
                _write_matrix_report(output_root, summary_path)
                if args.continue_on_case_failure:
                    continue
                raise RuntimeError(
                    f"Case '{case_name}' failed with status {result.returncode}. "
                    f"See {log_path}."
                )

    _write_matrix_report(output_root, summary_path)
    logger.log_info(f"All cases completed. Summary: {summary_path}", color="green")


def _load_case_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_value(case_result: dict[str, Any] | None, key: str) -> str:
    if case_result is None:
        return ""
    value = case_result.get(key)
    if value is None:
        return ""
    return str(value)


def _write_matrix_report(output_root: Path, summary_path: Path) -> None:
    report_path = output_root / "report.md"
    lines = [
        "# Apple Rearrangement Generalization Report",
        "",
        f"- Output root: `{output_root}`",
        f"- Summary: `{summary_path}`",
        "",
        "| Case | Exit | Program | Task | Layout | Forced Injected | Forced Triggered | Expectation | Notes |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                (
                    case_name,
                    exit_code,
                    program,
                    task_success,
                    layout_success,
                    forced_injected,
                    forced_triggered,
                    matched,
                    result_path,
                ) = parts[:9]
                notes = ""
                case_result = (
                    _load_case_result(Path(result_path)) if result_path else None
                )
                if case_result:
                    reasons = case_result.get("failure_reasons", [])
                    if reasons:
                        notes = "; ".join(str(reason) for reason in reasons[:3])
                    exception = case_result.get("exception")
                    if exception:
                        exception_text = (
                            f"{exception.get('type')}: {exception.get('message')}"
                        )
                        notes = (
                            exception_text
                            if not notes
                            else notes + "; " + exception_text
                        )
                lines.append(
                    f"| `{case_name}` | {exit_code} | {program} | {task_success} | "
                    f"{layout_success} | {forced_injected} | {forced_triggered} | "
                    f"{matched} | {notes} |"
                )
    lines.extend(
        [
            "",
            "Notes:",
            "- The logical object UIDs remain `fork` and `spoon` so cached "
            "Rearrangement agent artifacts can be reused without changing the "
            "environment success checker.",
            "- The mesh for those logical objects is replaced with "
            "`obj/asset_simready.obj` at runtime.",
            "- `layout_success` checks final xy placement around the plate, not only "
            "whether the executor completed.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_database_artifacts(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATABASE_ARTIFACT_ROOT.exists():
        raise FileNotFoundError(
            f"Rearrangement database artifacts not found: "
            f"{DEFAULT_DATABASE_ARTIFACT_ROOT}"
        )
    shutil.copytree(DEFAULT_DATABASE_ARTIFACT_ROOT, artifact_dir, dirs_exist_ok=True)
    logger.log_info(
        f"Copied Rearrangement agent artifacts from "
        f"{DEFAULT_DATABASE_ARTIFACT_ROOT} to {artifact_dir}.",
        color="cyan",
    )


def _resolve_apple_asset(args: argparse.Namespace) -> Path:
    asset_path = Path(args.apple_asset).expanduser()
    if not asset_path.is_absolute():
        asset_path = (ROOT_DIR / asset_path).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Apple mesh not found: {asset_path}")
    return asset_path


def _write_apple_gym_config(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    artifact_dir: Path,
) -> Path:
    gym_config = load_json(args.gym_config)
    apple_asset = _resolve_apple_asset(args)
    apple_objects = set(args.apple_objects)
    replaced_objects: list[str] = []

    for obj_cfg in gym_config.get("rigid_object", []):
        uid = obj_cfg.get("uid")
        if uid not in apple_objects:
            continue
        shape = obj_cfg.setdefault("shape", {})
        shape["shape_type"] = "Mesh"
        shape["fpath"] = str(apple_asset)
        shape["compute_uv"] = bool(args.apple_compute_uv)
        obj_cfg["body_scale"] = [args.apple_scale] * 3
        attrs = obj_cfg.setdefault("attrs", {})
        if args.apple_mass is not None:
            attrs["mass"] = args.apple_mass
        attrs.setdefault("static_friction", 0.95)
        attrs.setdefault("dynamic_friction", 0.9)
        attrs.setdefault("restitution", 0.03)
        replaced_objects.append(str(uid))

    missing_objects = sorted(apple_objects - set(replaced_objects))
    if missing_objects:
        raise ValueError(
            "Apple replacement object uid(s) not found in gym config: "
            + ", ".join(missing_objects)
        )

    if not args.keep_replaced_object_visual_randomization:
        events = gym_config.get("env", {}).get("events", {})
        for event_name in list(events):
            params = events[event_name].get("params", {})
            entity_cfg = params.get("entity_cfg", {})
            if (
                events[event_name].get("func") == "randomize_visual_material"
                and entity_cfg.get("uid") in apple_objects
            ):
                events.pop(event_name)

    dataset = gym_config.get("env", {}).get("dataset", {}).get("lerobot", {})
    instruction = dataset.get("params", {}).get("instruction")
    if isinstance(instruction, dict):
        instruction["lang"] = (
            "Place the two apple mesh objects neatly on opposite sides of the "
            "plate on the table."
        )
    extra = dataset.get("params", {}).get("extra")
    if isinstance(extra, dict):
        extra["task_description"] = "AppleRearrangementGeneralization"

    gym_config["apple_generalization"] = {
        "case": case.label,
        "apple_asset": str(apple_asset),
        "apple_scale": float(args.apple_scale),
        "logical_replaced_object_uids": replaced_objects,
        "note": (
            "Object UIDs remain compatible with the cached Rearrangement graph; "
            "only the runtime mesh and optional physical attributes are changed."
        ),
    }
    patched_path = artifact_dir / "apple_gym_config.json"
    patched_path.write_text(
        json.dumps(gym_config, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.log_info(f"Patched apple gym config written to {patched_path}", color="cyan")
    return patched_path


def _patch_cached_task_graph_for_apple(
    *,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.preserve_cached_grasp_runtime:
        return

    task_graph_path = artifact_dir / "agent_task_graph.json"
    if not task_graph_path.exists():
        return
    task_graph = json.loads(task_graph_path.read_text(encoding="utf-8"))
    changed = 0
    for action in _iter_atomic_actions(task_graph):
        if action.get("name") != "pick_up":
            continue
        target = action.get("target")
        if (
            not isinstance(target, dict)
            or target.get("obj_name") not in args.apple_objects
        ):
            continue
        runtime_kwargs = dict(action.get("runtime_kwargs") or {})
        runtime_kwargs.update(
            {
                "public_grasp_strategy": args.nominal_public_grasp_strategy,
                "public_grasp_candidate_num": args.public_grasp_candidate_num,
                "public_grasp_pre_grasp_distance": (
                    args.public_grasp_pre_grasp_distance
                ),
                "public_grasp_lift_height": args.public_grasp_lift_height,
                "public_grasp_filter_ground_collision": (
                    args.public_grasp_filter_ground_collision
                ),
                "public_grasp_rank_by_legacy_pose": False,
                "public_grasp_use_legacy_orientation": False,
                "validate_public_grasp_after_action": (
                    args.validate_public_grasp_after_action
                ),
                "public_grasp_validation_min_object_lift": (
                    args.public_grasp_validation_min_object_lift
                ),
                "public_grasp_validation_max_object_xy_displacement": (
                    args.public_grasp_validation_max_object_xy_displacement
                ),
            }
        )
        action["runtime_kwargs"] = {
            key: value for key, value in runtime_kwargs.items() if value is not None
        }
        changed += 1

    if changed == 0:
        logger.log_warning(
            f"No pick_up actions for {args.apple_objects} were patched in "
            f"{task_graph_path}."
        )
        return
    task_graph_path.write_text(
        json.dumps(task_graph, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.log_info(
        f"Patched {changed} nominal pick_up action(s) in {task_graph_path} "
        f"for apple mesh grasp generalization.",
        color="cyan",
    )


def _iter_atomic_actions(value: Any):
    if isinstance(value, dict):
        if value.get("kind") == "atomic_action":
            yield value
        for child in value.values():
            yield from _iter_atomic_actions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_atomic_actions(child)


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


def _build_forced_error_config(
    case: CaseSpec,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not (case.forced_error_injection or args.forced_error_injection):
        return None
    return {
        "enabled": True,
        "edge_index": args.error_injection_edge_index,
        "step_index": args.error_injection_step,
        "relative_error_xyz": args.error_injection_offset,
        "error_type": args.error_injection_type,
        "blind": bool(args.blind_error_injection),
        "blind_obj_name": args.error_injection_object or args.apple_objects[0],
    }


def _assert_forced_error_triggered(
    *,
    case: CaseSpec,
    forced_error_config: dict[str, Any] | None,
) -> None:
    if forced_error_config is None:
        return
    if not forced_error_config.get("_injected", False):
        raise RuntimeError(
            f"Case '{case.label}' did not inject a deterministic error. "
            "Try lowering --error_injection_edge_index or --error_injection_step."
        )
    if forced_error_config.get("blind", False):
        logger.log_info(
            "Deterministic blind error injected: "
            f"{forced_error_config.get('_injected_monitor')} "
            f"at step {forced_error_config.get('_injected_at_step')}.",
            color="green",
        )
        return
    if not forced_error_config.get("_triggered", False):
        raise RuntimeError(
            f"Case '{case.label}' injected a deterministic error but no monitor "
            "triggered."
        )
    logger.log_info(
        "Deterministic error triggered monitor: "
        f"{forced_error_config.get('_triggered_monitor_name')} "
        f"at step {forced_error_config.get('_triggered_step')}.",
        color="green",
    )


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
        "recovery_enabled": data.get("metadata", {}).get("recovery_enabled"),
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
    case: CaseSpec,
    args: argparse.Namespace,
    artifact_dir: Path,
    program_success: bool,
    task_success: bool | None,
    forced_error_config: dict[str, Any] | None = None,
    env=None,
    exception: Exception | None = None,
) -> dict[str, Any]:
    graph_path = artifact_dir / "agent_compiled_graph.json"
    layout = _rearrangement_layout_summary(env, object_uids=args.apple_objects)
    semantic_success = (
        bool(program_success)
        and task_success is True
        and layout is not None
        and layout.get("success_xy") is True
    )
    expectation_matched = (
        semantic_success if case.expected_success else not semantic_success
    )
    failure_reasons: list[str] = []
    if program_success is not True:
        failure_reasons.append("executor did not complete")
    if task_success is not True:
        failure_reasons.append("environment task success check did not pass")
    if layout is None:
        failure_reasons.append("layout summary unavailable")
    elif layout.get("success_xy") is not True:
        failure_reasons.append("final apple-object xy layout is outside tolerance")

    result = {
        "case": case.label,
        "task": "AppleRearrangementGeneralization",
        "program_success": program_success,
        "task_success": task_success,
        "semantic_success": semantic_success,
        "expected_success": case.expected_success,
        "expectation_matched": expectation_matched,
        "failure_reasons": failure_reasons,
        "apple_asset": str(_resolve_apple_asset(args)),
        "apple_scale": float(args.apple_scale),
        "apple_object_uids": list(args.apple_objects),
        "nominal_public_grasp_strategy": (
            None
            if args.preserve_cached_grasp_runtime
            else args.nominal_public_grasp_strategy
        ),
        "apple_grasp_profile": {
            "public_grasp_candidate_num": args.public_grasp_candidate_num,
            "public_grasp_pre_grasp_distance": (args.public_grasp_pre_grasp_distance),
            "public_grasp_lift_height": args.public_grasp_lift_height,
            "public_grasp_filter_ground_collision": (
                args.public_grasp_filter_ground_collision
            ),
            "grasp_x_thickness": args.grasp_x_thickness,
            "grasp_y_thickness": args.grasp_y_thickness,
            "grasp_root_z_width": args.grasp_root_z_width,
            "grasp_open_check_margin": args.grasp_open_check_margin,
        },
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
    logger.log_info(
        f"Apple rearrangement result written to {result_path}. "
        f"expectation_matched={expectation_matched}",
        color="green" if expectation_matched else "yellow",
    )
    return result


def _forced_error_summary(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "enabled": bool(config.get("enabled", False)),
        "edge_index": config.get("edge_index"),
        "blind": bool(config.get("blind", False)),
        "blind_obj_name": config.get("blind_obj_name"),
        "error_type": config.get("error_type"),
        "relative_error_xyz": config.get("relative_error_xyz"),
        "injected": bool(config.get("_injected", False)),
        "triggered": bool(config.get("_triggered", False)),
        "injected_at_step": config.get("_injected_at_step"),
        "triggered_step": config.get("_triggered_step"),
        "injected_monitor": config.get("_injected_monitor"),
        "triggered_monitor": config.get("_triggered_monitor_name"),
    }


def _rearrangement_layout_summary(
    env,
    *,
    object_uids: list[str],
) -> dict[str, Any] | None:
    if env is None:
        return None
    raw_env = getattr(env, "unwrapped", env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return None
    if len(object_uids) < 2:
        return {
            "available": False,
            "error": "layout validation needs at least two apple object uids",
        }

    left_uid = object_uids[0]
    right_uid = object_uids[1]
    try:
        plate_xy = _object_xy(raw_env, "plate")
        left_xy = _object_xy(raw_env, left_uid)
        right_xy = _object_xy(raw_env, right_uid)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    tolerance = float(
        getattr(raw_env, "metadata", {})
        .get("success_params", {})
        .get("tolerance", 0.02)
    )
    expected = {
        left_uid: [plate_xy[0], plate_xy[1] + 0.16],
        right_uid: [plate_xy[0], plate_xy[1] - 0.16],
    }
    errors = {
        left_uid: _xy_error(left_xy, expected[left_uid]),
        right_uid: _xy_error(right_xy, expected[right_uid]),
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
            left_uid: left_xy,
            right_uid: right_xy,
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
            "Run a RearrangementAgent generalization demo by replacing the cached "
            "fork/spoon meshes with obj/asset_simready.obj apple meshes."
        )
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--case",
        choices=[*CASE_SPECS.keys(), *CASE_ALIASES.keys(), "all"],
        default="apple_clean_with_recovery",
        help="Which apple generalization case to run.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "outputs/YYYYMMDD_HHMM_apple_rearrangement_generalization."
        ),
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default=str(DEFAULT_AGENT_CONFIG),
        help="Path to the Rearrangement agent config JSON.",
    )
    parser.add_argument(
        "--apple_asset",
        type=str,
        default=str(DEFAULT_APPLE_ASSET),
        help="Path to the apple OBJ mesh used to replace logical fork/spoon objects.",
    )
    parser.add_argument(
        "--apple_objects",
        type=_parse_csv_names,
        default=list(DEFAULT_APPLE_OBJECTS),
        help=(
            "Comma-separated logical object UIDs whose meshes are replaced by the "
            "apple asset. Defaults to fork,spoon for cached graph compatibility."
        ),
    )
    parser.add_argument(
        "--apple_scale",
        type=float,
        default=0.5,
        help="Uniform body scale for the replacement apple mesh. Defaults to 0.5.",
    )
    parser.add_argument("--apple_mass", type=float, default=0.05)
    parser.add_argument(
        "--apple_compute_uv",
        action="store_true",
        default=False,
        help="Recompute UVs for the apple mesh.",
    )
    parser.add_argument(
        "--keep_replaced_object_visual_randomization",
        action="store_true",
        default=False,
        help="Keep old fork/spoon visual randomization events after mesh replacement.",
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
        help=(
            "Regenerate the nominal task graph JSON. Cached recovery JSON is reused "
            "unless --runtime_llm_recovery is also set."
        ),
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
        "--continue_on_case_failure",
        action="store_true",
        default=False,
        help="When --case all is used, continue after a subprocess exits nonzero.",
    )
    parser.add_argument(
        "--runtime_llm_recovery",
        action="store_true",
        default=False,
        help=(
            "Regenerate recovery JSON and allow runtime recovery to call the "
            "RecoveryAgent instead of using only cached branches."
        ),
    )
    parser.add_argument(
        "--prefer_runtime_llm_recovery",
        action="store_true",
        default=False,
        help="Prefer runtime LLM recovery when a monitor triggers.",
    )
    parser.add_argument(
        "--recovery_max_monitor_attempts",
        type=int,
        default=2,
        help="Maximum static recovery attempts for the same edge/monitor pair.",
    )
    parser.add_argument(
        "--recovery_max_total_attempts",
        type=int,
        default=8,
        help="Maximum total static recovery attempts in one graph execution.",
    )
    parser.add_argument(
        "--require_atomic_action_graph",
        action="store_true",
        default=False,
        help="Fail if an atomic graph action cannot execute through AtomicActionEngine.",
    )
    parser.add_argument(
        "--preserve_cached_grasp_runtime",
        action="store_true",
        default=False,
        help=(
            "Do not patch cached nominal pick_up runtime kwargs. By default the "
            "script switches apple-object nominal grasps to auto_general."
        ),
    )
    parser.add_argument(
        "--nominal_public_grasp_strategy",
        choices=[
            "top_down",
            "bottle_lateral",
            "lateral_down",
            "legacy_guided",
            "auto_try_all",
            "auto_general",
        ],
        default="auto_general",
        help="Public grasp strategy patched into nominal apple-object pick_up actions.",
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
        help="Recovery-only semantic grasp strategy for compiled atomic pick_up actions.",
    )
    parser.add_argument(
        "--disable_public_grasp_candidate_generation",
        action="store_true",
        default=False,
        help="Require an existing public grasp annotation cache instead of generating candidates.",
    )
    parser.add_argument(
        "--public_grasp_candidate_num",
        type=int,
        default=96,
        help="Nominal apple semantic grasp candidates per approach direction.",
    )
    parser.add_argument(
        "--recovery_public_grasp_candidate_num",
        type=int,
        default=96,
        help="Recovery apple semantic grasp candidates per approach direction.",
    )
    parser.add_argument(
        "--public_grasp_pre_grasp_distance",
        type=float,
        default=0.025,
        help="Shorter nominal pre-grasp distance for small apple meshes.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pre_grasp_distance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--public_grasp_lift_height",
        type=float,
        default=0.08,
        help="Lower nominal lift height for small apple meshes.",
    )
    parser.add_argument("--recovery_public_grasp_lift_height", type=float, default=None)
    parser.add_argument(
        "--public_grasp_filter_ground_collision",
        nargs="?",
        const="true",
        default=False,
        type=_parse_optional_bool,
        help=(
            "Enable table/ground collision filtering for apple grasp candidates. "
            "Defaults to false because the apple mesh is small and sits on the table."
        ),
    )
    parser.add_argument(
        "--validate_public_grasp_after_action",
        nargs="?",
        const="true",
        default=True,
        type=_parse_optional_bool,
    )
    parser.add_argument(
        "--recovery_validate_public_grasp_after_action",
        nargs="?",
        const="true",
        default=None,
        type=_parse_optional_bool,
    )
    parser.add_argument(
        "--public_grasp_validation_min_object_lift",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_min_object_lift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=None,
    )
    parser.add_argument("--grasp_max_open_length", type=float, default=0.088)
    parser.add_argument("--grasp_min_open_length", type=float, default=0.003)
    parser.add_argument("--grasp_finger_length", type=float, default=0.078)
    parser.add_argument("--grasp_x_thickness", type=float, default=0.006)
    parser.add_argument("--grasp_y_thickness", type=float, default=0.012)
    parser.add_argument("--grasp_root_z_width", type=float, default=0.025)
    parser.add_argument("--grasp_open_check_margin", type=float, default=0.002)
    parser.add_argument("--grasp_point_sample_dense", type=float, default=0.012)
    parser.add_argument("--grasp_antipodal_n_sample", type=int, default=20000)
    parser.add_argument("--grasp_collision_query_batch_size", type=int, default=512)
    parser.add_argument(
        "--forced_error_injection",
        action="store_true",
        default=False,
        help="Inject a deterministic error in addition to the selected case default.",
    )
    parser.add_argument(
        "--blind_error_injection",
        action="store_true",
        default=False,
        help="Inject the deterministic error by object name instead of monitor matching.",
    )
    parser.add_argument(
        "--error_injection_edge_index",
        type=int,
        default=3,
        help=(
            "1-based monitored edge index where deterministic error injection is armed. "
            "The default targets the plate-offset move edge in the cached graph."
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
        default=[0.10, 0.0, 0.0],
        help="Injected xyz displacement, for example 0.10,0.0,0.0.",
    )
    parser.add_argument(
        "--error_injection_type",
        choices=["misplaced_object", "fallen_object"],
        default="misplaced_object",
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
    if len(args.apple_objects) < 2:
        raise ValueError("--apple_objects must include at least two object UIDs.")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _timestamped_output_root()
    )
    case_name = CASE_ALIASES.get(args.case, args.case)
    if case_name == "all":
        _run_all_cases_in_subprocesses(output_root, args)
        return

    case = CASE_SPECS[case_name]
    _configure_llms_for_args(args)

    case_dir = output_root / case.output_dir
    artifact_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    _copy_database_artifacts(artifact_dir)
    _patch_cached_task_graph_for_apple(artifact_dir=artifact_dir, args=args)
    patched_gym_config = _write_apple_gym_config(
        case=case,
        args=args,
        artifact_dir=artifact_dir,
    )

    original_gym_config = args.gym_config
    args.gym_config = str(patched_gym_config)
    try:
        env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    finally:
        args.gym_config = original_gym_config

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

    forced_error_config = _build_forced_error_config(case, args)
    task_success = None
    try:
        env.reset(seed=args.seed)
        common_kwargs = {
            "regenerate": args.regenerate,
            **_agent_regenerate_kwargs(args),
            "recovery": case.compile_recovery,
            "disable_recovery_branches": not case.runtime_recovery,
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
            "public_grasp_filter_ground_collision": (
                args.public_grasp_filter_ground_collision
            ),
            "recovery_validate_public_grasp_after_action": (
                args.recovery_validate_public_grasp_after_action
            ),
            "recovery_public_grasp_validation_min_object_lift": (
                args.recovery_public_grasp_validation_min_object_lift
            ),
            "recovery_public_grasp_validation_max_object_xy_displacement": (
                args.recovery_public_grasp_validation_max_object_xy_displacement
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
            "grasp_collision_query_batch_size": (args.grasp_collision_query_batch_size),
            "log_dir": str(artifact_dir),
        }
        if args.compile_only:
            env.unwrapped.generate_graph_for_actions(**common_kwargs)
            task_success = None
        else:
            valid = generate_and_execute_action_list(env, 0, False, **common_kwargs)
            if not valid:
                raise RuntimeError(f"Case '{case.label}' produced no valid actions.")
            _assert_forced_error_triggered(
                case=case,
                forced_error_config=forced_error_config,
            )
            task_success = _as_bool(env.unwrapped.is_task_success())
        result = _write_result(
            case=case,
            args=args,
            artifact_dir=artifact_dir,
            program_success=True,
            task_success=task_success,
            forced_error_config=forced_error_config,
            env=env,
        )
        if case.expected_success and result.get("expectation_matched") is False:
            raise RuntimeError(
                f"Case '{case.label}' did not satisfy apple layout validation. "
                f"Result: {artifact_dir / 'case_result.json'}"
            )
    except Exception as exc:
        _write_result(
            case=case,
            args=args,
            artifact_dir=artifact_dir,
            program_success=False,
            task_success=task_success,
            forced_error_config=forced_error_config,
            env=env,
            exception=exc,
        )
        if case.expected_success:
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
        f"Apple rearrangement generalization demo output: {case_dir}", color="green"
    )


if __name__ == "__main__":
    main()
