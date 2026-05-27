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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

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
from embodichain.lab.sim.agent.task_state import summarize_pour_water_state
from embodichain.utils import logger
from embodichain.utils.utility import load_json

DEFAULT_GYM_CONFIG = (
    ROOT_DIR / "configs/gym/agent/pour_water_agent/fast_gym_config.json"
)
DEFAULT_ARTIFACT_CACHE_ROOT = ROOT_DIR / "outputs/agent_repro_compare"
DEFAULT_DATABASE_ARTIFACT_ROOT = (
    ROOT_DIR / "embodichain/database/agent_generated_content"
)
DATABASE_ARTIFACT_SOURCE_BY_CASE_SOURCE = {
    "single_normal": "SinglePourWater",
    "single_recovery": "SinglePourWater",
    "dual_normal": "DualPourWater",
    "dual_recovery": "DualPourWater",
}


@dataclass(frozen=True)
class CaseSpec:
    label: str
    task_name: str
    agent_config: Path
    # compile_recovery builds monitor/recovery graph artifacts. runtime_recovery
    # controls whether a triggered monitor may execute its recovery branch.
    # error_no_recovery cases compile monitors but disable branch execution so
    # the same deterministic error can be compared with/without recovery.
    compile_recovery: bool
    runtime_recovery: bool
    error_injection: bool
    blind_error_injection: bool
    expected_success: bool
    output_dir: str
    artifact_source_dir: str


CASE_SPECS: dict[str, CaseSpec] = {
    "single_clean_no_recovery": CaseSpec(
        label="single_clean_no_recovery",
        task_name="PourWaterAgent_SingleNormal",
        agent_config=ROOT_DIR / "configs/gym/agent/pour_water_agent/agent_config.json",
        compile_recovery=False,
        runtime_recovery=False,
        error_injection=False,
        blind_error_injection=False,
        expected_success=True,
        output_dir="single_clean_no_recovery",
        artifact_source_dir="single_normal",
    ),
    "single_error_no_recovery": CaseSpec(
        label="single_error_no_recovery",
        task_name="PourWaterAgent_SingleRecovery",
        agent_config=ROOT_DIR / "configs/gym/agent/pour_water_agent/agent_config.json",
        compile_recovery=True,
        runtime_recovery=False,
        error_injection=True,
        blind_error_injection=False,
        expected_success=False,
        output_dir="single_error_no_recovery",
        artifact_source_dir="single_recovery",
    ),
    "single_error_blind_no_recovery": CaseSpec(
        label="single_error_blind_no_recovery",
        task_name="PourWaterAgent_SingleNormal",
        agent_config=ROOT_DIR / "configs/gym/agent/pour_water_agent/agent_config.json",
        compile_recovery=False,
        runtime_recovery=False,
        error_injection=True,
        blind_error_injection=True,
        expected_success=False,
        output_dir="single_error_blind_no_recovery",
        artifact_source_dir="single_normal",
    ),
    "single_clean_with_recovery": CaseSpec(
        label="single_clean_with_recovery",
        task_name="PourWaterAgent_SingleRecovery",
        agent_config=ROOT_DIR / "configs/gym/agent/pour_water_agent/agent_config.json",
        compile_recovery=True,
        runtime_recovery=True,
        error_injection=False,
        blind_error_injection=False,
        expected_success=True,
        output_dir="single_clean_with_recovery",
        artifact_source_dir="single_recovery",
    ),
    "single_error_with_recovery": CaseSpec(
        label="single_error_with_recovery",
        task_name="PourWaterAgent_SingleRecovery",
        agent_config=ROOT_DIR / "configs/gym/agent/pour_water_agent/agent_config.json",
        compile_recovery=True,
        runtime_recovery=True,
        error_injection=True,
        blind_error_injection=False,
        expected_success=True,
        output_dir="single_error_with_recovery",
        artifact_source_dir="single_recovery",
    ),
    "dual_clean_no_recovery": CaseSpec(
        label="dual_clean_no_recovery",
        task_name="DualPourWater",
        agent_config=ROOT_DIR
        / "configs/gym/agent/pour_water_agent/agent_config_dual.json",
        compile_recovery=False,
        runtime_recovery=False,
        error_injection=False,
        blind_error_injection=False,
        expected_success=True,
        output_dir="dual_clean_no_recovery",
        artifact_source_dir="dual_normal",
    ),
    "dual_error_no_recovery": CaseSpec(
        label="dual_error_no_recovery",
        task_name="DualPourWater_MultiRecovery",
        agent_config=ROOT_DIR
        / "configs/gym/agent/pour_water_agent/agent_config_dual.json",
        compile_recovery=True,
        runtime_recovery=False,
        error_injection=True,
        blind_error_injection=False,
        expected_success=False,
        output_dir="dual_error_no_recovery",
        artifact_source_dir="dual_recovery",
    ),
    "dual_error_blind_no_recovery": CaseSpec(
        label="dual_error_blind_no_recovery",
        task_name="DualPourWater",
        agent_config=ROOT_DIR
        / "configs/gym/agent/pour_water_agent/agent_config_dual.json",
        compile_recovery=False,
        runtime_recovery=False,
        error_injection=True,
        blind_error_injection=True,
        expected_success=False,
        output_dir="dual_error_blind_no_recovery",
        artifact_source_dir="dual_normal",
    ),
    "dual_clean_with_recovery": CaseSpec(
        label="dual_clean_with_recovery",
        task_name="DualPourWater_MultiRecovery",
        agent_config=ROOT_DIR
        / "configs/gym/agent/pour_water_agent/agent_config_dual.json",
        compile_recovery=True,
        runtime_recovery=True,
        error_injection=False,
        blind_error_injection=False,
        expected_success=True,
        output_dir="dual_clean_with_recovery",
        artifact_source_dir="dual_recovery",
    ),
    "dual_error_with_recovery": CaseSpec(
        label="dual_error_with_recovery",
        task_name="DualPourWater_MultiRecovery",
        agent_config=ROOT_DIR
        / "configs/gym/agent/pour_water_agent/agent_config_dual.json",
        compile_recovery=True,
        runtime_recovery=True,
        error_injection=True,
        blind_error_injection=False,
        expected_success=True,
        output_dir="dual_error_with_recovery",
        artifact_source_dir="dual_recovery",
    ),
}

CASE_ALIASES: dict[str, str] = {
    "single_normal": "single_clean_no_recovery",
    "single_recovery": "single_clean_with_recovery",
    "dual_normal": "dual_clean_no_recovery",
    "dual_recovery": "dual_clean_with_recovery",
}


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _OfflineLLM:
    def invoke(self, *args, **kwargs):
        raise RuntimeError(
            "This compare runner expected cached agent artifacts. "
            "Use --regenerate or --runtime_llm_recovery if LLM calls are needed."
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
    """Map CLI flags to per-agent regeneration controls."""

    task_regenerate = bool(args.regenerate)
    recovery_regenerate = bool(args.runtime_llm_recovery)
    return {
        "task_regenerate": task_regenerate,
        "recovery_regenerate": recovery_regenerate,
        "compile_regenerate": task_regenerate or recovery_regenerate,
    }


def _parse_xyz(value: str) -> list[float]:
    parts = [
        part.strip() for part in value.replace("[", "").replace("]", "").split(",")
    ]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated numbers, for example 0.03,0.0,0.0."
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated numbers, for example 0.03,0.0,0.0."
        ) from exc


def _parse_xyz_list(value: str) -> list[list[float]]:
    directions = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        directions.append(_parse_xyz(item))
    if not directions:
        raise argparse.ArgumentTypeError(
            "Expected at least one direction, for example 0,0,-1;1,0,0."
        )
    return directions


def _parse_csv_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected at least one object name.")
    return names


def _timestamped_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return ROOT_DIR / "outputs" / f"{timestamp}_pour_water_recovery_compare"


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
    """Run each case in a fresh process to avoid renderer resource accumulation."""
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.tsv"
    base_args = _argv_without(sys.argv[1:], {"--case", "--output_root"})

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(
            "case\texit_code\tprogram_success\tsemantic_success\t"
            "physical_upright_success\tassisted_correction_used\t"
            "debug_only\texpectation_matched\tcase_result\n"
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
            summary_file.write(
                f"{case_name}\t{result.returncode}\t"
                f"{_summary_value(case_result, 'program_success')}\t"
                f"{_summary_value(case_result, 'semantic_success')}\t"
                f"{_summary_value(case_result, 'physical_upright_success')}\t"
                f"{_summary_value(case_result, 'assisted_correction_used')}\t"
                f"{_summary_value(case_result, 'debug_only')}\t"
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


def _load_case_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _summary_value(case_result: dict | None, key: str) -> str:
    if case_result is None:
        return ""
    value = case_result.get(key)
    if value is None:
        return ""
    return str(value)


def _write_matrix_report(output_root: Path, summary_path: Path) -> None:
    report_path = output_root / "report.md"
    lines = [
        "# Pour Water Recovery Compare Report",
        "",
        f"- Output root: `{output_root}`",
        f"- Summary: `{summary_path}`",
        "",
        "| Case | Exit | Program | Semantic | Physical Upright | Assisted | Debug Only | Expectation | Notes |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                if len(parts) >= 9:
                    (
                        case_name,
                        exit_code,
                        program,
                        semantic,
                        physical_upright,
                        assisted,
                        debug_only,
                        matched,
                        result_path,
                    ) = parts
                else:
                    (
                        case_name,
                        exit_code,
                        program,
                        semantic,
                        physical_upright,
                        assisted,
                        matched,
                        result_path,
                    ) = parts
                    debug_only = str(assisted == "True")
                notes = ""
                case_result = (
                    _load_case_result(Path(result_path)) if result_path else None
                )
                if case_result:
                    reasons = case_result.get("failure_reasons", [])
                    if reasons:
                        notes = "; ".join(reasons[:3])
                    if case_result.get("debug_only"):
                        notes = (
                            "debug-only assisted correction; not physical success"
                            if not notes
                            else notes + "; debug-only assisted correction"
                        )
                lines.append(
                    f"| `{case_name}` | {exit_code} | {program} | {semantic} | "
                    f"{physical_upright} | {assisted} | {debug_only} | "
                    f"{matched} | {notes} |"
                )
    lines.extend(
        [
            "",
            "Notes:",
            "- `program_success` only means the executor completed without an exception.",
            "- `semantic_success` is a simulation-state check for upright objects, "
            "height sanity, and no final closed-gripper hold.",
            "- `physical_upright_success` reports whether `upright_object` itself "
            "physically stood the object up before any assisted correction.",
            "- `assisted_correction_used=True` means the simulator corrected an "
            "object pose after physical uprighting failed; use it only for "
            "recovery-policy debugging.",
            "- Any assisted-correction case is marked `debug_only=True` and must "
            "not be counted as physical task success, even if later task-state "
            "checks pass.",
            "- A case can have exit code 0 while semantic validation fails; those "
            "cases need video/frame review before being counted as task success.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disable_video_events(env_cfg) -> None:
    env_cfg.filter_dataset_saving = True
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for event_name in ("record_camera", "validation_cameras"):
        if hasattr(events, event_name):
            setattr(events, event_name, None)


def _build_forced_error_config(
    case: CaseSpec,
    args: argparse.Namespace,
) -> dict | None:
    if not (case.error_injection or args.forced_error_injection):
        return None
    return {
        "enabled": True,
        "edge_index": args.error_injection_edge_index,
        "step_index": args.error_injection_step,
        "relative_error_xyz": args.error_injection_offset,
        "error_type": args.error_injection_type,
        "blind": case.blind_error_injection,
        "blind_obj_name": args.blind_error_object,
    }


def _assert_forced_error_triggered(
    case: CaseSpec,
    forced_error_config: dict | None,
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
            f"Case '{case.label}' injected a deterministic error but no monitor triggered."
        )

    logger.log_info(
        "Deterministic error triggered monitor: "
        f"{forced_error_config.get('_triggered_monitor_name')} "
        f"at step {forced_error_config.get('_triggered_step')}.",
        color="green",
    )


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pour-water task x recovery x deterministic-error validation "
            "matrix and keep videos/artifacts per case."
        )
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--case",
        choices=[*CASE_SPECS.keys(), *CASE_ALIASES.keys(), "all"],
        default="single_clean_no_recovery",
        help="Which case to run.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root directory for compare outputs. Defaults to "
            "outputs/YYYYMMDD_HHMM_pour_water_recovery_compare."
        ),
    )
    parser.add_argument(
        "--open_window",
        action="store_true",
        default=False,
        help="Compatibility alias; keeps the simulator window enabled.",
    )
    parser.add_argument(
        "--no_record_video",
        action="store_true",
        default=False,
        help=(
            "Disable record/validation camera events and dataset video saving for "
            "low-memory smoke runs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for the first reset of each case.",
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
        "--interactive_error_injection",
        action="store_true",
        default=False,
        help="Enable terminal-triggered error injection during recovery cases.",
    )
    parser.add_argument(
        "--runtime_llm_recovery",
        action="store_true",
        default=False,
        help=(
            "Regenerate recovery JSON and enable runtime recovery planning after a "
            "monitor triggers. This asks RecoveryAgent for a one-edge recovery "
            "binding instead of only using the precompiled recovery graph."
        ),
    )
    parser.add_argument(
        "--prefer_runtime_llm_recovery",
        action="store_true",
        default=False,
        help="Prefer runtime recovery plans over precompiled recovery branches.",
    )
    parser.add_argument(
        "--runtime_recovery_heuristic",
        action="store_true",
        default=False,
        help=(
            "Use the deterministic runtime recovery fallback instead of making "
            "an LLM call. Useful for debugging the runtime planner path."
        ),
    )
    parser.add_argument(
        "--runtime_recovery_max_total_attempts",
        type=int,
        default=8,
        help="Maximum total runtime recovery plans allowed in one graph execution.",
    )
    parser.add_argument(
        "--runtime_recovery_max_monitor_attempts",
        type=int,
        default=4,
        help="Maximum runtime recovery plans allowed for one edge monitor trigger.",
    )
    parser.add_argument(
        "--runtime_recovery_max_exception_attempts",
        type=int,
        default=2,
        help="Maximum runtime recovery plans allowed after one edge exception.",
    )
    parser.add_argument(
        "--forced_error_injection",
        "--forced_recovery",
        dest="forced_error_injection",
        action="store_true",
        default=False,
        help=(
            "Deterministically inject one monitor-compatible error. "
            "--forced_recovery is kept as a deprecated alias."
        ),
    )
    parser.add_argument(
        "--error_injection_edge_index",
        "--forced_recovery_edge_index",
        dest="error_injection_edge_index",
        type=int,
        default=1,
        help="1-based monitored edge index where deterministic error injection is armed.",
    )
    parser.add_argument(
        "--error_injection_step",
        "--forced_recovery_step",
        dest="error_injection_step",
        type=int,
        default=-1,
        help="Step index within the armed edge; negative means the final step.",
    )
    parser.add_argument(
        "--error_injection_offset",
        "--forced_recovery_offset",
        dest="error_injection_offset",
        type=_parse_xyz,
        default=[0.12, 0.0, 0.0],
        help=(
            "Object offset used for deterministic error injection. The default "
            "is a horizontal translation that avoids intentionally toppling "
            "the bottle."
        ),
    )
    parser.add_argument(
        "--error_injection_type",
        choices=["misplaced_object", "fallen_object"],
        default="misplaced_object",
        help=(
            "Deterministic injected error type. The default keeps object "
            "orientation unchanged and only translates it; use fallen_object "
            "for explicit fallen-object recovery stress tests."
        ),
    )
    parser.add_argument(
        "--blind_error_object",
        type=str,
        default="bottle",
        help="Object moved in *_error_blind_no_recovery cases.",
    )
    parser.add_argument(
        "--demo_allow_bottle_upright_drop_retries",
        type=int,
        default=2,
        help=(
            "Demo-only tolerance for *_error_with_recovery cases. When positive, "
            "recovery grasp physical validation is deferred so upright bottle "
            "slips can be handled by the graph monitors/recovery branches. A "
            "toppled bottle is still reported as out of scope."
        ),
    )
    parser.add_argument(
        "--demo_bottle_toppled_upright_threshold",
        type=float,
        default=0.65,
        help=(
            "Bottle local-z/world-z alignment below this threshold is classified "
            "as toppled and outside the current demo recovery scope."
        ),
    )
    parser.add_argument(
        "--expected_failure_hold_seconds",
        type=float,
        default=8.0,
        help=(
            "Extra hold time recorded after an expected no-recovery failure, "
            "so the failure-state video is long enough to inspect."
        ),
    )
    parser.add_argument(
        "--min_video_seconds",
        type=float,
        default=30.0,
        help="Minimum duration for the actively recorded episode video.",
    )
    parser.add_argument(
        "--min_expected_failure_video_seconds",
        type=float,
        default=30.0,
        help="Minimum duration for expected-failure videos.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=600,
        help=(
            "Override gym max_episode_steps so longer validation videos are not "
            "split into multiple 15s episodes."
        ),
    )
    parser.add_argument(
        "--use_public_grasp_action",
        action="store_true",
        default=False,
        help=(
            "Route Agent grasp through public PickUpActionCfg using the existing "
            "Agent grasp_pose_obj target pose."
        ),
    )
    parser.add_argument(
        "--use_public_grasp_semantics",
        action="store_true",
        default=False,
        help="Use ObjectSemantics + AntipodalAffordance for Agent grasp.",
    )
    parser.add_argument(
        "--public_grasp_strategy",
        choices=[
            "top_down",
            "bottle_lateral",
            "lateral_down",
            "legacy_guided",
            "auto_try_all",
            "auto_general",
        ],
        default=None,
        help="Named semantic grasp strategy passed to the Agent adapter.",
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
        default=None,
        help=(
            "Recovery-only semantic grasp strategy for compiled atomic pick_up "
            "actions. Does not switch the nominal task grasp path."
        ),
    )
    parser.add_argument(
        "--require_public_grasp_action",
        action="store_true",
        default=False,
        help=(
            "Strict public-grasp validation mode. If the public grasp path cannot "
            "run, fail instead of falling back to legacy grasp_pose_obj."
        ),
    )
    parser.add_argument(
        "--allow_public_grasp_annotation",
        action="store_true",
        default=False,
        help="Allow viser annotation if semantic grasp cache is missing.",
    )
    parser.add_argument(
        "--force_public_grasp_reannotate",
        action="store_true",
        default=False,
        help="Force viser re-annotation for semantic grasp.",
    )
    parser.add_argument(
        "--public_grasp_candidate_num",
        type=int,
        default=8,
        help="Number of semantic grasp candidates to retry.",
    )
    parser.add_argument(
        "--recovery_public_grasp_candidate_num",
        type=int,
        default=None,
        help="Recovery-only semantic grasp candidate count override.",
    )
    parser.add_argument(
        "--public_grasp_pre_grasp_distance",
        type=float,
        default=None,
        help="Optional public PickUpActionCfg pre_grasp_distance override.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pre_grasp_distance",
        type=float,
        default=None,
        help="Recovery-only public PickUpActionCfg pre_grasp_distance override.",
    )
    parser.add_argument(
        "--generate_public_grasp_candidates",
        action="store_true",
        default=False,
        help="Generate whole-mesh antipodal candidates without opening viser.",
    )
    parser.add_argument(
        "--public_grasp_auto_approach_direction",
        action="store_true",
        default=False,
        help="Use the horizontal arm-base-to-object direction for semantic grasp.",
    )
    parser.add_argument(
        "--recovery_public_grasp_auto_approach_direction",
        action="store_true",
        default=None,
        help="Recovery-only auto approach direction override for semantic grasp.",
    )
    parser.add_argument(
        "--public_grasp_try_approach_directions",
        action="store_true",
        default=False,
        help=(
            "Try multiple semantic grasp approach directions before falling back "
            "or failing in strict mode, including arm-relative and object-local sides."
        ),
    )
    parser.add_argument(
        "--recovery_public_grasp_try_approach_directions",
        action="store_true",
        default=None,
        help="Recovery-only multi-direction semantic grasp override.",
    )
    parser.add_argument(
        "--public_grasp_approach_direction",
        type=_parse_xyz,
        default=None,
        help="Explicit semantic grasp approach direction, e.g. 0,0,-1.",
    )
    parser.add_argument(
        "--recovery_public_grasp_approach_direction",
        type=_parse_xyz,
        default=None,
        help="Recovery-only explicit semantic grasp approach direction.",
    )
    parser.add_argument(
        "--public_grasp_approach_directions",
        type=_parse_xyz_list,
        default=None,
        help=(
            "Semicolon-separated semantic grasp approach directions, "
            "e.g. 0,0,-1;1,0,0. Overrides auto/multi-direction defaults."
        ),
    )
    parser.add_argument(
        "--recovery_public_grasp_approach_directions",
        type=_parse_xyz_list,
        default=None,
        help="Recovery-only semantic grasp approach direction list.",
    )
    parser.add_argument(
        "--public_grasp_lift_height",
        type=float,
        default=None,
        help="Optional lift_height passed directly to PickUpActionCfg.",
    )
    parser.add_argument(
        "--recovery_public_grasp_lift_height",
        type=float,
        default=None,
        help="Recovery-only lift_height passed to compiled atomic pick_up actions.",
    )
    parser.add_argument(
        "--public_grasp_pose_offset_world",
        type=_parse_xyz,
        default=[0.0, 0.0, 0.0],
        help="Optional world-frame xyz offset applied to public semantic grasp poses.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pose_offset_world",
        type=_parse_xyz,
        default=None,
        help="Recovery-only world-frame xyz offset for semantic grasp poses.",
    )
    parser.add_argument(
        "--public_grasp_pose_offset_along_approach",
        type=float,
        default=0.0,
        help="Optional offset along public semantic grasp approach direction.",
    )
    parser.add_argument(
        "--recovery_public_grasp_pose_offset_along_approach",
        type=float,
        default=None,
        help="Recovery-only offset along semantic grasp approach direction.",
    )
    parser.add_argument(
        "--validate_public_grasp_after_action",
        action="store_true",
        default=False,
        help=(
            "Validate object lift immediately after a public PickUpActionCfg grasp. "
            "Use with --public_grasp_lift_height for same-action validation."
        ),
    )
    parser.add_argument(
        "--recovery_validate_public_grasp_after_action",
        action="store_true",
        default=None,
        help="Enable recovery-only physical validation after atomic pick_up actions.",
    )
    parser.add_argument(
        "--public_grasp_validation_min_object_lift",
        type=float,
        default=0.05,
        help="Minimum object z displacement for adapter-level public grasp validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_min_object_lift",
        type=float,
        default=None,
        help="Recovery-only minimum object z displacement for public grasp validation.",
    )
    parser.add_argument(
        "--public_grasp_validation_max_object_lift",
        type=float,
        default=None,
        help="Optional maximum object z displacement for adapter-level validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_max_object_lift",
        type=float,
        default=None,
        help="Recovery-only maximum object z displacement for public grasp validation.",
    )
    parser.add_argument(
        "--public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=None,
        help="Optional maximum object xy displacement for adapter-level validation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_validation_max_object_xy_displacement",
        type=float,
        default=None,
        help="Recovery-only maximum object xy displacement for public grasp validation.",
    )
    parser.add_argument(
        "--public_grasp_rank_by_legacy_pose",
        action="store_true",
        default=False,
        help="Rank planned semantic grasp candidates by legacy grasp_pose_obj similarity.",
    )
    parser.add_argument(
        "--recovery_public_grasp_rank_by_legacy_pose",
        action="store_true",
        default=None,
        help="Recovery-only legacy-pose ranking override for semantic grasp candidates.",
    )
    parser.add_argument(
        "--public_grasp_use_legacy_orientation",
        action="store_true",
        default=False,
        help="Use semantic grasp centers with legacy grasp_pose_obj orientation.",
    )
    parser.add_argument(
        "--recovery_public_grasp_use_legacy_orientation",
        action="store_true",
        default=None,
        help="Recovery-only legacy-orientation override for semantic grasp candidates.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_position_weight",
        type=float,
        default=1.0,
        help="Position weight for legacy-pose semantic grasp candidate scoring.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_rotation_weight",
        type=float,
        default=0.05,
        help="Rotation weight for legacy-pose semantic grasp candidate scoring.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_max_position_error",
        type=float,
        default=None,
        help="Optional maximum candidate position error against legacy grasp pose.",
    )
    parser.add_argument(
        "--public_grasp_legacy_pose_max_rotation_error",
        type=float,
        default=None,
        help="Optional maximum candidate rotation error against legacy grasp pose.",
    )
    parser.add_argument(
        "--public_grasp_validate_relative_to_legacy_pose",
        action="store_true",
        default=False,
        help="Validate object-in-EEF pose after grasp against legacy grasp reference.",
    )
    parser.add_argument(
        "--public_grasp_max_legacy_relative_pos_error",
        type=float,
        default=0.08,
        help="Maximum post-grasp object-in-EEF position error against legacy reference.",
    )
    parser.add_argument(
        "--public_grasp_max_legacy_relative_rot_error",
        type=float,
        default=0.7,
        help="Maximum post-grasp object-in-EEF rotation error against legacy reference.",
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
        "--grasp_max_deviation_angle", type=float, default=float(torch.pi / 6)
    )
    parser.add_argument(
        "--force_valid_agent_ik",
        action="store_true",
        default=False,
        help="Let Agent pose skills fall back to the nearest valid IK pose.",
    )
    parser.add_argument(
        "--allow_move_relative_orientation_fallback",
        action="store_true",
        default=False,
        help="Try alternate EEF orientations if move_relative_to_object IK fails.",
    )
    parser.add_argument(
        "--disable_public_place_action",
        action="store_true",
        default=False,
        help="Disable public PlaceActionCfg for Agent place_on_table.",
    )
    parser.add_argument(
        "--disable_public_place_upright",
        action="store_true",
        default=False,
        help=(
            "Disable object-upright target pose construction for public "
            "place_on_table."
        ),
    )
    parser.add_argument(
        "--public_place_upright_eps",
        type=float,
        default=0.0,
        help="Extra object-center height offset for public upright placement.",
    )
    parser.add_argument(
        "--public_place_post_open_wait_steps",
        type=int,
        default=20,
        help="Hold waypoints inserted after gripper opening before retreating.",
    )
    parser.add_argument(
        "--public_place_validation_max_xy_error",
        type=float,
        default=0.16,
        help="Maximum final xy error accepted by public place validation.",
    )
    parser.add_argument(
        "--require_public_non_grasp_actions",
        action="store_true",
        default=False,
        help=(
            "Fail if non-grasp Agent skills cannot use public atomic actions "
            "instead of falling back to legacy planning."
        ),
    )
    parser.add_argument(
        "--require_atomic_action_graph",
        action="store_true",
        default=False,
        help=(
            "Fail if a compiled atomic_action/atomic_sequence cannot execute "
            "through AtomicActionEngine."
        ),
    )
    parser.add_argument(
        "--disable_task_state_validation",
        action="store_true",
        default=False,
        help="Do not write semantic task-state validation results.",
    )
    parser.add_argument(
        "--strict_task_state_validation",
        action="store_true",
        default=False,
        help=(
            "Fail a case when program completion and semantic task-state "
            "validation do not match the case expectation."
        ),
    )
    parser.add_argument(
        "--task_state_objects",
        type=_parse_csv_names,
        default=["bottle", "cup"],
        help="Comma-separated object names included in final task-state validation.",
    )
    parser.add_argument(
        "--task_state_upright_threshold",
        type=float,
        default=0.65,
        help="Minimum absolute local-z/world-z alignment for an object to be upright.",
    )
    parser.add_argument(
        "--task_state_max_height_drop",
        type=float,
        default=0.04,
        help="Maximum final object z drop below its initial height.",
    )
    parser.add_argument(
        "--task_state_max_final_hold_distance",
        type=float,
        default=0.08,
        help="Arm-object distance below which a closed gripper counts as still holding.",
    )
    parser.add_argument(
        "--task_state_closed_gripper_threshold",
        type=float,
        default=0.025,
        help="Gripper opening below which final near-object state counts as held.",
    )
    parser.add_argument(
        "--disable_upright_object_validation",
        action="store_true",
        default=False,
        help="Disable post-action validation for the upright_object recovery skill.",
    )
    parser.add_argument(
        "--disable_upright_object_strict_physical",
        action="store_true",
        default=False,
        help=(
            "Do not fail immediately when upright_object physical validation "
            "fails. The final task-state validation still runs."
        ),
    )
    parser.add_argument(
        "--upright_object_assisted_correction",
        action="store_true",
        default=False,
        help=(
            "Allow upright_object to use sim-side pose correction after a failed "
            "physical upright attempt. This is for recovery-policy debugging, "
            "not pure physical validation."
        ),
    )
    parser.add_argument(
        "--upright_object_validation_max_xy_error",
        type=float,
        default=0.16,
        help="Maximum xy error accepted by upright_object post-action validation.",
    )
    parser.add_argument(
        "--upright_object_validation_max_height_error",
        type=float,
        default=0.08,
        help="Maximum z error accepted by upright_object post-action validation.",
    )
    parser.add_argument(
        "--upright_object_physical_primitive",
        choices=[
            "auto",
            "top_down_grasp_place",
            "grasp_place",
            "lever_sweep",
            "top_clamp",
            "grasp_release",
        ],
        default="auto",
        help="Physical strategy used by the upright_object recovery skill.",
    )
    parser.add_argument(
        "--continue_on_case_failure",
        action="store_true",
        default=False,
        help="For --case all, continue running later cases after a subprocess fails.",
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


def _copy_cached_artifacts(
    case: CaseSpec, artifact_dir: Path, _regenerate: bool
) -> None:
    database_source = DATABASE_ARTIFACT_SOURCE_BY_CASE_SOURCE.get(
        case.artifact_source_dir
    )
    if database_source is not None:
        source_dir = DEFAULT_DATABASE_ARTIFACT_ROOT / database_source
        if source_dir.exists():
            shutil.copytree(source_dir, artifact_dir, dirs_exist_ok=True)
            logger.log_info(
                f"Copied database agent artifacts from {source_dir} "
                f"to {artifact_dir}.",
                color="cyan",
            )
            return

    source_dir = DEFAULT_ARTIFACT_CACHE_ROOT / case.artifact_source_dir / "artifacts"
    if not source_dir.exists():
        return

    shutil.copytree(source_dir, artifact_dir, dirs_exist_ok=True)
    logger.log_info(
        f"Copied cached agent artifacts from {source_dir} to {artifact_dir}.",
        color="cyan",
    )


def _forced_error_was_observed(forced_error_config: dict | None) -> bool:
    if not forced_error_config or not forced_error_config.get("_injected", False):
        return False
    if forced_error_config.get("blind", False):
        return True
    return bool(forced_error_config.get("_triggered", False))


def _flush_recorded_videos(env) -> int:
    """Flush validation camera frames without resetting the episode state."""
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
            frame_count = len(getattr(functor, "_frames", []))
            if frame_count == 0:
                continue
            functor.save_and_clear()
            flushed += 1
    return flushed


def _recorded_video_frame_count(env) -> int:
    raw_env = getattr(env, "unwrapped", env)
    event_manager = getattr(raw_env, "event_manager", None)
    if event_manager is None:
        return 0

    frame_count = 0
    for mode_cfgs in getattr(event_manager, "_mode_functor_cfgs", {}).values():
        for functor_cfg in mode_cfgs:
            functor = getattr(functor_cfg, "func", None)
            frames = getattr(functor, "_frames", None)
            if frames is not None:
                frame_count = max(frame_count, len(frames))
    return frame_count


def _hold_expected_failure_state(env, seconds: float, fps: int = 20) -> int:
    """Keep stepping the current robot command so failure videos show the aftermath."""
    if seconds <= 0:
        return 0

    raw_env = getattr(env, "unwrapped", env)
    robot = getattr(raw_env, "robot", None)
    if robot is None:
        return 0

    hold_action = robot.get_qpos().to(dtype=torch.float32)
    steps = max(1, int(round(seconds * fps)))
    for _ in range(steps):
        env.step(hold_action)
    return steps


def _hold_current_video_until(env, min_seconds: float, fps: int = 20) -> int:
    current_frames = _recorded_video_frame_count(env)
    hold_seconds = max(0.0, min_seconds - current_frames / fps)
    return _hold_expected_failure_state(env, hold_seconds, fps=fps)


def _use_public_grasp_from_args(args: argparse.Namespace) -> bool:
    return bool(
        args.use_public_grasp_action
        or _use_public_grasp_semantics_from_args(args)
        or args.require_public_grasp_action
    )


def _use_public_grasp_semantics_from_args(args: argparse.Namespace) -> bool:
    return bool(
        args.use_public_grasp_semantics
        or args.public_grasp_strategy is not None
        or args.public_grasp_rank_by_legacy_pose
        or args.public_grasp_use_legacy_orientation
        or args.public_grasp_legacy_pose_max_position_error is not None
        or args.public_grasp_legacy_pose_max_rotation_error is not None
    )


def _demo_bottle_drop_tolerance_enabled(
    case: CaseSpec,
    args: argparse.Namespace,
) -> bool:
    return bool(
        case.error_injection
        and case.runtime_recovery
        and int(args.demo_allow_bottle_upright_drop_retries) > 0
    )


def _effective_recovery_public_grasp_validation(
    case: CaseSpec,
    args: argparse.Namespace,
) -> bool | None:
    if args.recovery_validate_public_grasp_after_action is not None:
        return args.recovery_validate_public_grasp_after_action
    if _demo_bottle_drop_tolerance_enabled(case, args):
        return False
    return None


class _CaseTaskValidationError(RuntimeError):
    pass


def _write_case_result(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    env,
    artifact_dir: Path,
    program_success: bool,
    exception: Exception | None = None,
) -> dict:
    raw_env = getattr(env, "unwrapped", env)
    upright_records = list(getattr(raw_env, "_upright_object_validation_records", []))
    assisted_correction_used = bool(
        getattr(raw_env, "_upright_object_assisted_correction_used", False)
    )
    physical_upright_success = (
        None
        if not upright_records
        else all(record.get("physical_success") is True for record in upright_records)
    )
    debug_only = assisted_correction_used
    physical_task_success = (
        bool(program_success)
        and not debug_only
        and physical_upright_success is not False
    )
    if env is None:
        result = {
            "case": case.label,
            "program_success": program_success,
            "semantic_success": None,
            "expected_success": case.expected_success,
            "expectation_matched": False,
            "failure_reasons": [
                "environment initialization failed before task-state validation"
            ],
            "physical_upright_success": physical_upright_success,
            "assisted_correction_used": assisted_correction_used,
            "debug_only": debug_only,
            "physical_task_success": False,
            "upright_object_validations": upright_records,
            "exception": _exception_summary(exception),
        }
    elif args.disable_task_state_validation:
        result = {
            "case": case.label,
            "program_success": program_success,
            "semantic_success": None,
            "expectation_matched": None,
            "failure_reasons": [],
            "physical_upright_success": physical_upright_success,
            "assisted_correction_used": assisted_correction_used,
            "debug_only": debug_only,
            "physical_task_success": None,
            "upright_object_validations": upright_records,
            "exception": _exception_summary(exception),
        }
    else:
        summary = summarize_pour_water_state(
            env,
            object_names=args.task_state_objects,
            upright_threshold=args.task_state_upright_threshold,
            max_height_drop=args.task_state_max_height_drop,
            max_final_hold_distance=args.task_state_max_final_hold_distance,
            closed_gripper_threshold=args.task_state_closed_gripper_threshold,
        )
        semantic_success = bool(summary.semantic_success)
        expectation_matched = _case_expectation_matched(
            case=case,
            program_success=program_success,
            semantic_success=semantic_success,
            debug_only=debug_only,
            physical_upright_success=physical_upright_success,
        )
        physical_task_success = physical_task_success and semantic_success
        result = {
            "case": case.label,
            "program_success": program_success,
            "semantic_success": semantic_success,
            "expected_success": case.expected_success,
            "expectation_matched": expectation_matched,
            "failure_reasons": list(summary.failure_reasons),
            "task_state": summary.to_dict(),
            "physical_upright_success": physical_upright_success,
            "assisted_correction_used": assisted_correction_used,
            "debug_only": debug_only,
            "physical_task_success": physical_task_success,
            "upright_object_validations": upright_records,
            "exception": _exception_summary(exception),
        }

    demo_scope = _demo_bottle_recovery_scope(case=case, args=args, result=result)
    if demo_scope is not None:
        result["demo_recovery_scope"] = demo_scope

    result_path = artifact_dir / "case_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.log_info(
        f"Case '{case.label}' validation: "
        f"program_success={result.get('program_success')} "
        f"semantic_success={result.get('semantic_success')} "
        f"expectation_matched={result.get('expectation_matched')} "
        f"result={result_path}",
        color="green" if result.get("expectation_matched") is not False else "yellow",
    )
    for reason in result.get("failure_reasons", [])[:5]:
        logger.log_warning(f"Case '{case.label}' semantic failure reason: {reason}")
    demo_scope = result.get("demo_recovery_scope")
    if demo_scope and demo_scope.get("classification") == "out_of_scope_toppled_bottle":
        logger.log_warning(
            f"Case '{case.label}' ended with a toppled bottle; this is outside "
            "the current upright-drop recovery demo scope."
        )
    return result


def _demo_bottle_recovery_scope(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    result: dict,
) -> dict | None:
    if not _demo_bottle_drop_tolerance_enabled(case, args):
        return None

    bottle_state = result.get("task_state", {}).get("object_states", {}).get("bottle")
    vertical_alignment = None
    height_drop = None
    is_toppled = None
    if isinstance(bottle_state, dict):
        vertical_alignment = bottle_state.get("vertical_alignment")
        height_drop = bottle_state.get("height_drop")
        if vertical_alignment is not None:
            is_toppled = float(vertical_alignment) < float(
                args.demo_bottle_toppled_upright_threshold
            )

    exception_message = (result.get("exception") or {}).get("message", "")
    grasp_validation_failed = (
        "Public grasp physical validation failed" in exception_message
        or "object_lift" in exception_message
    )

    if is_toppled is True:
        classification = "out_of_scope_toppled_bottle"
    elif result.get("semantic_success") is True:
        classification = "within_scope_completed"
    elif grasp_validation_failed:
        classification = "upright_drop_retry_candidate"
    else:
        classification = "within_scope_incomplete"

    return {
        "enabled": True,
        "classification": classification,
        "allowed_upright_drop_retries": int(
            args.demo_allow_bottle_upright_drop_retries
        ),
        "recovery_grasp_validation_deferred": (
            args.recovery_validate_public_grasp_after_action is None
        ),
        "toppled_upright_threshold": float(args.demo_bottle_toppled_upright_threshold),
        "bottle_vertical_alignment": vertical_alignment,
        "bottle_height_drop": height_drop,
        "bottle_toppled": is_toppled,
        "note": (
            "This demo tolerates upright bottle slips by deferring recovery "
            "grasp validation to graph monitors. Fallen/toppled bottle "
            "uprighting remains outside the current demo scope."
        ),
    }


def _case_expectation_matched(
    *,
    case: CaseSpec,
    program_success: bool,
    semantic_success: bool,
    debug_only: bool = False,
    physical_upright_success: bool | None = None,
) -> bool:
    physical_task_success = (
        program_success
        and semantic_success
        and not debug_only
        and physical_upright_success is not False
    )
    if case.expected_success:
        return physical_task_success
    return not physical_task_success


def _exception_summary(exception: Exception | None) -> dict | None:
    if exception is None:
        return None
    return {
        "type": type(exception).__name__,
        "message": str(exception),
    }


def _raise_if_strict_validation_failed(case: CaseSpec, result: dict) -> None:
    if result.get("expectation_matched") is False:
        raise _CaseTaskValidationError(
            f"Case '{case.label}' failed task-state validation: "
            f"program_success={result.get('program_success')}, "
            f"semantic_success={result.get('semantic_success')}, "
            f"expected_success={case.expected_success}, "
            f"reasons={result.get('failure_reasons', [])}"
        )


def run_case(case: CaseSpec, args: argparse.Namespace, output_root: Path) -> None:
    _configure_llms_for_args(args)
    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    if args.no_record_video:
        _disable_video_events(env_cfg)
    if args.max_episode_steps is not None:
        env_cfg.max_episode_steps = int(args.max_episode_steps)
        gym_config["max_episode_steps"] = int(args.max_episode_steps)
    agent_config = load_json(str(case.agent_config))

    case_dir = output_root / case.output_dir
    artifact_dir = case_dir / "artifacts"
    case_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _copy_cached_artifacts(case, artifact_dir, args.regenerate)

    try:
        env = gymnasium.make(
            id=gym_config["id"],
            cfg=env_cfg,
            agent_config=agent_config,
            agent_config_path=str(case.agent_config),
            task_name=case.task_name,
            **action_config,
        )
    except Exception as exc:
        _write_case_result(
            case=case,
            args=args,
            env=None,
            artifact_dir=artifact_dir,
            program_success=False,
            exception=exc,
        )
        logger.log_warning(
            f"Case '{case.label}' failed during environment initialization: "
            f"{type(exc).__name__}: {exc}. Result: {artifact_dir / 'case_result.json'}"
        )
        raise

    logger.log_info(
        f"Running case '{case.label}' into {case_dir} "
        f"(compile_recovery={case.compile_recovery}, "
        f"runtime_recovery={case.runtime_recovery}, "
        f"runtime_llm_recovery={args.runtime_llm_recovery}, "
        f"error_injection={case.error_injection or args.forced_error_injection}, "
        f"no_record_video={args.no_record_video}, "
        f"use_public_grasp={_use_public_grasp_from_args(args)}, "
        f"use_public_grasp_semantics={_use_public_grasp_semantics_from_args(args)}, "
        f"public_grasp_strategy={args.public_grasp_strategy}, "
        f"recovery_public_grasp_strategy={args.recovery_public_grasp_strategy}, "
        f"use_public_place={not args.disable_public_place_action}, "
        f"strict_public_non_grasp={args.require_public_non_grasp_actions}, "
        f"generate_public_grasp_candidates={args.generate_public_grasp_candidates}, "
        f"auto_public_grasp_approach={args.public_grasp_auto_approach_direction}, "
        f"try_public_grasp_approaches={args.public_grasp_try_approach_directions}, "
        f"force_valid_agent_ik={args.force_valid_agent_ik}, "
        f"move_orientation_fallback={args.allow_move_relative_orientation_fallback}, "
        f"public_place_upright={not args.disable_public_place_upright}, "
        f"strict_public_grasp={args.require_public_grasp_action}, "
        f"demo_bottle_drop_tolerance="
        f"{_demo_bottle_drop_tolerance_enabled(case, args)})",
        color="cyan",
    )
    forced_error_config = _build_forced_error_config(case, args)
    disable_recovery_branches = case.compile_recovery and not case.runtime_recovery
    try:
        with pushd(case_dir):
            env.reset(seed=args.seed)
            try:
                valid = generate_and_execute_action_list(
                    env,
                    0,
                    False,
                    regenerate=args.regenerate,
                    **_agent_regenerate_kwargs(args),
                    recovery=case.compile_recovery,
                    interactive_error_injection=args.interactive_error_injection,
                    runtime_llm_recovery=args.runtime_llm_recovery,
                    prefer_runtime_llm_recovery=args.prefer_runtime_llm_recovery,
                    runtime_recovery_use_llm=not args.runtime_recovery_heuristic,
                    runtime_recovery_max_total_attempts=(
                        args.runtime_recovery_max_total_attempts
                    ),
                    runtime_recovery_max_monitor_attempts=(
                        args.runtime_recovery_max_monitor_attempts
                    ),
                    runtime_recovery_max_exception_attempts=(
                        args.runtime_recovery_max_exception_attempts
                    ),
                    forced_recovery_injection=forced_error_config,
                    disable_recovery_branches=disable_recovery_branches,
                    use_public_grasp_action=_use_public_grasp_from_args(args),
                    use_public_grasp_semantics=(
                        _use_public_grasp_semantics_from_args(args)
                    ),
                    require_public_grasp_action=args.require_public_grasp_action,
                    allow_public_grasp_annotation=args.allow_public_grasp_annotation,
                    force_public_grasp_reannotate=args.force_public_grasp_reannotate,
                    public_grasp_strategy=args.public_grasp_strategy,
                    recovery_public_grasp_strategy=(
                        args.recovery_public_grasp_strategy
                    ),
                    public_grasp_candidate_num=args.public_grasp_candidate_num,
                    recovery_public_grasp_candidate_num=(
                        args.recovery_public_grasp_candidate_num
                    ),
                    public_grasp_pre_grasp_distance=(
                        args.public_grasp_pre_grasp_distance
                    ),
                    recovery_public_grasp_pre_grasp_distance=(
                        args.recovery_public_grasp_pre_grasp_distance
                    ),
                    generate_public_grasp_candidates=(
                        args.generate_public_grasp_candidates
                    ),
                    public_grasp_auto_approach_direction=(
                        args.public_grasp_auto_approach_direction
                    ),
                    recovery_public_grasp_auto_approach_direction=(
                        args.recovery_public_grasp_auto_approach_direction
                    ),
                    public_grasp_try_approach_directions=(
                        args.public_grasp_try_approach_directions
                    ),
                    recovery_public_grasp_try_approach_directions=(
                        args.recovery_public_grasp_try_approach_directions
                    ),
                    public_grasp_approach_direction=args.public_grasp_approach_direction,
                    recovery_public_grasp_approach_direction=(
                        args.recovery_public_grasp_approach_direction
                    ),
                    public_grasp_approach_directions=(
                        args.public_grasp_approach_directions
                    ),
                    recovery_public_grasp_approach_directions=(
                        args.recovery_public_grasp_approach_directions
                    ),
                    public_grasp_lift_height=args.public_grasp_lift_height,
                    recovery_public_grasp_lift_height=(
                        args.recovery_public_grasp_lift_height
                    ),
                    public_grasp_pose_offset_world=(
                        args.public_grasp_pose_offset_world
                    ),
                    recovery_public_grasp_pose_offset_world=(
                        args.recovery_public_grasp_pose_offset_world
                    ),
                    public_grasp_pose_offset_along_approach=(
                        args.public_grasp_pose_offset_along_approach
                    ),
                    recovery_public_grasp_pose_offset_along_approach=(
                        args.recovery_public_grasp_pose_offset_along_approach
                    ),
                    validate_public_grasp_after_action=(
                        args.validate_public_grasp_after_action
                    ),
                    recovery_validate_public_grasp_after_action=(
                        _effective_recovery_public_grasp_validation(case, args)
                    ),
                    public_grasp_validation_min_object_lift=(
                        args.public_grasp_validation_min_object_lift
                    ),
                    recovery_public_grasp_validation_min_object_lift=(
                        args.recovery_public_grasp_validation_min_object_lift
                    ),
                    public_grasp_validation_max_object_lift=(
                        args.public_grasp_validation_max_object_lift
                    ),
                    recovery_public_grasp_validation_max_object_lift=(
                        args.recovery_public_grasp_validation_max_object_lift
                    ),
                    public_grasp_validation_max_object_xy_displacement=(
                        args.public_grasp_validation_max_object_xy_displacement
                    ),
                    recovery_public_grasp_validation_max_object_xy_displacement=(
                        args.recovery_public_grasp_validation_max_object_xy_displacement
                    ),
                    public_grasp_rank_by_legacy_pose=(
                        args.public_grasp_rank_by_legacy_pose
                    ),
                    recovery_public_grasp_rank_by_legacy_pose=(
                        args.recovery_public_grasp_rank_by_legacy_pose
                    ),
                    public_grasp_use_legacy_orientation=(
                        args.public_grasp_use_legacy_orientation
                    ),
                    recovery_public_grasp_use_legacy_orientation=(
                        args.recovery_public_grasp_use_legacy_orientation
                    ),
                    public_grasp_legacy_pose_position_weight=(
                        args.public_grasp_legacy_pose_position_weight
                    ),
                    public_grasp_legacy_pose_rotation_weight=(
                        args.public_grasp_legacy_pose_rotation_weight
                    ),
                    public_grasp_legacy_pose_max_position_error=(
                        args.public_grasp_legacy_pose_max_position_error
                    ),
                    public_grasp_legacy_pose_max_rotation_error=(
                        args.public_grasp_legacy_pose_max_rotation_error
                    ),
                    public_grasp_validate_relative_to_legacy_pose=(
                        args.public_grasp_validate_relative_to_legacy_pose
                    ),
                    public_grasp_max_legacy_relative_pos_error=(
                        args.public_grasp_max_legacy_relative_pos_error
                    ),
                    public_grasp_max_legacy_relative_rot_error=(
                        args.public_grasp_max_legacy_relative_rot_error
                    ),
                    grasp_max_open_length=args.grasp_max_open_length,
                    grasp_min_open_length=args.grasp_min_open_length,
                    grasp_finger_length=args.grasp_finger_length,
                    grasp_x_thickness=args.grasp_x_thickness,
                    grasp_y_thickness=args.grasp_y_thickness,
                    grasp_root_z_width=args.grasp_root_z_width,
                    grasp_open_check_margin=args.grasp_open_check_margin,
                    grasp_point_sample_dense=args.grasp_point_sample_dense,
                    grasp_antipodal_n_sample=args.grasp_antipodal_n_sample,
                    grasp_collision_query_batch_size=(
                        args.grasp_collision_query_batch_size
                    ),
                    grasp_max_deviation_angle=args.grasp_max_deviation_angle,
                    force_valid=args.force_valid_agent_ik,
                    allow_move_relative_orientation_fallback=(
                        args.allow_move_relative_orientation_fallback
                    ),
                    use_public_place_action=not args.disable_public_place_action,
                    public_place_upright=not args.disable_public_place_upright,
                    public_place_upright_eps=args.public_place_upright_eps,
                    public_place_post_open_wait_steps=(
                        args.public_place_post_open_wait_steps
                    ),
                    public_place_validation_max_xy_error=(
                        args.public_place_validation_max_xy_error
                    ),
                    validate_place_preconditions=True,
                    validate_public_place_after_action=True,
                    require_public_non_grasp_actions=(
                        args.require_public_non_grasp_actions
                    ),
                    require_atomic_action_graph=args.require_atomic_action_graph,
                    validate_upright_object_after_action=(
                        not args.disable_upright_object_validation
                    ),
                    upright_object_strict_physical=(
                        not args.disable_upright_object_strict_physical
                    ),
                    upright_object_assisted_correction=(
                        args.upright_object_assisted_correction
                    ),
                    upright_object_validation_min_upright_dot=(
                        args.task_state_upright_threshold
                    ),
                    upright_object_validation_max_xy_error=(
                        args.upright_object_validation_max_xy_error
                    ),
                    upright_object_validation_max_height_error=(
                        args.upright_object_validation_max_height_error
                    ),
                    upright_object_physical_primitive=(
                        args.upright_object_physical_primitive
                    ),
                    log_dir=str(artifact_dir),
                )
                if not valid:
                    raise RuntimeError(
                        f"Case '{case.label}' produced no valid actions."
                    )
                _assert_forced_error_triggered(case, forced_error_config)
                result = _write_case_result(
                    case=case,
                    args=args,
                    env=env,
                    artifact_dir=artifact_dir,
                    program_success=True,
                )
                if args.strict_task_state_validation:
                    _raise_if_strict_validation_failed(case, result)
                if not case.expected_success and result.get("semantic_success") is True:
                    logger.log_warning(
                        f"Case '{case.label}' completed semantically even though "
                        "it was intended to be a no-recovery failure. Treat this "
                        "as a non-disruptive error-injection probe, not as a "
                        "successful recovery demonstration."
                    )
            except Exception as exc:
                if isinstance(exc, _CaseTaskValidationError):
                    raise
                _write_case_result(
                    case=case,
                    args=args,
                    env=env,
                    artifact_dir=artifact_dir,
                    program_success=False,
                    exception=exc,
                )
                if case.expected_success:
                    if args.no_record_video:
                        held_steps = 0
                        flushed_videos = 0
                        video_note = "Video recording disabled"
                    else:
                        held_steps = _hold_current_video_until(
                            env, args.min_video_seconds
                        )
                        flushed_videos = _flush_recorded_videos(env)
                        video_note = (
                            f"Video directory: {case_dir / 'outputs' / 'videos'}"
                        )
                    logger.log_warning(
                        f"Case '{case.label}' failed unexpectedly: "
                        f"{type(exc).__name__}: {exc}. "
                        f"{video_note} "
                        f"(held_steps={held_steps}, "
                        f"flushed_recorders={flushed_videos})"
                    )
                    raise
                if not _forced_error_was_observed(forced_error_config):
                    raise
                if args.no_record_video:
                    held_steps = 0
                    flushed_videos = 0
                    video_note = "Video recording disabled"
                else:
                    min_video_seconds = max(
                        args.min_video_seconds,
                        args.min_expected_failure_video_seconds,
                    )
                    current_frames = _recorded_video_frame_count(env)
                    min_hold_seconds = max(
                        0.0, min_video_seconds - current_frames / 20.0
                    )
                    held_steps = _hold_expected_failure_state(
                        env,
                        max(args.expected_failure_hold_seconds, min_hold_seconds),
                    )
                    flushed_videos = _flush_recorded_videos(env)
                    video_note = (
                        f"Expected-failure video directory: "
                        f"{case_dir / 'outputs' / 'videos'}"
                    )
                logger.log_info(
                    f"Case '{case.label}' failed as expected after deterministic "
                    f"error injection: {type(exc).__name__}: {exc}",
                    color="green",
                )
                logger.log_info(
                    f"Case '{case.label}' {video_note} "
                    f"(held_steps={held_steps}, flushed_recorders={flushed_videos})",
                    color="green",
                )
                return

            held_steps = (
                0
                if args.no_record_video
                else _hold_current_video_until(env, args.min_video_seconds)
            )
            if held_steps > 0:
                logger.log_info(
                    f"Case '{case.label}' padded active video with {held_steps} hold steps.",
                    color="green",
                )
            if not args.no_record_video:
                env.reset()
    finally:
        env.close()

    if args.no_record_video:
        logger.log_info(
            f"Case '{case.label}' finished. Video recording disabled. "
            f"Artifacts: {artifact_dir}",
            color="green",
        )
    else:
        video_path = case_dir / "outputs" / "videos" / "episode_0_cam1.mp4"
        logger.log_info(
            f"Case '{case.label}' finished. Video: {video_path}",
            color="green",
        )


def main() -> None:
    args = build_parser()
    if args.open_window:
        args.headless = False
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _timestamped_output_root()
    )

    if args.case == "all":
        _run_all_cases_in_subprocesses(output_root, args)
        return
    else:
        case_names = [CASE_ALIASES.get(args.case, args.case)]

    for case_name in case_names:
        run_case(CASE_SPECS[case_name], args, output_root)


if __name__ == "__main__":
    main()
