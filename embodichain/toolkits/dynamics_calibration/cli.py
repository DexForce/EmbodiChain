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

"""Command-line workflow for effective dynamics calibration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .asset_audit import audit_assets, audits_admit_calibration
from .evaluator import EvaluationError, run_candidate
from .metrics import qualify
from .overlay import load_overlay, write_overlay
from .report import build_calibration_report, write_calibration_reports
from .schema import load_calibration_config
from .tuning import tune_drive


def build_parser() -> argparse.ArgumentParser:
    """Build the dynamics-calibration command parser.

    Returns:
        Parser for the ``audit``, ``tune-drive``, and ``qualify`` commands.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain calibrate-dynamics",
        description=(
            "Audit robot assets, tune effective drive properties, and qualify "
            "them on held-out application trajectories."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="Run DexSim SimReady checks without modifying assets."
    )
    audit_parser.add_argument("assets", nargs="+", type=Path)
    audit_parser.add_argument("--reference-link", action="append", default=[])
    audit_parser.add_argument("--output-dir", type=Path)

    tune_parser = subparsers.add_parser(
        "tune-drive", help="Search effective drive parameters, then qualify the best."
    )
    _add_config_arguments(tune_parser)

    qualify_parser = subparsers.add_parser(
        "qualify", help="Evaluate one existing overlay on held-out conditions."
    )
    _add_config_arguments(qualify_parser)
    qualify_parser.add_argument("--overlay", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a calibration command and preserve nonzero failure semantics.

    Args:
        argv: Command arguments excluding the executable name. Uses process
            arguments when omitted.

    Raises:
        SystemExit: With status 2 when auditing, evaluation, or qualification
            fails.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            _run_audit(args)
        elif args.command == "tune-drive":
            _run_tune(args)
        else:
            _run_qualify(args)
    except (
        EvaluationError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"calibrate-dynamics: error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dynamics_calibration_output")
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reference-link", action="append", default=[])


def _run_audit(args: argparse.Namespace) -> None:
    reports = audit_assets(args.assets, reference_links=args.reference_link)
    payload = {
        "schema_version": 1,
        "kind": "embodichain.dynamics_calibration.asset_audits",
        "status": (
            "pass"
            if all(report.status == "pass" for report in reports)
            else "review" if audits_admit_calibration(reports) else "fail"
        ),
        "reports": [report.to_dict() for report in reports],
    }
    if args.output_dir is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "audit.md").write_text(
            "\n".join(report.to_markdown().rstrip() for report in reports) + "\n",
            encoding="utf-8",
        )
    if not audits_admit_calibration(reports):
        raise SystemExit(2)


def _run_tune(args: argparse.Namespace) -> None:
    config = load_calibration_config(args.config)
    reports = audit_assets(config.assets, reference_links=args.reference_link)
    if not audits_admit_calibration(reports):
        report = build_calibration_report(config, audits=reports)
        write_calibration_reports(args.output_dir, report)
        raise SystemExit(2)
    cache_dir = args.cache_dir or args.output_dir / "cache"
    tuning = tune_drive(config, cache_dir=cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_overlay(args.output_dir / "drive_overlay.yaml", tuning.overlay)
    held_out = run_candidate(
        config.evaluator,
        tuning.overlay,
        config.evaluation_context("qualification"),
        cache_dir=cache_dir,
    )
    qualification = qualify(held_out.metrics, config.qualification)
    report = build_calibration_report(
        config,
        audits=reports,
        tuning=tuning,
        qualification_evaluation=held_out,
        qualification=qualification,
    )
    json_path, markdown_path = write_calibration_reports(args.output_dir, report)
    print(f"Wrote {json_path} and {markdown_path}")
    if qualification.status != "pass":
        raise SystemExit(2)


def _run_qualify(args: argparse.Namespace) -> None:
    config = load_calibration_config(args.config)
    reports = audit_assets(config.assets, reference_links=args.reference_link)
    if not audits_admit_calibration(reports):
        report = build_calibration_report(config, audits=reports)
        write_calibration_reports(args.output_dir, report)
        raise SystemExit(2)
    overlay = load_overlay(args.overlay)
    if overlay.get("assets") != config.asset_records():
        raise ValueError("overlay asset hashes do not match the current configuration")
    cache_dir = args.cache_dir or args.output_dir / "cache"
    held_out = run_candidate(
        config.evaluator,
        overlay,
        config.evaluation_context("qualification"),
        cache_dir=cache_dir,
    )
    qualification = qualify(held_out.metrics, config.qualification)
    report = build_calibration_report(
        config,
        audits=reports,
        qualification_evaluation=held_out,
        qualification=qualification,
    )
    json_path, markdown_path = write_calibration_reports(args.output_dir, report)
    print(f"Wrote {json_path} and {markdown_path}")
    if qualification.status != "pass":
        raise SystemExit(2)


__all__ = ["build_parser", "main"]
