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

"""Machine-readable and reviewable calibration reports."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator import CandidateEvaluation
from .metrics import QualificationResult
from .schema import CalibrationConfig, resolve_control_schedule
from .tuning import TuningResult


def build_calibration_report(
    config: CalibrationConfig,
    *,
    audits: Iterable[Any] = (),
    tuning: TuningResult | None = None,
    qualification_evaluation: CandidateEvaluation | None = None,
    qualification: QualificationResult | None = None,
) -> dict[str, Any]:
    """Build the complete evidence envelope for a calibration run.

    Args:
        config: Validated calibration inputs and runtime configuration.
        audits: DexSim SimReady reports for the configured assets.
        tuning: Optional candidate-search result.
        qualification_evaluation: Optional held-out candidate evaluation.
        qualification: Optional hard-gate result for the held-out evaluation.

    Returns:
        Strict-JSON-compatible report payload with inputs and provenance.
    """
    audit_payloads = [report.to_dict() for report in audits]
    schedule = resolve_control_schedule(
        config.physics_dt,
        config.control_frequency_hz,
        allow_approximate=config.allow_approximate_control_frequency,
    )
    audit_failed = any(report["status"] == "fail" for report in audit_payloads)
    audit_review = any(report["status"] == "review" for report in audit_payloads)
    if audit_failed or (qualification is not None and qualification.status == "fail"):
        status = "fail"
    elif qualification is not None:
        status = "review" if audit_review else qualification.status
    elif tuning is not None:
        status = "review" if audit_review else "candidate"
    else:
        status = "review" if audit_review else "audited"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "embodichain.dynamics_calibration.report",
        "claim": "effective_drive_tuning",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "assets": config.asset_records(),
            "backend": config.backend,
            "device": config.device,
            "physics_dt": config.physics_dt,
            "control_schedule": schedule.to_dict(),
            "seed": config.seed,
            "candidate_count": config.candidate_count,
            "evaluator_target": config.evaluator.target,
        },
        "versions": {
            "embodichain": _package_version("embodichain"),
            "dexsim_engine": _package_version("dexsim_engine"),
            "dexsim_commit_id": _module_attribute("dexsim", "__commit_id__"),
            "embodichain_git_sha": _git_sha(Path(__file__).resolve()),
        },
        "asset_audits": audit_payloads,
        "uncertainty": {
            "status": "not_estimated",
            "reason": (
                "V1 tunes effective drive behavior from tracking evidence and does "
                "not identify physical parameters or confidence intervals."
            ),
        },
    }
    if tuning is not None:
        payload["tuning"] = tuning.to_dict()
    if qualification_evaluation is not None:
        payload["qualification_evaluation"] = {
            "cache_hit": qualification_evaluation.cache_hit,
            "cache_key": qualification_evaluation.cache_key,
            "metrics": qualification_evaluation.metrics.to_dict(),
            "evaluator_metadata": dict(qualification_evaluation.metadata),
        }
    if qualification is not None:
        payload["qualification"] = qualification.to_dict()
    return payload


def calibration_report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown review of a calibration report.

    Args:
        report: Report produced by :func:`build_calibration_report`.

    Returns:
        Markdown summary suitable for reviewer inspection.
    """
    inputs = report["inputs"]
    schedule = inputs["control_schedule"]
    lines = [
        "# EmbodiChain Dynamics Calibration",
        "",
        f"- Status: **{report['status']}**",
        f"- Claim: `{report['claim']}`",
        f"- Backend/device: `{inputs['backend']}` / `{inputs['device']}`",
        f"- Physics timestep: `{inputs['physics_dt']:.9g}` s",
        (
            "- Control frequency: "
            f"requested `{schedule['requested_hz']:.9g}` Hz, "
            f"actual `{schedule['actual_hz']:.9g}` Hz"
        ),
        f"- Seed: `{inputs['seed']}`",
        "",
        "## Asset audits",
        "",
    ]
    audits = report.get("asset_audits", [])
    if audits:
        for audit in audits:
            lines.append(
                f"- **{audit['status']}** — `{audit['source']}` "
                f"(`{audit['asset_sha256']}`)"
            )
    else:
        lines.append("No asset audit was attached.")

    tuning = report.get("tuning")
    if tuning is not None:
        lines.extend(
            [
                "",
                "## Drive tuning",
                "",
                f"- Baseline objective: `{_format_number(tuning['baseline_objective'])}`",
                f"- Best objective: `{_format_number(tuning['best_objective'])}`",
                f"- Best candidate: `{json.dumps(tuning['best_candidate'], sort_keys=True)}`",
                f"- Evaluated candidates: `{len(tuning['trials'])}`",
            ]
        )

    qualification = report.get("qualification")
    if qualification is not None:
        lines.extend(["", "## Qualification", ""])
        for gate in qualification["gates"]:
            marker = "PASS" if gate["passed"] else "FAIL"
            entity = f" ({gate['entity']})" if gate.get("entity") else ""
            lines.append(
                f"- **{marker}** `{gate['name']}`{entity}: "
                f"observed `{gate['observed']}`, expected `{gate['expected']}`"
            )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report supports effective drive tuning only. It does not claim "
            "mass, center-of-mass, inertia, or friction identification.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_calibration_reports(
    output_dir: str | Path, report: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Write canonical JSON and Markdown reports into one output directory.

    Args:
        output_dir: Directory that receives ``report.json`` and ``report.md``.
        report: Strict-JSON-compatible calibration report.

    Returns:
        Paths to the JSON and Markdown reports, respectively.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(calibration_report_to_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _module_attribute(module_name: str, attribute: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "unavailable"
    value = getattr(module, attribute, "unavailable")
    return str(value)


def _format_number(value: Any) -> str:
    return f"{value:.9g}" if isinstance(value, (int, float)) else str(value)


def _git_sha(source: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


__all__ = [
    "build_calibration_report",
    "calibration_report_to_markdown",
    "write_calibration_reports",
]
