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

"""Validate and diagnose user-deployed GenSim model services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

__all__: list[str] = []


def _parser(prog: str, *, doctor: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Initialize configured model services and run their health checks."
            if doctor
            else "Validate model-service profiles without contacting providers."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit model_services JSON; otherwise use normal discovery.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv used to resolve profile references.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    if doctor:
        parser.add_argument(
            "--role",
            action="append",
            default=[],
            help="Check one configured role; repeat to select multiple roles.",
        )
    return parser


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": getattr(error, "code", type(error).__name__),
        "detail": str(error),
    }


def _emit(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if report["status"] == "failed" and "roles" not in report:
        print(f"failed: {report['error']}: {report['detail']}")
        return
    for item in report.get("roles", []):
        # Validation reports carry no per-role status; only doctor runs checks.
        status = item.get("status", "configured")
        suffix = f" ({item['error']})" if status == "failed" else ""
        print(
            f"{item['role']}: {status}{suffix} "
            f"[{item['profile']} -> {item['adapter']}]"
        )
    print(f"model services: {report['status']}")


def validate_main(
    argv: Sequence[str] | None = None, *, prog: str | None = None
) -> None:
    """Validate selected profiles without contacting providers.

    Args:
        argv: Command arguments after ``services validate``.
        prog: Optional program name supplied by the unified dispatcher.
    """
    parser = _parser(prog or "embodichain gensim services validate", doctor=False)
    args = parser.parse_args(argv)
    try:
        from .diagnostics import validate_services

        report, _, _ = validate_services(args.config, env_path=args.env_file)
    except Exception as error:
        report = _failure(error)
    _emit(report, as_json=args.json)
    if report["status"] != "ok":
        raise SystemExit(1)


def doctor_main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Initialize roles and call adapter health checks when available.

    Args:
        argv: Command arguments after ``services doctor``.
        prog: Optional program name supplied by the unified dispatcher.
    """
    parser = _parser(prog or "embodichain gensim services doctor", doctor=True)
    args = parser.parse_args(argv)
    try:
        from .diagnostics import doctor_services

        report = doctor_services(
            args.config,
            env_path=args.env_file,
            roles=tuple(args.role),
        )
    except Exception as error:
        report = _failure(error)
    _emit(report, as_json=args.json)
    if report["status"] != "ok":
        raise SystemExit(1)
