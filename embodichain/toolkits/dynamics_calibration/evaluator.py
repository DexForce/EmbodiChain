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

"""Isolated candidate evaluation with content-addressed caching."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import TrackingMetrics, compute_tracking_metrics
from .schema import EvaluatorConfig


class EvaluationError(RuntimeError):
    """Raised when an isolated evaluator does not produce valid evidence."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """Metrics and cache provenance for one candidate.

    Attributes:
        metrics: Centralized metrics computed from evaluator observations.
        cache_hit: Whether the result came from an existing cache entry.
        cache_key: Content-addressed identity for the evaluation inputs.
        metadata: Strict-JSON metadata supplied by the application evaluator.
    """

    metrics: TrackingMetrics
    cache_hit: bool
    cache_key: str
    metadata: dict[str, Any]


def run_candidate(
    evaluator: EvaluatorConfig,
    overlay: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    cache_dir: str | Path,
) -> CandidateEvaluation:
    """Run one evaluator in a fresh process, or restore its cached result.

    Args:
        evaluator: Isolated evaluator target, timeout, and application payload.
        overlay: Candidate drive overlay passed to the evaluator.
        context: Reproducible backend, timing, asset, seed, and phase context.
        cache_dir: Directory containing content-addressed result entries.

    Returns:
        Metrics, evaluator metadata, and cache provenance for the candidate.

    Raises:
        EvaluationError: If the worker times out, exits unsuccessfully, or
            returns invalid observations or metadata.
        TypeError: If inputs cannot be encoded as strict JSON.
        ValueError: If inputs contain non-finite JSON numbers.
    """
    canonical_input = {
        "schema_version": 1,
        "evaluator": {
            "target": evaluator.target,
            "fingerprint": _evaluator_fingerprint(evaluator.target),
            "timeout_seconds": evaluator.timeout_seconds,
            "payload": evaluator.payload,
        },
        "runtime": _runtime_fingerprint(),
        "overlay": dict(overlay),
        "context": dict(context),
    }
    encoded = json.dumps(
        canonical_input,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = resolved_cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return CandidateEvaluation(
                TrackingMetrics.from_dict(cached["metrics"]),
                True,
                cache_key,
                dict(cached.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A partial or incompatible cache entry is recomputed and replaced.
            pass

    with tempfile.TemporaryDirectory(prefix="embodichain-calibration-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "input.json"
        output_path = temp_root / "output.json"
        input_path.write_bytes(encoded)
        command = [
            sys.executable,
            "-m",
            "embodichain.toolkits.dynamics_calibration.worker",
            str(input_path),
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=evaluator.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise EvaluationError(
                f"evaluator timed out after {evaluator.timeout_seconds:g} seconds"
            ) from error

        payload: dict[str, Any] = {}
        if output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        if completed.returncode != 0 or payload.get("status") != "ok":
            detail = payload.get("error")
            if not detail:
                detail = completed.stderr.strip() or completed.stdout.strip()
            if not detail:
                detail = f"worker exited with status {completed.returncode}"
            raise EvaluationError(str(detail))
        raw_result = payload.get("result")
        if not isinstance(raw_result, Mapping):
            raise EvaluationError("evaluator result must be a mapping")
        try:
            metrics = compute_tracking_metrics(raw_result)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"invalid evaluator result: {error}") from error
        metadata = raw_result.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise EvaluationError("evaluator metadata must be a mapping")
        metadata = dict(metadata)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise EvaluationError(
                f"evaluator metadata is not strict JSON: {error}"
            ) from error

    cache_payload = {
        "schema_version": 1,
        "metrics": metrics.to_dict(),
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=resolved_cache_dir,
        prefix=f".{cache_key}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(cache_payload, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary_cache = Path(temporary.name)
    os.replace(temporary_cache, cache_path)
    return CandidateEvaluation(metrics, False, cache_key, metadata)


def _evaluator_fingerprint(target: str) -> str:
    module_or_path, _ = target.rsplit(":", maxsplit=1)
    source_path = Path(module_or_path)
    if not source_path.is_file():
        relative = Path(*module_or_path.split("."))
        for entry in sys.path:
            root = Path(entry or ".")
            candidates = (
                root / relative.with_suffix(".py"),
                root / relative / "__init__.py",
            )
            source_path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                source_path,
            )
            if source_path.is_file():
                break
    if source_path.is_file():
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def _runtime_fingerprint() -> dict[str, str]:
    versions = {
        "python": sys.version,
        "calibration_implementation": _calibration_implementation_fingerprint(),
    }
    for package in ("embodichain", "dexsim_engine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _calibration_implementation_fingerprint() -> str:
    """Hash toolkit sources that define worker and metric cache semantics."""
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["CandidateEvaluation", "EvaluationError", "run_candidate"]
