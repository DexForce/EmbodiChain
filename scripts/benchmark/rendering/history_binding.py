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

"""Import a verified untimed history proof without changing original timings.

This is an explicit follow-up mutation of an experiment's evidence directory.
It never changes original run files, ledgers, quality outcomes or reports. The
returned path is consumed by the independent evaluator's --history-evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from .frozen_source import artifact, load_original_run, read_json, sha256, write_json
from .history_contract import validate_history_proof

__all__ = ["import_history_proof", "main"]


def import_history_proof(manifest_path: Path, experiment_root: Path) -> dict[str, Any]:
    """Copy a validated proof closure and bind it to its immutable original run.

    Args:
        manifest_path: Completed acquisition's history.json. Its parent contains
            the entire portable, checksummed evidence closure.
        experiment_root: Original formal experiment receiving extra evidence.

    Returns:
        Import receipt and root-relative history path for offline evaluation.
        Proof validity establishes acquisition provenance, not image quality.
    """
    manifest_path = Path(manifest_path).resolve()
    source_root, target = manifest_path.parent, Path(experiment_root).resolve()
    manifest = read_json(manifest_path)
    original = load_original_run(target, manifest["original"]["run_id"])
    expected = {
        **original["row"],
        "_history_artifact_hashes": original["input_hashes"],
    }
    proof = validate_history_proof(source_root, manifest_path, expected)
    if not proof["valid"]:
        raise ValueError(
            "Cannot import invalid history: " + "; ".join(proof["reasons"])
        )
    digest = sha256(manifest_path)
    destination = target / "history_imports" / digest
    if destination.resolve() != destination:
        raise ValueError("History import destination must not contain symlinks")
    # Validate every existing file before considering an idempotent import. An
    # interrupted copy can be resumed only when the already written bytes match.
    for evidence in proof["artifacts"]:
        path = destination / evidence["path"]
        if path.resolve() != path:
            raise ValueError("History import evidence path must not contain symlinks")
        if path.exists() and (not path.is_file() or sha256(path) != evidence["sha256"]):
            raise ValueError("Existing imported history differs: " + evidence["path"])
    destination.mkdir(parents=True, exist_ok=True)
    for evidence in proof["artifacts"]:
        source_path, path = (
            source_root / evidence["path"],
            destination / evidence["path"],
        )
        if sha256(source_path) != evidence["sha256"]:
            raise ValueError(
                "History source changed during import: " + evidence["path"]
            )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, path)
        if sha256(path) != evidence["sha256"]:
            raise ValueError("Imported history checksum mismatch: " + evidence["path"])
    copied = destination / manifest_path.name
    # Recheck against target raw files directly; imported metadata cannot define
    # the authoritative original counts, source code or shader controls.
    checked = validate_history_proof(target, copied, original["row"])
    if not checked["valid"]:
        raise ValueError(
            "Imported history binding failed: " + "; ".join(checked["reasons"])
        )
    receipt = {
        "schema_version": 1,
        "kind": "timed_render_history_import/v1",
        "original_run_id": original["row"]["run_id"],
        "history_evidence": artifact(target, copied, "timed_history_manifest"),
        "schedule_sha256": checked["schedule_sha256"],
        "sequence_frames": checked["sequence_frames"],
        "original_input_sha256": original["input_hashes"],
        "artifacts": checked["artifacts"],
        "quality_status": "not_evaluated",
        "performance_claim": False,
    }
    receipt_path = destination / "import_receipt.json"
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError("Existing history import receipt differs")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Import one complete history proof for a subsequent independent evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(import_history_proof(args.history, args.experiment_root), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
