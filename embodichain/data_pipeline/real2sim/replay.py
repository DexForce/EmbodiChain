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

"""Read-only replay indexes over committed Real2Sim recording attachments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import NAMESPACE_URL, uuid5

from embodichain.data_pipeline.recording.journal import inspect_recording
from .io import load_prototype

__all__ = ["export_real2sim_replay_manifest"]


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _json(path: Path) -> object:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("replay metadata exceeds its read budget")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(root: Path, value: Mapping[str, object]) -> Path:
    if not isinstance(value, Mapping) or not {"path", "sha256"} <= set(value):
        raise ValueError("replay attachment needs path and SHA-256")
    path = (root / str(value["path"])).resolve()
    if (
        root not in path.parents
        or not path.is_file()
        or _digest(path) != value["sha256"]
    ):
        raise ValueError("replay attachment is missing, changed or outside its dataset")
    return path


def export_real2sim_replay_manifest(
    datasets: Sequence[str | Path],
    destination: str | Path,
    *,
    source_assets: Mapping[str, Mapping[str, str]] | None = None,
) -> Path:
    """Export verified recording references without executing or resubmitting.

    Existing candidate, attempt, commit, submission and episode identities are
    retained. A confirmed receipt is necessary, along with a complete recorder
    journal and matching source/trace/evidence digests. The index does not turn
    an unexecuted target path into a physical execution or certify dynamics.
    Failed attempts are retained separately even for datasets with no commits.

    Args:
        datasets: Local recorder roots, each containing ``meta/info.json``.
        destination: New JSON index; existing exports are never overwritten.
        source_assets: Optional named original videos or calibration evidence,
            each with a local absolute ``path`` and pinned ``sha256``. These
            are read-only source references, not additional collection sessions.

    Returns:
        Absolute path of the newly published replay index.

    Raises:
        ValueError: A recording, identity, source or necessary attachment cannot
            be verified. Nothing is written to any input dataset.
        FileExistsError: The destination already exists.
    """
    if isinstance(datasets, (str, bytes)) or not datasets:
        raise ValueError("replay export needs an explicit sequence of datasets")
    path = Path(destination).resolve()
    if path.exists():
        raise FileExistsError(path)
    roots = tuple(Path(value).resolve() for value in datasets)
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate replay dataset")
    entries, failures, recordings, commits = [], [], [], set()
    for root in roots:
        if not (root / "meta/info.json").is_file():
            raise ValueError("replay input must be an existing recorder root")
        journal = inspect_recording(root)
        if journal["errors"] or any(
            item["status"] != "complete" for item in journal["commits"]
        ):
            raise ValueError("recorder journal has incomplete or corrupt commits")
        rows_path = root / "meta/embodichain_episodes.jsonl"
        rows = []
        if rows_path.exists():
            if rows_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("episode index exceeds its read budget")
            rows = [
                json.loads(line)
                for line in rows_path.read_text().splitlines()
                if line.strip()
            ]
        journal_entries = {item["episode_uuid"]: item for item in journal["commits"]}
        info = _json(root / "meta/info.json")
        if (
            len(journal_entries) != len(journal["commits"])
            or len(rows) != len(journal_entries)
            or info["total_episodes"] != len(rows)
        ):
            raise ValueError(
                "every replay episode requires a matching complete journal entry"
            )
        for row in rows:
            record = journal_entries.get(row["episode_uuid"])
            if (
                record is None
                or record["lerobot_episode_index"] != row["lerobot_episode_index"]
            ):
                raise ValueError(
                    "replay episode is missing its matching complete journal entry"
                )
            receipt_path = (
                root
                / "meta/real2sim/receipts"
                / f"{uuid5(NAMESPACE_URL, row['commit_id'])}.json"
            )
            receipt = _json(receipt_path)
            expected = dict(
                candidate_id=row["candidate_id"],
                attempt_id=row["attempt_id"],
                commit_id=row["commit_id"],
                storage_id=f"{root}#{row['episode_uuid']}",
            )
            if receipt.get("confirmed") is not True or any(
                receipt.get(name) != value for name, value in expected.items()
            ):
                raise ValueError("replay receipt does not match the existing episode")
            if row["commit_id"] in commits:
                raise ValueError("duplicate replay commit identity")
            commits.add(row["commit_id"])
            lineage = row["real2sim_lineage"]
            _artifact(root, lineage["trace"])
            for name in ("real2sim_episode_evidence", "real2sim_observed_geometry"):
                _artifact(root, row[name])
            sources = []
            for call in lineage["calls"]:
                bundle = row["real2sim_artifacts"][call["source_bundle"]]
                for value in bundle.values():
                    _artifact(root, value)
                prototype = load_prototype(root / bundle["prototype_manifest"]["path"])
                source = prototype.unit.source
                if bundle["prototype_manifest"]["sha256"] != call[
                    "prototype_sha256"
                ] or (source.source_id, source.source_revision) != (
                    call.get("source_id", row["source_id"]),
                    call.get("source_revision", row["source_revision"]),
                ):
                    raise ValueError(
                        "replay source identity differs from its call lineage"
                    )
                if (
                    source.command_representation
                    == "droid_arm_position_rad_and_gripper_closed_fraction"
                ):
                    from .droid import verify_droid_unit_evidence

                    verify_droid_unit_evidence(
                        root / bundle["source_evidence"]["path"],
                        root / bundle["original_recording"]["path"],
                        source,
                    )
                sources.append(
                    dict(
                        call_index=call["call_index"],
                        origin=source.origin,
                        pose_origin=source.pose_origin,
                        source_id=source.source_id,
                        source_revision=source.source_revision,
                        unit_id=source.unit_id,
                        prototype_sha256=call["prototype_sha256"],
                    )
                )
            entries.append(
                dict(dataset=str(root), receipt=receipt, sources=sources, metadata=row)
            )
        for failure in sorted((root / "meta/real2sim/failed-attempts").glob("*.json")):
            detail = _json(failure)
            if "partial_rollout" in detail:
                _artifact(root, detail["partial_rollout"])
            failures.append(
                dict(
                    dataset=str(root),
                    path=str(failure),
                    sha256=_digest(failure),
                    detail=detail,
                )
            )
        recordings.append(
            dict(dataset=str(root), episodes=len(rows), journal_complete=True)
        )
    assets = {}
    for name, item in (source_assets or {}).items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
        ):
            raise ValueError("source asset needs a name, path and pinned SHA-256")
        source = Path(item["path"])
        if (
            not source.is_absolute()
            or not source.is_file()
            or _digest(source) != item["sha256"]
        ):
            raise ValueError("original replay source asset is missing or changed")
        assets[name] = dict(path=str(source.resolve()), sha256=item["sha256"])
    payload = dict(
        schema="real2sim-replay/v1",
        read_only=True,
        recordings=recordings,
        episodes=entries,
        failed_attempts=failures,
        source_assets=assets,
        collection_identity_created=False,
    )
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
