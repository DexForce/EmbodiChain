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


"""One independently readable LeRobot dataset per accepted expansion episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from embodichain.data_pipeline.recording.media import (
    _digest as file_digest,
    _atomic_json as atomic_json,
)

__all__ = ["export_expansion_episodes"]


def _verify_dataset(dataset: Path, expected: dict) -> None:
    from embodichain.data_pipeline.recording.journal import inspect_recording
    from embodichain.lab.expansion import ArtifactEpisodeSink

    saved = json.loads((dataset / "expansion-episode.json").read_text())
    if saved != expected:
        raise ValueError(f"LeRobot dataset belongs to different evidence: {dataset}")
    info = json.loads((dataset / "meta/info.json").read_text())
    if (info["total_episodes"], info["total_frames"]) != (1, expected["steps"]):
        raise ValueError(f"Expected exactly one complete LeRobot episode: {dataset}")
    report = inspect_recording(dataset)
    identity = str(uuid5(NAMESPACE_URL, expected["episode_id"]))
    if not report["ok"] or [row["episode_uuid"] for row in report["commits"]] != [
        identity
    ]:
        raise ValueError(f"Incomplete LeRobot recording: {dataset}: {report}")
    receipts = ArtifactEpisodeSink(dataset / "embodichain_artifacts").inspect()
    if [(row.episode_id, row.commit_id) for row in receipts] != [
        (expected["episode_id"], expected["commit_id"])
    ]:
        raise ValueError(f"LeRobot evidence bundle differs: {dataset}")


def export_expansion_episodes(directory: str | Path) -> dict:
    """Save each accepted episode under its source-owned output directory.

    Each returned
    ``dataset_path`` is relative to ``directory/lerobot`` and contains exactly
    one LeRobot episode. The full T+1 evidence and media remain in the portable
    ``embodichain_artifacts/`` bundle alongside the T action rows.

    Args:
        directory: Source-owned ArtifactEpisodeSink directory with confirmed
            native receipts. It is read and fully verified before exporting.

    Returns:
        Manifest with the episode count, export directory and content-addressed
        relative dataset paths. Each dataset contains exactly one accepted
        episode and its complete portable source/media evidence. Existing exports
        are verified rather than silently overwritten. No World is constructed.
    """
    from embodichain.lab.expansion import ArtifactEpisodeSink, load_expansion_episode
    from embodichain.lab.expansion.export import export_expansion_lerobot

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Expansion directory does not exist: {directory}")
    receipts = ArtifactEpisodeSink(directory).inspect()
    output = directory / "lerobot"
    output.mkdir(parents=True, exist_ok=True)
    datasets = []
    for receipt in receipts:
        episode = load_expansion_episode(receipt.storage_id)
        # Content-addressed names cannot introduce path components from user IDs.
        key = hashlib.sha256(receipt.commit_id.encode()).hexdigest()
        destination = output / key
        identity = {
            "schema_version": 1,
            "episode_id": episode.episode_id,
            "commit_id": episode.commit_id,
            "source_sha256": file_digest(Path(receipt.storage_id)),
            "steps": len(episode.actions),
        }
        if destination.exists():
            _verify_dataset(destination, identity)
        else:
            # The native exporter accepts a complete job. Give it one frozen
            # accepted episode, then publish only the self-contained dataset.
            # No environment, physics replay or render window is constructed.
            with TemporaryDirectory(prefix=".lerobot-", dir=output) as temporary:
                isolated = Path(temporary) / "job"
                source = Path(receipt.storage_id).parent
                if (source / "artifacts.json").is_file():
                    # Portable committed evidence is already frozen and fully
                    # verified above. Preserve its bytes and receipts instead
                    # of loading/freezing/compressing an identical new episode.
                    shutil.copytree(source, isolated / "episodes" / source.name)
                else:
                    ArtifactEpisodeSink(isolated).submit(episode, submission_id=0)
                result = export_expansion_lerobot(None, isolated)
                dataset = Path(result["dataset_path"])
                atomic_json(dataset / "expansion-episode.json", identity)
                _verify_dataset(dataset, identity)
                dataset.rename(destination)
        datasets.append({**identity, "dataset_path": key})
    manifest = {
        "schema_version": 1,
        "format": "lerobot",
        "episodes": len(datasets),
        "datasets": datasets,
    }
    atomic_json(output / "manifest.json", manifest)
    return {**manifest, "directory": str(output)}
