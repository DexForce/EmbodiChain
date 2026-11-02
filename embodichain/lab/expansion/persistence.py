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

"""Content-verified local expert episode commits and write-only recovery."""

from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np
from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.sim.motion.expansion.contracts import ExpertEpisode, CommitReceipt
from embodichain.data_pipeline.recording.media import (
    _atomic_json,
    _mkdir,
    _sync_directory,
    _digest,
    verify_media_manifest,
)

from .artifacts import ArtifactReference, EpisodeArtifactManifest, resolve_artifact_path

__all__ = ["ArtifactEpisodeSink", "load_expansion_episode"]


def _source_references(metadata):
    references = [
        reference
        for bundle in metadata.get("real2sim_source_artifacts", {}).values()
        for reference in bundle.values()
    ]
    references.extend(metadata.get("expansion_source_artifacts", {}).values())
    if metadata.get("real2sim_lineage") is not None:
        references.append(metadata["real2sim_lineage"]["trace"])
    if metadata.get("real2sim_observed_geometry") is not None:
        references.append(metadata["real2sim_observed_geometry"])
    return references


def load_expansion_episode(path: str | Path) -> ExpertEpisode:
    """Load bounded numerical evidence and resolve bundle-local source/media references.

    Legacy staged episodes remain readable in place. New artifacts carry only
    relative references and can be moved without changing their frozen bytes.
    """
    from embodichain.lab.gym.envs.managers.real2sim_collection import (
        load_staged_real2sim_episode,
    )

    path = Path(path)
    episode = load_staged_real2sim_episode(path)
    metadata = json_safe_copy(episode.metadata, field_name="episode metadata")
    portable = metadata.get("artifact_schema_version") == 2
    for reference in _source_references(metadata):
        reference["path"] = str(
            resolve_artifact_path(
                path.parent, reference["path"], legacy_absolute=not portable
            )
        )
    if metadata.get("media_manifest"):
        metadata["media_manifest"] = str(
            resolve_artifact_path(
                path.parent, metadata["media_manifest"], legacy_absolute=not portable
            )
        )
    return replace(episode, metadata=metadata)


class ArtifactEpisodeSink:
    """Idempotently persist accepted episodes with a final durable receipt.

    Args:
        directory: Job-owned output directory. Each commit owns an addressed folder.

    The numeric NPZ uses the existing staged episode schema and retains actual
    T actions and T+1 observations. Media is verified before publishing a receipt.
    One host owns writes; cross-process job serialization belongs to the runner.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        _mkdir(self.directory)

    def _archive_reference(self, reference: dict, destination: Path) -> dict:
        source = Path(reference["path"])
        checksum = reference["sha256"]
        _mkdir(destination.parent)
        if destination.exists():
            if _digest(destination) != checksum:
                raise ValueError("archived source content changed")
        else:
            if _digest(source) != checksum:
                raise ValueError("source attachment changed before staging")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent, prefix=".source-"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream, source.open("rb") as origin:
                    shutil.copyfileobj(origin, stream, 1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _digest(temporary) != checksum:
                    raise ValueError("source attachment changed while copying")
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        return {**reference, "path": str(destination.resolve())}

    def _archive_sources(
        self, metadata: dict, directory: Path, *, portable: bool = False
    ) -> None:
        paths = {
            "prototype_manifest": "prototype.json",
            "normalized_arrays": "normalized.npz",
            "source_manifest": "source/trajectory.json",
            "source_arrays": "source/trajectory.npz",
        }
        for key, bundle in metadata.get("real2sim_source_artifacts", {}).items():
            name = hashlib.sha256(str(key).encode()).hexdigest()[:16]
            for field, reference in tuple(bundle.items()):
                relative = paths.get(
                    field,
                    "evidence/"
                    + hashlib.sha256(field.encode()).hexdigest()
                    + Path(reference["path"]).suffix,
                )
                bundle[field] = self._archive_reference(
                    reference, directory / "sources" / name / relative
                )
        for field, reference in tuple(
            metadata.get("expansion_source_artifacts", {}).items()
        ):
            metadata["expansion_source_artifacts"][field] = self._archive_reference(
                reference,
                directory
                / "job_sources"
                / (
                    hashlib.sha256(field.encode()).hexdigest()
                    + Path(reference["path"]).suffix
                ),
            )
        lineage = metadata.get("real2sim_lineage")
        if lineage is not None:
            lineage["trace"] = self._archive_reference(
                lineage["trace"], directory / "trace.npz"
            )
        geometry = metadata.get("real2sim_observed_geometry")
        if geometry is not None:
            metadata["real2sim_observed_geometry"] = self._archive_reference(
                geometry, directory / "observed-geometry.npz"
            )

        if portable:
            for reference in _source_references(metadata):
                reference["path"] = (
                    Path(reference["path"]).relative_to(directory.resolve()).as_posix()
                )

    def _archive_media(self, episode: ExpertEpisode, directory: Path) -> str | None:
        original = episode.metadata.get("media_manifest")
        if original is None:
            return None
        original = Path(original)
        expected = episode.metadata.get("media_manifest_sha256")
        if expected is None or _digest(original) != expected:
            raise ValueError("episode media manifest changed or is not attested")
        media = verify_media_manifest(
            original,
            expected_frames=len(episode.timestamps),
            parent_episode_id=episode.episode_id,
            expected_timestamps=episode.timestamps.cpu().numpy(),
        )
        destination = directory / "media"
        for chunk in media["chunks"]:
            self._archive_reference(
                {
                    "path": str(original.parent / chunk["path"]),
                    "sha256": chunk["sha256"],
                },
                destination / chunk["path"],
            )
        self._archive_reference(
            {"path": str(original), "sha256": expected}, destination / "manifest.json"
        )
        return "media/manifest.json"

    @staticmethod
    def _verify_sources(metadata: dict) -> None:
        references = [
            reference
            for bundle in metadata.get("real2sim_source_artifacts", {}).values()
            for reference in bundle.values()
        ]
        references.extend(metadata.get("expansion_source_artifacts", {}).values())
        if metadata.get("real2sim_lineage") is not None:
            references.append(metadata["real2sim_lineage"]["trace"])
        if metadata.get("real2sim_observed_geometry") is not None:
            references.append(metadata["real2sim_observed_geometry"])
        for reference in references:
            if _digest(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("archived source attachment changed")

    def stage(self, episode: ExpertEpisode) -> Path:
        """Durably freeze accepted numeric/media evidence before attempting commit."""
        return self._write(episode, submission_id=0, confirm=False)

    def submit(self, episode: ExpertEpisode, *, submission_id: int) -> CommitReceipt:
        """Publish an identity-matching receipt for durably verified episode content."""
        return self._write(episode, submission_id=submission_id, confirm=True)

    def _write(
        self, episode: ExpertEpisode, *, submission_id: int, confirm: bool
    ) -> CommitReceipt | Path:
        """Commit one accepted, frozen episode or reconcile an identical retry.

        Args:
            episode: Measured evidence accepted by the canonical session.
            submission_id: Nonnegative session write-submission ordinal.
        Returns:
            Identity-matching confirmed receipt only after content verification.
        """
        if not isinstance(episode, ExpertEpisode) or not episode.validation.accepted:
            raise ValueError("only measured accepted ExpertEpisode values can commit")
        if type(submission_id) is not int or submission_id < 0:
            raise ValueError("submission_id must be nonnegative")
        key = hashlib.sha256(episode.commit_id.encode()).hexdigest()
        directory = self.directory / "episodes" / key
        _mkdir(directory)
        portable = True
        if (directory / "episode.npz").exists():
            with np.load(directory / "episode.npz", allow_pickle=False) as saved:
                portable = (
                    json.loads(saved["manifest"].tobytes())["metadata"].get(
                        "artifact_schema_version"
                    )
                    == 2
                )
        names = sorted(episode.observations)
        metadata = {
            "identity": asdict(episode.identity),
            "episode_id": episode.episode_id,
            "commit_id": episode.commit_id,
            "action_representation": episode.action_representation,
            "observations": names,
            "validation": [
                {
                    "check_id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                    "metrics": dict(check.metrics),
                }
                for check in episode.validation.checks
            ],
            "phases": [asdict(phase) for phase in episode.phases],
            "metadata": json_safe_copy(episode.metadata, field_name="episode metadata"),
        }
        self._archive_sources(metadata["metadata"], directory, portable=portable)
        if portable:
            metadata["metadata"]["artifact_schema_version"] = 2
            archived_media = self._archive_media(episode, directory)
            if archived_media is not None:
                metadata["metadata"]["media_manifest"] = archived_media
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        arrays = {
            f"observation_{i}": episode.observations[name].detach().cpu().numpy()
            for i, name in enumerate(names)
        }
        arrays.update(
            actions=episode.actions.detach().cpu().numpy(),
            timestamps=episode.timestamps.detach().cpu().numpy(),
            manifest=np.frombuffer(encoded, dtype=np.uint8),
        )
        media = episode.metadata.get("media_manifest")
        if media is not None:
            expected_media = episode.metadata.get("media_manifest_sha256")
            if expected_media is None or _digest(Path(media)) != expected_media:
                raise ValueError("episode media manifest changed or is not attested")
            verify_media_manifest(
                media,
                expected_frames=len(episode.timestamps),
                parent_episode_id=episode.episode_id,
                expected_timestamps=episode.timestamps.detach().cpu().numpy(),
            )
        destination = directory / "episode.npz"
        descriptor, name = tempfile.mkstemp(dir=directory, prefix=".pending-")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size > 128 * 1024 * 1024:
                raise BufferError(
                    "numeric episode exceeds the 128 MiB archive reader limit"
                )
            try:
                os.link(temporary, destination)
            except FileExistsError:
                with np.load(destination, allow_pickle=False) as existing:
                    if set(existing.files) != set(arrays) or any(
                        not np.array_equal(existing[key], value)
                        for key, value in arrays.items()
                    ):
                        raise ValueError(
                            "commit identity already owns different episode bytes"
                        )
            receipt = CommitReceipt(
                episode_id=episode.episode_id,
                commit_id=episode.commit_id,
                candidate_id=episode.identity.candidate_id,
                attempt_id=episode.identity.attempt_id,
                scene_case_id=episode.identity.scene_case_id,
                submission_id=submission_id,
                storage_id=str(destination.resolve()),
                confirmed=True,
            )
            _atomic_json(
                directory / "content.json",
                {
                    "sha256": _digest(destination),
                    "frames": len(episode.timestamps),
                    "media_manifest": metadata["metadata"].get("media_manifest"),
                    "media_sha256": None if media is None else _digest(Path(media)),
                },
            )
            if portable:
                references = tuple(
                    ArtifactReference(ref["path"], ref["sha256"], "source")
                    for ref in _source_references(metadata["metadata"])
                )
                artifact = EpisodeArtifactManifest(
                    episode.episode_id,
                    episode.commit_id,
                    episode.metadata.get("recipe_id"),
                    episode.identity.attempt_id,
                    ArtifactReference("episode.npz", _digest(destination), "numerical"),
                    (
                        ArtifactReference(
                            metadata["metadata"]["media_manifest"],
                            metadata["metadata"]["media_manifest_sha256"],
                            "media",
                        )
                        if media
                        else None
                    ),
                    references,
                    episode.metadata.get("physical_parent_episode_id"),
                    source_ancestors=tuple(
                        episode.metadata.get("source_episode_uuids")
                        or (
                            [episode.metadata["source_episode_uuid"]]
                            if episode.metadata.get("source_episode_uuid")
                            else []
                        )
                    ),
                )
                _atomic_json(directory / "artifacts.json", artifact.to_dict())
            if not confirm:
                return destination
            stored = replace(
                receipt, storage_id=destination.relative_to(self.directory).as_posix()
            )
            _atomic_json(directory / f"receipt-{submission_id}.json", asdict(stored))
            return receipt
        finally:
            temporary.unlink(missing_ok=True)

    def recover(self, commit_id: str) -> CommitReceipt | None:
        """Confirm a complete staged episode without executing its source again."""
        directory = (
            self.directory / "episodes" / hashlib.sha256(commit_id.encode()).hexdigest()
        )
        path = directory / "episode.npz"
        if not path.exists():
            return None
        if (directory / "content.json").exists():
            content = json.loads((directory / "content.json").read_text())
            if _digest(path) != content["sha256"]:
                raise ValueError("staged episode content changed")
            if (
                content["media_manifest"] is not None
                and _digest(
                    resolve_artifact_path(
                        directory, content["media_manifest"], legacy_absolute=True
                    )
                )
                != content["media_sha256"]
            ):
                raise ValueError("staged media manifest changed")
        # An atomically linked NPZ may precede the content manifest at a crash.
        # Its bounded loader validates ZIP CRCs and the complete value contract;
        # submit then revalidates every source and media pin before confirming.
        episode = load_expansion_episode(path)
        if episode.commit_id != commit_id:
            raise ValueError("staged episode identity differs")
        self._verify_sources(episode.metadata)
        return self.submit(episode, submission_id=0)

    def inspect(self) -> tuple[CommitReceipt, ...]:
        """Read verified committed episodes, deduplicated by commit identity.

        Returns:
            Latest verified receipt for each commit. Partial writes have no credit.
        """
        receipts = {}
        for path in sorted((self.directory / "episodes").glob("*/receipt-*.json")):
            receipt = CommitReceipt(**json.loads(path.read_text()))
            storage = resolve_artifact_path(
                self.directory, receipt.storage_id, legacy_absolute=True
            )
            receipt = replace(receipt, storage_id=str(storage.resolve()))
            content = json.loads((path.parent / "content.json").read_text())
            episode = path.parent / "episode.npz"
            if (
                _digest(episode) != content["sha256"]
                or Path(receipt.storage_id).resolve() != episode.resolve()
            ):
                raise ValueError("committed episode content or receipt address changed")
            data = load_expansion_episode(episode)
            if (path.parent / "artifacts.json").exists():
                artifact = EpisodeArtifactManifest.read(path.parent / "artifacts.json")
                artifact.verify(path.parent)
                if (artifact.episode_id, artifact.commit_id, artifact.attempt_id) != (
                    data.episode_id,
                    data.commit_id,
                    data.identity.attempt_id,
                ):
                    raise ValueError(
                        "artifact identity differs from numerical evidence"
                    )
                if (
                    artifact.recipe_id != data.metadata.get("recipe_id")
                    or artifact.numerical.sha256 != content["sha256"]
                ):
                    raise ValueError("artifact recipe or numerical reference differs")
                if (artifact.media is None) != (content["media_manifest"] is None) or (
                    artifact.media is not None
                    and (
                        artifact.media.path != content["media_manifest"]
                        or artifact.media.sha256 != content["media_sha256"]
                    )
                ):
                    raise ValueError("artifact media reference differs")
            if not receipt.confirmed or any(
                (
                    receipt.episode_id != data.episode_id,
                    receipt.commit_id != data.commit_id,
                    receipt.candidate_id != data.identity.candidate_id,
                    receipt.attempt_id != data.identity.attempt_id,
                    receipt.scene_case_id != data.identity.scene_case_id,
                    path.parent.name
                    != hashlib.sha256(data.commit_id.encode()).hexdigest(),
                    content["frames"] != len(data.timestamps),
                    (
                        str(
                            resolve_artifact_path(
                                path.parent,
                                content["media_manifest"],
                                legacy_absolute=True,
                            )
                        )
                        if content["media_manifest"]
                        else None
                    )
                    != data.metadata.get("media_manifest"),
                )
            ):
                raise ValueError("receipt differs from frozen episode identity")
            self._verify_sources(data.metadata)
            if content["media_manifest"] is not None:
                if (
                    _digest(
                        resolve_artifact_path(
                            path.parent, content["media_manifest"], legacy_absolute=True
                        )
                    )
                    != content["media_sha256"]
                ):
                    raise ValueError("committed media manifest changed")
                verify_media_manifest(
                    resolve_artifact_path(
                        path.parent, content["media_manifest"], legacy_absolute=True
                    ),
                    expected_frames=content["frames"],
                    parent_episode_id=receipt.episode_id,
                    expected_timestamps=data.timestamps.cpu().numpy(),
                )
            previous = receipts.get(receipt.commit_id)
            if previous is None or receipt.submission_id > previous.submission_id:
                receipts[receipt.commit_id] = receipt
        return tuple(receipts.values())
