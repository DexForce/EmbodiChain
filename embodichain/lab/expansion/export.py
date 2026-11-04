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

"""Stream frozen accepted artifacts through the existing LeRobot recorder."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import shutil
import hashlib
import os
from dataclasses import replace
import json
from uuid import NAMESPACE_URL, uuid4, uuid5
import numpy as np
import torch
from tensordict import TensorDict
from embodichain.data_pipeline.recording.media import (
    MediaFrameReader,
    _atomic_json,
    _digest,
    _mkdir,
    _sync_directory,
)
from embodichain.data_pipeline.recording.journal import inspect_recording
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from .persistence import ArtifactEpisodeSink, load_expansion_episode, _source_references
from .artifacts import resolve_artifact_path

__all__ = ["export_expansion_lerobot"]


class _ObservationFrames:
    def __init__(
        self, episode, *, start=0, stop=None, owner=None, max_frame_bytes=None
    ):
        self.episode = episode
        self.start, self.stop = start, len(episode.actions) if stop is None else stop
        self.owner = self if owner is None else owner
        if owner is None:
            media = episode.metadata.get("media_manifest")
            self.path = None if media is None else Path(media)
            self.manifest = None if media is None else json.loads(self.path.read_text())
            self.reader = (
                None
                if self.manifest is None
                else MediaFrameReader(
                    self.path,
                    fields=tuple(
                        name
                        for name in self.manifest["features"]
                        if name.startswith("sensor/")
                    ),
                    max_frame_bytes=max_frame_bytes,
                )
            )

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("frame slices must be contiguous")
            return _ObservationFrames(
                self.episode,
                start=self.start + start,
                stop=self.start + stop,
                owner=self.owner,
            )
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        index += self.start
        data = {
            "robot": {
                name: self.episode.observations[key][index].clone()
                for name, key in (
                    ("qpos", "joint_positions"),
                    ("qvel", "joint_velocities"),
                    ("qf", "joint_efforts"),
                )
            }
        }
        owner = self.owner
        if owner.manifest is not None:
            timestamp, values = owner.reader.read(index)
            if not np.isclose(
                timestamp, float(self.episode.timestamps[index]), atol=1e-9, rtol=0
            ):
                raise ValueError("media clock differs from the physical episode")
            sensor = {}
            for name, value in values.items():
                uid, field = name.removeprefix("sensor/").rsplit("/", 1)
                sensor.setdefault(uid, {})[field] = torch.from_numpy(value)
            data["sensor"] = sensor
        return TensorDict(data, batch_size=[])

    def close(self):
        if self.owner.reader is not None:
            self.owner.reader.close()


class _RecordedExpertContext:
    """Writer schema reconstructed exclusively from immutable recorded evidence."""

    def __init__(self, episode):
        from embodichain.lab.gym.envs.expert_trajectory import build_expert_action_spec
        from embodichain.data_pipeline.recording.schema import RecordedCamera

        spec = episode.metadata["expert_action_spec"]
        names = tuple(spec["joint_names"])
        mode = spec.get(
            "joint_command_mode",
            (
                "position"
                if episode.action_representation in {"qpos", "joint_position"}
                else "position_velocity"
            ),
        )
        self.expert_action_spec = build_expert_action_spec(
            joint_names=names, joint_command_mode=mode
        )
        self.step_dt = float(
            spec.get("step_dt", episode.timestamps[1] - episode.timestamps[0])
        )
        if not torch.allclose(
            torch.diff(episode.timestamps),
            torch.full_like(episode.timestamps[1:], self.step_dt),
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("LeRobot export requires a uniform recorded control clock")
        self.active_joint_ids = tuple(range(len(names)))
        self.robot = SimpleNamespace(joint_names=names)
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.observation_manager = SimpleNamespace(active_functors={})
        self.cfg = SimpleNamespace(physics=episode.metadata.get("physics_backend"))
        self.single_observation_space = {"robot": {}}
        self._sensors = {}
        media = episode.metadata.get("media_manifest")
        if media:
            manifest = json.loads(Path(media).read_text())
            sensors = {}
            for name, schema in manifest["features"].items():
                if not name.startswith("sensor/"):
                    continue
                uid, field = name.removeprefix("sensor/").rsplit("/", 1)
                shape = tuple(schema["shape"])
                sensors.setdefault(uid, {})[field] = SimpleNamespace(
                    shape=shape, dtype=np.dtype(schema["dtype"])
                )
                descriptor = RecordedCamera(shape[1], shape[0])
                if uid in self._sensors and self._sensors[uid] != descriptor:
                    raise ValueError("recorded camera frame dimensions disagree")
                self._sensors[uid] = descriptor
            self.single_observation_space["sensor"] = sensors
        self.has_sensors = bool(self._sensors)
        self.signature = (
            self.expert_action_spec,
            self.step_dt,
            json.dumps(manifest["features"], sort_keys=True) if media else None,
        )

    def get_sensor(self, name):
        return self._sensors[name]


def _copy_file(source: Path, destination: Path) -> None:
    _mkdir(destination.parent)
    with source.open("rb") as origin, destination.open("xb") as target:
        shutil.copyfileobj(origin, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    _sync_directory(destination.parent)


def _copy_bundle(root, destination, receipts, children):
    """Copy bounded files before the export is published; no dense media buffers."""
    _mkdir(destination)
    from .artifacts import EpisodeArtifactManifest

    for receipt in receipts:
        source = Path(receipt.storage_id).parent
        if not (source / "artifacts.json").exists():
            original = load_expansion_episode(receipt.storage_id)
            metadata = json_safe_copy(original.metadata, field_name="legacy metadata")
            metadata.setdefault("expansion_source_artifacts", {})[
                "original_episode"
            ] = {
                "path": receipt.storage_id,
                "sha256": _digest(Path(receipt.storage_id)),
            }
            metadata["artifact_derivation"] = {
                "kind": "portable_projection",
                "origin_sha256": _digest(Path(receipt.storage_id)),
                "numerical_evidence": "unchanged",
            }
            projected = replace(original, metadata=metadata)
            ArtifactEpisodeSink(destination).submit(
                projected, submission_id=receipt.submission_id
            )
            continue
        manifest = EpisodeArtifactManifest.read(source / "artifacts.json")
        files = {
            "episode.npz",
            "content.json",
            "artifacts.json",
            *(p.name for p in source.glob("receipt-*.json")),
            *(ref.path for ref in manifest.sources),
        }
        if manifest.media:
            media = json.loads(manifest.media.verify(source).read_text())
            files.add(manifest.media.path)
            files.update(
                (Path(manifest.media.path).parent / chunk["path"]).as_posix()
                for chunk in media["chunks"]
            )
        target = destination / "episodes" / source.name
        for relative in sorted(files):
            origin = resolve_artifact_path(source, relative)
            path = target / relative
            _copy_file(origin, path)
    for child in children:
        media = Path(child["media_manifest"])
        relative = media.parent.relative_to(root)
        manifest = json.loads(media.read_text())
        _mkdir(destination / relative)
        for filename in [media.name, *(chunk["path"] for chunk in manifest["chunks"])]:
            _copy_file(
                resolve_artifact_path(media.parent, filename),
                destination / relative / filename,
            )
        parent = next(
            receipt
            for receipt in receipts
            if receipt.commit_id == child["parent_commit_id"]
        )
        copied = (
            destination
            / "episodes"
            / Path(parent.storage_id).parent.name
            / "episode.npz"
        )
        record = {
            **child,
            "media_manifest": (relative / media.name).as_posix(),
            "source_sha256": _digest(copied),
        }
        if record["source_sha256"] != child["source_sha256"]:
            record["origin_source_sha256"] = child["source_sha256"]
        _mkdir(destination / "observations")
        _atomic_json(
            destination / "observations" / (child["variant_id"] + ".json"), record
        )
    ArtifactEpisodeSink(destination).inspect()
    from .visual import inspect_observation_variants

    inspect_observation_variants(destination)


def _portable_metadata(metadata, dataset):
    root = dataset
    prefix = Path(".")
    for reference in _source_references(metadata):
        reference["path"] = (
            prefix / Path(reference["path"]).relative_to(root)
        ).as_posix()
    if metadata.get("media_manifest"):
        manifest = json.loads(Path(metadata["media_manifest"]).read_text())
        metadata["camera_calibration"] = manifest["calibration"]
        metadata["media_manifest"] = (
            prefix / Path(metadata["media_manifest"]).relative_to(root)
        ).as_posix()
        metadata["calibration_stream"] = {
            "manifest": metadata["media_manifest"],
            "feature_prefix": "calibration/",
            "observations": "T+1",
        }
    metadata["source_artifact"] = (
        prefix / Path(metadata["source_artifact"]).relative_to(root)
    ).as_posix()
    metadata["artifact_reference_root"] = "dataset"
    return metadata


def export_expansion_lerobot(
    environment: object, directory: str | Path, *, verify_only: bool = False
) -> dict[str, object]:
    """Export already committed artifacts without another expert rollout.

    Args:
        environment: Optional live deployment for action-joint compatibility checks.
            Pass None for standalone export; schema comes from frozen evidence.
            This function never steps or resets a live deployment.
        directory: Expansion job directory with verified canonical receipts.
        verify_only: Validate an existing matching export without writing one.
    Returns:
        Published dataset path and episode identities after SDK finalization and
        recording-journal verification. Failed export attempts remain unpublished;
        retry exports the same frozen evidence into a fresh private directory.
    """
    root = Path(directory).resolve()
    marker = root / "lerobot-export.json"
    receipts = ArtifactEpisodeSink(root).inspect()
    from .visual import inspect_observation_variants

    children = inspect_observation_variants(root)
    ids = sorted(
        [receipt.episode_id for receipt in receipts]
        + ["observation_" + child["variant_id"] for child in children]
    )
    expected_uuids = {str(uuid5(NAMESPACE_URL, identity)) for identity in ids}

    def verified(reports):
        return (
            reports["ok"]
            and {row["episode_uuid"] for row in reports["commits"]} == expected_uuids
            and all(row["status"] == "complete" for row in reports["commits"])
        )

    if marker.exists():
        result = json.loads(marker.read_text())
        dataset = resolve_artifact_path(
            root, result["dataset_path"], legacy_absolute=True
        )
        reports = inspect_recording(dataset)
        if not reports["ok"] or not all(
            row["status"] == "complete" for row in reports["commits"]
        ):
            raise ValueError("published LeRobot export is no longer complete")
        if result["episode_ids"] == ids:
            if not verified(reports):
                raise ValueError("published export journal identities differ")
            if result.get("bundle_version") == 2:
                bundle = dataset / "embodichain_artifacts"
                ArtifactEpisodeSink(bundle).inspect()
                inspect_observation_variants(bundle)
            return {**result, "dataset_path": str(dataset)}
        if environment is None:
            # Reconstruct the writer below; changing parents produces a new snapshot.
            pass
    if verify_only:
        raise ValueError("a newer artifact set needs export")
    if not receipts:
        raise ValueError("there are no confirmed episodes to export")
    execution = (
        json.loads((root / "execution.json").read_text())["config"]
        if (root / "execution.json").exists()
        else {}
    )
    frame_budget = execution.get("media_buffer_bytes", 256 * 1024 * 1024)
    first = load_expansion_episode(receipts[0].storage_id)
    recorded = _RecordedExpertContext(first)
    if (
        environment is not None
        and tuple(environment.expert_action_spec.joint_names)
        != recorded.expert_action_spec.joint_names
    ):
        raise ValueError("export action schema differs from deployment")
    environment = recorded
    destination = root / "exports" / uuid4().hex
    recorder = LeRobotRecorder(
        DatasetFunctorCfg(
            func=LeRobotRecorder,
            params={
                "save_path": str(destination),
                "robot_meta": {"robot_type": "expansion"},
                "extra": {"scene_type": "qualified", "task_description": "experts"},
                "image_writer_threads": 0,
                "use_videos": False,
                "depth_video": {"enable": True, "keep_numeric_fallback": False},
            },
        ),
        environment,
    )

    def artifacts():
        for receipt in receipts:
            parent = load_expansion_episode(
                dataset_root
                / "embodichain_artifacts"
                / "episodes"
                / Path(receipt.storage_id).parent.name
                / "episode.npz"
            )
            yield receipt, parent
            for child in children:
                if child["parent_commit_id"] != parent.commit_id:
                    continue
                metadata = json_safe_copy(parent.metadata, field_name="parent metadata")
                metadata.update(
                    media_manifest=str(
                        dataset_root
                        / "embodichain_artifacts"
                        / Path(child["media_manifest"]).relative_to(root)
                    ),
                    media_manifest_sha256=child["media_sha256"],
                    parent_episode_uuid=str(uuid5(NAMESPACE_URL, parent.episode_id)),
                    observation_origin="state_rerender",
                    observation_variant_id=child["variant_id"],
                    visual_profile=child["profile"],
                    physical_parent_commit_id=parent.commit_id,
                )
                yield receipt, replace(
                    parent,
                    episode_id="observation_" + child["variant_id"],
                    commit_id="observation_" + child["variant_id"],
                    metadata=metadata,
                )

    try:
        dataset_root = Path(recorder.dataset_path)
        _copy_bundle(root, dataset_root / "embodichain_artifacts", receipts, children)
        for receipt, episode in artifacts():
            if _RecordedExpertContext(episode).signature != recorded.signature:
                raise ValueError(
                    "recorded action/observation schema differs across export episodes"
                )
            if tuple(episode.metadata["expert_action_spec"]["joint_names"]) != tuple(
                environment.expert_action_spec.joint_names
            ):
                raise ValueError("export action schema differs from deployment")
            metadata = json_safe_copy(episode.metadata, field_name="episode metadata")
            metadata.update(
                episode_uuid=str(uuid5(NAMESPACE_URL, episode.episode_id)),
                run_uuid=str(uuid5(NAMESPACE_URL, str(root.resolve()))),
                expansion_episode_id=episode.episode_id,
                expansion_commit_id=episode.commit_id,
                source_artifact=str(
                    dataset_root
                    / "embodichain_artifacts"
                    / "episodes"
                    / Path(receipt.storage_id).parent.name
                    / "episode.npz"
                ),
            )
            metadata = _portable_metadata(metadata, dataset_root)
            frames = _ObservationFrames(episode, max_frame_bytes=frame_budget)
            try:
                recorder.persist_episode_payload(
                    0, frames, episode.actions, episode_metadata=metadata
                )
            finally:
                frames.close()
        path = recorder.finalize()
        reports = inspect_recording(path)
        if not verified(reports):
            raise RuntimeError(
                "LeRobot export did not produce complete recording journals"
            )
        result = {
            "dataset_path": Path(path).relative_to(root).as_posix(),
            "episode_ids": ids,
            "episodes": len(ids),
            "bundle_version": 2,
        }
        snapshots = root / "export_snapshots"
        snapshots.mkdir(exist_ok=True)
        snapshot_id = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        _atomic_json(snapshots / (snapshot_id + ".json"), result)
        _atomic_json(marker, result)
        return {**result, "dataset_path": str(path)}
    finally:
        recorder.finalize()
