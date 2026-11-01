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
from bisect import bisect_right
from pathlib import Path
from dataclasses import replace
import json
from uuid import NAMESPACE_URL, uuid4, uuid5
import numpy as np
import torch
from tensordict import TensorDict
from embodichain.data_pipeline.recording.media import _atomic_json
from embodichain.data_pipeline.recording.journal import inspect_recording
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.lab.gym.envs.managers.real2sim_collection import (
    load_staged_real2sim_episode,
)
from .persistence import ArtifactEpisodeSink

__all__ = ["export_expansion_lerobot"]


class _ObservationFrames:
    def __init__(self, episode, *, start=0, stop=None, owner=None):
        self.episode = episode
        self.start, self.stop = start, len(episode.actions) if stop is None else stop
        self.owner = self if owner is None else owner
        if owner is None:
            media = episode.metadata.get("media_manifest")
            self.path = None if media is None else Path(media)
            self.manifest = None if media is None else json.loads(self.path.read_text())
            self.cache_index = None
            self.cache = {}

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
                name: self.episode.observations[key][index]
                for name, key in (
                    ("qpos", "joint_positions"),
                    ("qvel", "joint_velocities"),
                    ("qf", "joint_efforts"),
                )
            }
        }
        owner = self.owner
        if owner.manifest is not None:
            chunks = owner.manifest["chunks"]
            chunk_index = bisect_right([chunk["start"] for chunk in chunks], index) - 1
            chunk = chunks[chunk_index]
            if owner.cache_index != chunk_index:
                owner.cache.clear()
                with np.load(
                    owner.path.parent / chunk["path"], allow_pickle=False
                ) as arrays:
                    owner.cache = {
                        key: arrays[key] for key in arrays.files if key != "timestamps"
                    }
                owner.cache_index = chunk_index
            sensor = {}
            for i, name in enumerate(sorted(owner.manifest["features"])):
                if not name.startswith("sensor/"):
                    continue
                uid, field = name.removeprefix("sensor/").rsplit("/", 1)
                sensor.setdefault(uid, {})[field] = torch.from_numpy(
                    owner.cache[f"feature_{i}"][index - chunk["start"]]
                )
            data["sensor"] = sensor
        return TensorDict(data, batch_size=[])


def export_expansion_lerobot(
    environment: object, directory: str | Path
) -> dict[str, object]:
    """Export already committed artifacts without another expert rollout.

    Args:
        environment: Live deployment supplying the recorder's exact feature schema.
            This function never steps or resets it.
        directory: Expansion job directory with verified canonical receipts.
    Returns:
        Published dataset path and episode identities after SDK finalization and
        recording-journal verification. Failed export attempts remain unpublished;
        retry exports the same frozen evidence into a fresh private directory.
    """
    root = Path(directory)
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
        if result["episode_ids"] != ids:
            raise ValueError("published export differs from committed episode set")
        reports = inspect_recording(result["dataset_path"])
        if not verified(reports):
            raise ValueError("published LeRobot export is no longer complete")
        return result
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
            parent = load_staged_real2sim_episode(receipt.storage_id)
            yield receipt, parent
            for child in children:
                if child["parent_commit_id"] != parent.commit_id:
                    continue
                metadata = json_safe_copy(parent.metadata, field_name="parent metadata")
                metadata.update(
                    media_manifest=child["media_manifest"],
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
        for receipt, episode in artifacts():
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
                source_artifact=str(Path(receipt.storage_id).resolve()),
            )
            recorder._persist_episode_payload(
                0,
                _ObservationFrames(episode),
                episode.actions,
                episode_metadata=metadata,
            )
        path = recorder.finalize()
        reports = inspect_recording(path)
        if not verified(reports):
            raise RuntimeError(
                "LeRobot export did not produce complete recording journals"
            )
        result = {"dataset_path": str(path), "episode_ids": ids, "episodes": len(ids)}
        _atomic_json(marker, result)
        return result
    finally:
        recorder.finalize()
