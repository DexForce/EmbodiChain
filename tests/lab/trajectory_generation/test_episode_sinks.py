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

"""Real LeRobot episode sealing, causal evidence, and submission retries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink

_FPS = 20
_STEPS = 3


def _episode(commit: str = "commit-1", *, rgb: bool = False) -> ExpertEpisode:
    observations = {
        "joint_positions": torch.arange((_STEPS + 1) * 2, dtype=torch.float32).reshape(
            _STEPS + 1, 2
        )
        / 10,
        "scalar": torch.arange(_STEPS + 1, dtype=torch.int64).reshape(-1, 1),
    }
    if rgb:
        observations["images.front"] = torch.arange(
            (_STEPS + 1) * 4 * 5 * 3, dtype=torch.uint8
        ).reshape(_STEPS + 1, 4, 5, 3)
    return ExpertEpisode(
        identity=CandidateIdentity(
            "case",
            "initial",
            f"candidate-{commit}",
            "family",
            "source",
            "revision",
            "template",
        ),
        observations=observations,
        actions=torch.tensor([[0.1], [0.2], [0.3]]),
        timestamps=7 + torch.arange(_STEPS + 1, dtype=torch.float64) / _FPS,
        action_representation="qpos",
        validation=ValidationResult((ValidationCheck("task_success", "passed"),)),
        episode_id=f"episode-{commit}",
        commit_id=commit,
        metadata={"task": "move the cube", "nested": {"actual": True}},
    )


def test_real_lerobot_sealing_preserves_causal_frames_rgb_and_terminal_evidence(
    tmp_path: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episode = _episode(rgb=True)
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        receipt = sink.submit(episode)
        assert receipt.confirmed, receipt.error
        assert sink.drain() == ()
    shard = Path(receipt.storage_id)
    dataset = LeRobotDataset(
        repo_id="embodichain/trajectory-generation", root=shard / "dataset"
    )
    assert dataset.num_episodes == 1
    assert len(dataset) == _STEPS
    assert torch.equal(
        dataset[0]["observation.joint_positions"],
        episode.observations["joint_positions"][0],
    )
    assert dataset[_STEPS - 1]["action"].item() == pytest.approx(
        episode.actions[-1].item()
    )
    assert dataset[0]["observation.images.front"].shape == (3, 4, 5)
    with np.load(shard / "terminal.npz", allow_pickle=False) as evidence:
        np.testing.assert_array_equal(
            evidence["timestamps"], episode.timestamps.numpy()
        )
        np.testing.assert_array_equal(
            evidence["observation.joint_positions"],
            episode.observations["joint_positions"][-1].numpy(),
        )
        np.testing.assert_array_equal(
            evidence["observation.images.front"],
            episode.observations["images.front"][-1].numpy(),
        )
    metadata = json.loads((shard / "episode.json").read_text())
    assert metadata["identity"]["candidate_id"] == episode.identity.candidate_id
    assert metadata["metadata"] == {"task": "move the cube", "nested": {"actual": True}}
    assert metadata["validation"][0]["status"] == "passed"
    manifest = json.loads((shard.parent / "manifest.json").read_text())
    assert [item["commit_id"] for item in manifest["episodes"]] == [episode.commit_id]


def test_same_commit_is_idempotent_and_new_commits_continue_after_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        calls = []
        original = sink._write_dataset

        def write(*args):
            calls.append(args[0])
            return original(*args)

        monkeypatch.setattr(sink, "_write_dataset", write)
        first = sink.submit(_episode())
        duplicate = sink.submit(_episode(), submission_id=1)
        second = sink.submit(_episode("commit-2"))
        assert first.confirmed and duplicate.confirmed and second.confirmed
        assert first.storage_id == duplicate.storage_id
        assert duplicate.submission_id == 1
        assert len(calls) == 2
        manifest = json.loads((sink.root / "manifest.json").read_text())
        assert len(manifest["episodes"]) == 2


@pytest.mark.parametrize(
    "failure_stage",
    ["_write_dataset", "_write_evidence", "_write_manifest", "_verify_evidence"],
)
def test_failure_receipt_retry_reuses_one_logical_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        original = getattr(sink, failure_stage)

        def fail_after_success(*args):
            original(*args)
            raise OSError("injected persistence failure")

        monkeypatch.setattr(sink, failure_stage, fail_after_success)
        failed = sink.submit(_episode())
        assert not failed.confirmed
        assert "injected persistence failure" in failed.error
        monkeypatch.setattr(sink, failure_stage, original)

        # The first attempt already sealed the shard even when a later barrier
        # failed. Retrying must never create another training episode.
        def reject_rewrite(*args):
            raise AssertionError("A sealed shard must be reused")

        monkeypatch.setattr(sink, "_write_dataset", reject_rewrite)
        accepted = sink.submit(_episode(), submission_id=1)
        assert accepted.confirmed, accepted.error
        assert accepted.storage_id == failed.storage_id
        assert (
            len(json.loads((sink.root / "manifest.json").read_text())["episodes"]) == 1
        )


def test_partial_writer_failure_can_rebuild_same_uncommitted_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    original = LeRobotDataset.add_frame
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:

        def fail_frame(*args):
            raise OSError("frame write failed")

        monkeypatch.setattr(LeRobotDataset, "add_frame", fail_frame)
        failed = sink.submit(_episode())
        assert not failed.confirmed
        assert not (sink.root / "manifest.json").exists()
        monkeypatch.setattr(LeRobotDataset, "add_frame", original)
        accepted = sink.submit(_episode(), submission_id=1)
        assert accepted.confirmed, accepted.error
        assert accepted.storage_id == failed.storage_id


def test_changed_payload_under_same_commit_is_rejected(tmp_path: Path) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        episode = _episode()
        assert sink.submit(episode).confirmed
        changed = replace(episode, actions=episode.actions + 0.1)
        with pytest.raises(ValueError, match="changed episode payload"):
            sink.submit(changed, submission_id=1)
        assert (
            len(json.loads((sink.root / "manifest.json").read_text())["episodes"]) == 1
        )


@pytest.mark.parametrize(
    "observation",
    [
        torch.zeros(_STEPS + 1, 0, 2),
        torch.zeros(_STEPS + 1, 2, dtype=torch.bool),
        torch.zeros(_STEPS + 1, 4, 5, 1, dtype=torch.uint8),
        torch.zeros(_STEPS + 1, 4, 5, 3),
    ],
)
def test_unsupported_observations_fail_before_shard_write(
    tmp_path: Path, observation: torch.Tensor
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        episode = replace(_episode(), observations={"unsupported": observation})
        with pytest.raises(ValueError, match="Unsupported observation"):
            sink.submit(episode)
        assert tuple(sink.root.iterdir()) == (sink.root / ".writer.lock",)


def test_nonuniform_clock_size_and_failed_validation_rejected_before_writes(
    tmp_path: Path,
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        episode = _episode()
        times = episode.timestamps.clone()
        times[-1] += 0.01
        with pytest.raises(ValueError, match="fixed control clock"):
            sink.submit(replace(episode, timestamps=times))
        failed = ValidationResult((ValidationCheck("task_success", "failed"),))
        with pytest.raises(ValueError, match="accepted episodes"):
            sink.submit(replace(episode, validation=failed))
    with LeRobotEpisodeSink(
        tmp_path / "bounded", fps=_FPS, max_episode_bytes=1
    ) as sink:
        with pytest.raises(ValueError, match="max_episode_bytes"):
            sink.submit(_episode())


def test_terminal_corruption_cannot_return_confirmed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        original = sink._write_evidence

        def corrupt(path, *args):
            original(path, *args)
            (path / "terminal.npz").write_bytes(b"invalid")

        monkeypatch.setattr(sink, "_write_evidence", corrupt)
        receipt = sink.submit(_episode())
        assert not receipt.confirmed
        assert not (sink.root / "manifest.json").exists()


def test_sink_rejects_existing_collection_and_closed_submissions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collection"
    sink = LeRobotEpisodeSink(root, fps=_FPS)
    with pytest.raises(ValueError, match="new or empty"):
        LeRobotEpisodeSink(root, fps=_FPS)
    sink.close()
    sink.close()
    with pytest.raises(RuntimeError, match="closed"):
        sink.submit(_episode())


def test_float64_storage_preserves_precision_beyond_torch_reader_defaults(
    tmp_path: Path,
) -> None:
    precise = torch.tensor(
        [[0.123456789123456, 0.987654321987654]], dtype=torch.float64
    )
    episode = replace(
        _episode(),
        observations={"precise": precise.repeat(_STEPS + 1, 1)},
        actions=precise.repeat(_STEPS, 1),
    )
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        receipt = sink.submit(episode)
        assert receipt.confirmed, receipt.error


def test_confirmed_duplicate_only_reads_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LeRobotEpisodeSink(tmp_path / "collection", fps=_FPS) as sink:
        first = sink.submit(_episode())
        assert first.confirmed, first.error

        def no_rewrite(*args):
            raise AssertionError("Confirmed duplicates must not write again")

        monkeypatch.setattr(sink, "_write_dataset", no_rewrite)
        monkeypatch.setattr(sink, "_write_evidence", no_rewrite)
        monkeypatch.setattr(sink, "_write_manifest", no_rewrite)
        duplicate = sink.submit(_episode(), submission_id=1)
        assert duplicate.confirmed, duplicate.error


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numeric_pose_matrices_roundtrip_with_original_shape(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    poses = torch.eye(4, dtype=dtype).repeat(_STEPS + 1, 1, 1)
    poses[:, :3, 3] = torch.arange((_STEPS + 1) * 3, dtype=dtype).reshape(-1, 3) / 7
    episode = replace(_episode(), observations={"object_pose": poses})
    with LeRobotEpisodeSink(tmp_path / "matrix", fps=_FPS) as sink:
        receipt = sink.submit(episode)
        assert receipt.confirmed, receipt.error
    shard = Path(receipt.storage_id)
    metadata = json.loads((shard / "episode.json").read_text())
    assert metadata["observation_shapes"]["object_pose"] == [4, 4]
    assert metadata["features"]["observation.object_pose"]["shape"] == [16]
    dataset = LeRobotDataset(
        repo_id="embodichain/trajectory-generation", root=shard / "dataset"
    )
    assert dataset[0]["observation.object_pose"].shape == (16,)
    with np.load(shard / "terminal.npz", allow_pickle=False) as terminal:
        np.testing.assert_array_equal(
            terminal["observation.object_pose"], poses[-1].numpy()
        )
