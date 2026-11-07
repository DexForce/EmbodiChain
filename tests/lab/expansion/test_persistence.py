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

"""Durable local commit identity, staging and media consistency checks."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import hashlib
import numpy as np
import pytest
import torch
from embodichain.lab.expansion.contracts import (
    CandidateIdentity,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.data_pipeline.recording.media import MediaSpool


def episode():
    return ExpertEpisode(
        CandidateIdentity(
            "scene", "initial", "candidate", "family", "source", "v1", "template"
        ),
        {"joint_positions": torch.tensor([[0.0], [0.1], [0.2]])},
        torch.tensor([[0.1], [0.2]]),
        torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
        "qpos",
        ValidationResult((ValidationCheck("task_success", "passed"),)),
        "episode",
        "commit",
    )


def test_staging_has_no_credit_and_recovery_never_reruns_physics(tmp_path):
    sink = ArtifactEpisodeSink(tmp_path)
    data = episode()
    sink.stage(data)
    assert sink.inspect() == ()
    recovered = sink.recover(data.commit_id)
    assert recovered.confirmed
    assert sink.submit(data, submission_id=0) == recovered
    assert len(sink.inspect()) == 1
    with pytest.raises(ValueError, match="different episode"):
        sink.submit(replace(data, actions=data.actions + 1), submission_id=0)
    assert len(sink.inspect()) == 1


def test_media_alignment_is_required_before_receipt(tmp_path):
    sink = ArtifactEpisodeSink(tmp_path)
    data = episode()
    spool = MediaSpool(
        tmp_path / "media",
        max_buffer_bytes=16,
        parent_episode_id=data.episode_id,
        calibration={},
    )
    for t in (0.0, 0.1, 0.3):
        spool.append(t, {"depth": np.zeros((1, 1), dtype=np.float32)})
    data = replace(
        data,
        metadata={
            "media_manifest": str(spool.finish()),
            "media_manifest_sha256": hashlib.sha256(
                spool.finish().read_bytes()
            ).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match="physical observations"):
        sink.submit(data, submission_id=0)
    assert sink.inspect() == ()


def test_changed_media_manifest_cannot_pass_inspection(tmp_path):
    sink = ArtifactEpisodeSink(tmp_path)
    data = episode()
    spool = MediaSpool(
        tmp_path / "media",
        max_buffer_bytes=16,
        parent_episode_id=data.episode_id,
        calibration={},
    )
    for t in (0.0, 0.1, 0.2):
        spool.append(t, {"depth": np.zeros((1, 1), dtype=np.float32)})
    path = spool.finish()
    sink.submit(
        replace(
            data,
            metadata={
                "media_manifest": str(path),
                "media_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        ),
        submission_id=0,
    )
    from embodichain.lab.expansion.persistence import load_expansion_episode

    path = Path(
        load_expansion_episode(sink.inspect()[0].storage_id).metadata["media_manifest"]
    )
    manifest = json.loads(path.read_text())
    manifest["calibration"] = {"changed": True}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="(manifest|artifact content) changed"):
        sink.inspect()


def test_archived_job_sources_survive_input_removal_and_detect_corruption(tmp_path):
    source = tmp_path / "program.yaml"
    source.write_text("program_id: pinned")
    data = replace(
        episode(),
        metadata={
            "expansion_source_artifacts": {
                "program": {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            }
        },
    )
    sink = ArtifactEpisodeSink(tmp_path / "output")
    sink.submit(data, submission_id=0)
    source.unlink()
    assert len(sink.inspect()) == 1
    copied = next((tmp_path / "output").glob("episodes/*/job_sources/*.yaml"))
    copied.write_text("program_id: changed")
    with pytest.raises(ValueError, match="(attachment|artifact content) changed"):
        sink.inspect()


def test_frozen_media_and_sources_survive_bundle_relocation(tmp_path):
    import shutil
    from embodichain.lab.expansion.persistence import load_expansion_episode
    from embodichain.lab.expansion.artifacts import EpisodeArtifactManifest

    root = tmp_path / "job"
    source = tmp_path / "source.json"
    source.write_text('{"version":1}')
    spool = MediaSpool(
        tmp_path / "frames",
        max_buffer_bytes=16,
        parent_episode_id="episode",
        calibration={"camera": {"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
    )
    for stamp in (0.0, 0.1, 0.2):
        spool.append(stamp, {"sensor/camera/depth": np.ones((1, 1), np.float32)})
    media = spool.finish()
    data = replace(
        episode(),
        metadata={
            "media_manifest": str(media),
            "media_manifest_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "expansion_source_artifacts": {
                "program": {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            },
        },
    )
    sink = ArtifactEpisodeSink(root)
    sink.stage(data)
    # Crash after the numerical link but before content/receipt publication.
    next(root.glob("episodes/*/content.json")).unlink()
    moved = tmp_path / "moved"
    root.rename(moved)
    shutil.rmtree(tmp_path / "frames")
    source.unlink()
    repaired = ArtifactEpisodeSink(moved).recover(data.commit_id)
    assert repaired.confirmed
    assert len(ArtifactEpisodeSink(moved).inspect()) == 1
    frozen = load_expansion_episode(repaired.storage_id)
    torch.testing.assert_close(frozen.actions, data.actions)
    assert Path(frozen.metadata["media_manifest"]).is_relative_to(moved)
    EpisodeArtifactManifest.read(
        Path(repaired.storage_id).parent / "artifacts.json"
    ).verify(Path(repaired.storage_id).parent)
