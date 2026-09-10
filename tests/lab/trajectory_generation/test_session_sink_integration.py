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

"""Session accounting driven by actual sealed LeRobot persistence receipts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateTrajectoryBatch,
    ExpertEpisode,
    GenerationSession,
    SceneCase,
    TrajectoryGenerationJobCfg,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink


def test_failed_real_commit_retry_preserves_rollout_count_and_confirms_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Synthetic rollout evidence isolates receipt/accounting from physical quality.
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {"collection": {"target_committed_episodes": 1}}
    )
    session = GenerationSession(cfg)
    case = SceneCase("case", "initial", "scene", "move", "robot")
    session.register_case(
        case, torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]), joint_names=("a", "b")
    )
    identity, _ = session.propose(
        "case",
        "initial",
        source_id="test_source",
        source_revision="v1",
        template_id="reference",
        operator_id="replay",
    )
    positions = torch.tensor([[[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]]])
    dt = torch.tensor([[0.0, 0.05, 0.05]], dtype=torch.float64)
    plan = CandidateTrajectoryBatch(
        positions, dt, torch.tensor([3]), (identity,), ("a", "b")
    )
    session.add_planned(
        plan, ValidationResult((ValidationCheck("path_collision", "passed"),))
    )
    assert session.take_ready("case", "initial", episode_byte_budget=4096) is not None
    session.mark_rollout_started(identity)
    episode_id, commit_id = session.episode_ids(identity)
    episode = ExpertEpisode(
        identity,
        {"joint_positions": positions[0]},
        positions[0, 1:],
        dt[0].cumsum(0),
        "qpos",
        ValidationResult(
            (
                ValidationCheck("path_collision", "passed"),
                ValidationCheck("task_success", "passed"),
            )
        ),
        episode_id,
        commit_id,
    )
    assert session.accept_episode(episode)
    with LeRobotEpisodeSink(tmp_path / "episodes", fps=20) as sink:
        write_manifest = sink._write_manifest

        def fail_manifest(record):
            raise OSError("manifest not written")

        monkeypatch.setattr(sink, "_write_manifest", fail_manifest)
        failed = sink.submit(episode)
        assert not failed.confirmed
        session.apply_receipt(failed)
        assert session.snapshot()["counts"]["committed"] == 0
        assert session.snapshot()["coverage"]["case"] == 0
        retry_episode, submission = session.retry_write(commit_id)
        monkeypatch.setattr(sink, "_write_manifest", write_manifest)
        confirmed = sink.submit(retry_episode, submission_id=submission)
        assert confirmed.confirmed, confirmed.error
        session.apply_receipt(confirmed)
        session.apply_receipt(confirmed)
        report = session.snapshot()
        assert report["counts"]["committed"] == 1
        assert report["counts"]["rollout_attempted"] == 1
        assert report["coverage"]["case"] == 1
        assert report["pending_reserved_bytes"] == 0
        manifest = json.loads((sink.root / "manifest.json").read_text())
        assert len(manifest["episodes"]) == 1
        assert manifest["episodes"][0]["commit_id"] == episode.commit_id
