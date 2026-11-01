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

"""Observation children must survive standard dataset export with parent ancestry."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import torch
from test_persistence import episode
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.data_pipeline.recording.media import MediaSpool, _digest
from embodichain.lab.expansion import export as module


def test_export_includes_visual_children_without_new_physical_receipts(
    tmp_path, monkeypatch
):
    parent = episode()
    parent = replace(
        parent,
        observations={
            **parent.observations,
            "joint_velocities": torch.zeros(3, 1),
            "joint_efforts": torch.zeros(3, 1),
        },
        metadata={
            "expert_action_spec": {"joint_names": ["joint"]},
            "source_episode_uuids": ["source"],
        },
    )
    streams = []
    for name, value in (("parent", 0), ("child", 42)):
        spool = MediaSpool(
            tmp_path / name,
            max_buffer_bytes=16,
            parent_episode_id=parent.episode_id,
            calibration={},
        )
        for t in parent.timestamps:
            spool.append(
                float(t),
                {"sensor/camera/rgb": np.full((1, 1, 3), value, dtype=np.uint8)},
            )
        streams.append(spool.finish())
    parent = replace(
        parent,
        metadata={
            **parent.metadata,
            "media_manifest": str(streams[0]),
            "media_manifest_sha256": _digest(streams[0]),
        },
    )
    sink = ArtifactEpisodeSink(tmp_path)
    sink.submit(parent, submission_id=0)
    import embodichain.lab.expansion.visual as visual

    child = {
        "parent_commit_id": parent.commit_id,
        "variant_id": "blue",
        "media_manifest": str(streams[1]),
        "media_sha256": _digest(streams[1]),
        "profile": {"materials": {}},
    }
    monkeypatch.setattr(visual, "inspect_observation_variants", lambda _: (child,))
    captured = []

    class Recorder:
        def __init__(self, *args):
            pass

        def _persist_episode_payload(self, row, frames, actions, *, episode_metadata):
            captured.append(
                (
                    episode_metadata,
                    torch.stack([frame["sensor", "camera", "rgb"] for frame in frames]),
                    actions.clone(),
                )
            )

        def finalize(self):
            return tmp_path / "dataset"

    monkeypatch.setattr(module, "LeRobotRecorder", Recorder)
    monkeypatch.setattr(
        module,
        "inspect_recording",
        lambda _: {
            "ok": True,
            "commits": [
                {"episode_uuid": v[0]["episode_uuid"], "status": "complete"}
                for v in captured
            ],
        },
    )
    env = SimpleNamespace(expert_action_spec=SimpleNamespace(joint_names=("joint",)))
    result = module.export_expansion_lerobot(env, tmp_path)
    assert result["episodes"] == 2
    assert len(sink.inspect()) == 1
    assert captured[1][0]["parent_episode_uuid"] == captured[0][0]["episode_uuid"]
    assert captured[1][0]["source_episode_uuids"] == ["source"]
    assert captured[1][0]["observation_origin"] == "state_rerender"
    assert torch.equal(captured[0][2], captured[1][2])
    assert (captured[0][1] == 0).all() and (captured[1][1] == 42).all()
    assert module.export_expansion_lerobot(None, tmp_path) == result
    assert len(captured) == 2
