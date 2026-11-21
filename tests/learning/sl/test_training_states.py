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

"""Commit and restore explicit update participants independently of objectives."""

from __future__ import annotations
import copy
import torch
import pytest
from .test_trainer import build


class UpdateAverage:
    """Independent running-weight average exercising trainer state recovery."""

    def __init__(self):
        self.count = 0
        self.total = torch.zeros(())

    def on_update(self, learner):
        self.count += 1
        self.total += next(learner.model.parameters()).detach().cpu().sum()

    def state_dict(self):
        return {"count": self.count, "total": self.total.clone()}

    def validate_state_dict(self, state):
        if (
            type(state["count"]) is not int
            or state["count"] < 0
            or state["total"].shape != ()
        ):
            raise ValueError("Invalid average state")

    def load_state_dict(self, state):
        self.count = state["count"]
        self.total.copy_(state["total"])


def test_update_participant_restores_exact_committed_average(tmp_path):
    full = build(tmp_path / "state-full")
    full.training_states["average"] = UpdateAverage()
    full.fit()
    split = build(tmp_path / "state-split")
    split.training_states["average"] = UpdateAverage()
    split.fit(stop_after=7)
    saved = torch.load(tmp_path / "state-split/last.pt", weights_only=False)
    assert saved["version"] == 3
    resumed = build(tmp_path / "state-resume")
    resumed.training_states["average"] = UpdateAverage()
    resumed.resume(tmp_path / "state-split/last.pt")
    assert resumed.training_states["average"].count == 7
    resumed.fit()
    assert resumed.training_states["average"].count == 20
    torch.testing.assert_close(
        resumed.training_states["average"].total,
        full.training_states["average"].total,
        atol=0,
        rtol=0,
    )
    for key, value in full.learner.model.state_dict().items():
        torch.testing.assert_close(
            resumed.learner.model.state_dict()[key], value, atol=0, rtol=0
        )


def test_participant_layout_rejected_before_model_mutation(tmp_path):
    source = build(tmp_path / "participant-source")
    source.training_states["average"] = UpdateAverage()
    source.fit(stop_after=2)
    target = build(tmp_path / "participant-target")
    target.training_states["another_average"] = UpdateAverage()
    before = copy.deepcopy(target.learner.model.state_dict())
    with pytest.raises(ValueError, match="training state names"):
        target.resume(tmp_path / "participant-source/last.pt")
    assert target.learner.num_updates == 0
    for key, value in before.items():
        torch.testing.assert_close(
            target.learner.model.state_dict()[key], value, atol=0, rtol=0
        )


def test_default_recipe_resumes_checkpoint_without_new_optional_fields(tmp_path):
    source = build(tmp_path / "legacy-source")
    source.fit(stop_after=2)
    state = torch.load(tmp_path / "legacy-source/last.pt", weights_only=False)
    assert state["version"] == 1
    state.pop("training_states")
    state.pop("ddp_broadcast_buffers")
    path = tmp_path / "legacy.pt"
    torch.save(state, path)
    target = build(tmp_path / "legacy-target")
    target.resume(path)
    assert target.learner.num_updates == 2
    for key, value in source.learner.model.state_dict().items():
        torch.testing.assert_close(
            target.learner.model.state_dict()[key], value, atol=0, rtol=0
        )
