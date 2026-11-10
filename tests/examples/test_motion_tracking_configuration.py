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

from __future__ import annotations

from copy import deepcopy

import pytest

from examples.learning.motion_tracking._configuration import (
    normalize_recipe,
    recipe_identity,
    validate_recipe,
)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("algorithm", "APG_ACTOR_CRITIC"),
        ("policy", "MOTION_TRACKING"),
        ("learning_env", "motiontrackingmodelrl"),
    ],
)
def test_case_insensitive_recipe_validation_preserves_checkpoint_identity(owner, name):
    config = {
        "trainer": {"learning_env": {"name": "MotionTrackingModelRL", "cfg": {}}},
        "policy": {"name": "motion_tracking", "cfg": {}},
        "algorithm": {"name": "apg_actor_critic", "cfg": {}},
        "workflow": {"iterations": 10},
    }
    block = config["trainer"][owner] if owner == "learning_env" else config[owner]
    block["name"] = name
    original = deepcopy(config)
    saved_identity = recipe_identity(config)
    resumed = normalize_recipe(config)
    resumed["workflow"]["iterations"] = 20
    validate_recipe(resumed)
    assert config == original
    original["workflow"]["iterations"] = 20
    assert resumed == original
    assert recipe_identity(resumed) == saved_identity
    assert saved_identity["algorithm"]["name"] == config["algorithm"]["name"]


@pytest.mark.parametrize("name", ["ppo", None, 1])
def test_recipe_validation_rejects_unsupported_algorithm_names(name):
    config = {
        "trainer": {"learning_env": {"name": "MotionTrackingModelRL"}},
        "policy": {"name": "motion_tracking"},
        "algorithm": {"name": name},
        "workflow": {},
    }
    with pytest.raises(ValueError, match="requires algorithm.name='apg_actor_critic'"):
        validate_recipe(config)


@pytest.mark.parametrize("saved_digest", ["motion-v1:old", None])
def test_motion_checkpoint_mismatch_precedes_learning_state_restore(
    tmp_path, monkeypatch, saved_digest
):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch

    example = Path(__file__).resolve().parents[2] / "examples/learning/motion_tracking"
    monkeypatch.syspath_prepend(str(example))
    from examples.learning.motion_tracking.train import MotionTraining

    config = {
        "trainer": {"learning_env": {"name": "MotionTrackingModelRL", "cfg": {}}},
        "policy": {"name": "motion_tracking", "cfg": {}},
        "algorithm": {"name": "apg_actor_critic", "cfg": {}},
        "workflow": {},
    }
    workflow = object.__new__(MotionTraining)
    workflow.env = SimpleNamespace(device=torch.device("cpu"), start_sampler=Mock())
    workflow.config = config
    workflow.identity = {"motion": {"content_digest": "motion-v1:current"}}
    workflow.schedule = Mock()
    workflow.policy, workflow.learner, workflow.algorithm = Mock(), Mock(), Mock()
    saved_motion = {} if saved_digest is None else {"content_digest": saved_digest}
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"schema": 1, "config": config, "identity": {"motion": saved_motion}}, path
    )
    with pytest.raises(ValueError, match="task identity differs"):
        workflow.restore(path)
    for owner in (
        workflow.policy,
        workflow.learner,
        workflow.algorithm,
        workflow.schedule,
        workflow.env.start_sampler,
    ):
        assert owner.mock_calls == []
