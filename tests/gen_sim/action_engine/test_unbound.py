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

from embodichain.gen_sim.action_engine.agent import ActionAgent
import embodichain.gen_sim.action_engine.agent as action_agent_module
from embodichain.gen_sim.action_engine.unbound import validate_unbound_action_plan


def _selector(reference: str) -> dict:
    return {
        "kind": "scene_ref",
        "step_id": "",
        "reference": reference,
        "quantifier": "one",
        "count": 0,
    }


def _candidate() -> dict:
    return {
        "candidate_id": "candidate_01",
        "draft": {
            "task_id": "place_can",
            "instruction": "Place the can on the table.",
            "steps": [
                {
                    "id": "place",
                    "task_type": "E1",
                    "object": _selector("the can"),
                    "target": _selector("the table"),
                    "depends_on": [],
                }
            ],
        },
    }


def test_action_agent_drafts_without_scene_uids() -> None:
    candidate = _candidate()
    original = deepcopy(candidate)

    draft = ActionAgent(registry=object()).draft(candidate)

    assert draft["candidate_id"] == "candidate_01"
    assert draft["steps"][0]["object"]["reference"] == "the can"
    assert "uid" not in str(draft).lower()
    assert candidate == original


def test_unbound_plan_rejects_noncanonical_action_recipe() -> None:
    draft = ActionAgent(registry=object()).draft(_candidate())
    draft["steps"][0]["actions"] = ["UnknownAction"]

    with pytest.raises(ValueError, match="task contract"):
        validate_unbound_action_plan(draft)


def test_action_agent_rejects_missing_atomic_action_during_draft() -> None:
    class Registry:
        def names(self):
            return ()

        def executable_names(self):
            return ()

    with pytest.raises(ValueError, match="AtomicAction is not registered"):
        ActionAgent(registry=Registry()).draft(_candidate())


def test_bind_and_plan_requires_the_exact_unbound_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = ActionAgent(registry=object())
    unbound = agent.draft(_candidate())
    grounded = {
        "selected_candidate_id": "candidate_01",
        "task_draft": deepcopy(_candidate()["draft"]),
    }
    monkeypatch.setattr(
        action_agent_module,
        "_validate_grounded_plan",
        lambda value: deepcopy(value),
    )
    monkeypatch.setattr(agent, "plan", lambda value: {"task": value["task_draft"]})

    graph = agent.bind_and_plan(unbound, grounded)
    assert graph["task"] == grounded["task_draft"]

    altered = deepcopy(unbound)
    altered["instruction"] = "A different instruction."
    with pytest.raises(ValueError, match="does not match"):
        agent.bind_and_plan(altered, grounded)
