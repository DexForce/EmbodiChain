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

"""An intended landing contact is deferred, never reported as collision proof."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    pick_continuation as continuation,
    place_recovery as recovery,
)
from embodichain.lab.sim.atomic_actions import PlannerDiagnostics


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize("bound", [False, True])
def test_only_registered_ordinary_on_forecasts_are_deferred(monkeypatch, bound):
    action = NS(robot=NS(device=torch.device("cpu")))
    context = NS(robot=NS(qpos=torch.zeros(1, 2)))
    request = NS(invocation_id="acquire")
    plan = Plan(torch.tensor([True]), PlannerDiagnostics("test"))
    engine = NS(
        actions={"place": NS(ordinary_on_invocations={"release"} if bound else set())}
    )
    preview = NS(
        route=NS(pick_id="acquire", place_id="release"),
        compiler=NS(integration=NS(engine=engine)),
    )
    monkeypatch.setattr(
        continuation,
        "_validate_final_plan_velocity",
        lambda *a: {"valid_mask": torch.tensor([True])},
    )
    predicted = Mock(return_value={"valid": True})
    monkeypatch.setattr(continuation, "preview_place", predicted)
    result = continuation.recover(action, request, context, plan, NS(), preview)
    assert bool(result.plan_success.all())
    if bound:
        predicted.assert_not_called()
        receipt = result.diagnostics.metadata["gen_sim_pick_continuation_deferred"]
        assert receipt["place_invocation_id"] == "release"
        assert receipt["physical_evidence"] is False
    else:
        predicted.assert_called_once()


@pytest.mark.parametrize(
    "damage", [None, "unbound", "obstacle", "velocity", "joint_limit", "robot"]
)
def test_only_the_intended_support_owner_is_deferred(monkeypatch, damage):
    points = torch.tensor(
        [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]
    )[None]
    positions = torch.zeros(1, 2, 2)
    if damage == "joint_limit":
        positions[0, 1, 0] = 2
    robot = NS(
        get_qpos_limits=lambda **kw: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
        get_solver=lambda _: None,
        compute_batch_fk=lambda **kw: torch.eye(4).repeat(1, 2, 1, 1),
    )
    action = NS(
        robot=robot,
        ordinary_on_invocations=set() if damage == "unbound" else {"release"},
        support_surface_owners={"landing_frame": "support"},
        recovery_scene_vertices={"item": points, "table": points, "support": points},
    )
    endpoint = NS(
        task_state_key="hand",
        require_target=lambda _: NS(control_part="arm", joint_ids=(0, 1)),
    )
    request = NS(
        invocation_id="release",
        goal=NS(xpos=NS(entity_id="landing_frame")),
        binding=NS(endpoint=lambda *a: endpoint),
    )
    context = NS(
        last_qpos=torch.zeros(1, 2),
        get_held_object=lambda _: NS(
            semantics=NS(entity_id="item"), object_to_eef=torch.eye(4)[None]
        ),
    )
    plan = NS(
        plan_success=torch.tensor([True]),
        joint_trajectory=NS(positions=positions),
        segments=[NS(name="approach", stop=2)],
    )
    monkeypatch.setattr(
        recovery,
        "_validate_final_plan_velocity",
        lambda *a: {"valid_mask": torch.tensor([damage != "velocity"])},
    )
    monkeypatch.setattr(
        recovery, "_inspect_pick_motion", lambda *a: {"valid": damage != "robot"}
    )
    monkeypatch.setattr(
        recovery, "_pose_validation_parameters", lambda _: {"rot_eps": 0.05}
    )
    table_pose = torch.eye(4)[None]
    table_pose[:, 2, 3] = -10
    components = [
        ("support", "support", points, torch.eye(4)[None]),
        ("table", "table", points, table_pose),
    ]
    if damage == "obstacle":
        components.append(("obstacle", "obstacle", points, torch.eye(4)[None]))
    monkeypatch.setattr(recovery, "scene_components", lambda *a, **kw: iter(components))
    blocked = Mock(return_value={"valid": False, "minimum_distance_m": 0.0, "frame": 0})
    monkeypatch.setattr(recovery, "_convex_clearance", blocked)
    result = recovery.inspect(action, request, context, plan)
    assert result["valid"] is (damage is None)
    if damage is None:
        blocked.assert_not_called()
        assert result["intended_support_contact_deferred"] == ["support"]
        assert result["physical_evidence"] is False
    elif damage == "obstacle":
        assert result["entity"] == "obstacle"
    elif damage == "unbound":
        assert result["entity"] == "support"
