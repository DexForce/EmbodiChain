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

"""Generated release offsets agree with their bounded runtime consumers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program import place_yaw
from embodichain.gen_sim.task_engine._task_program.measured_rigid import (
    MeasuredRigidPolicy,
)
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.task_program.integrations.simulation.bindings import (
    ContainerAffordanceBinding,
)
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    SceneAffordanceRef,
    SceneObjectRef,
)


def _binding(clearance: float) -> SimpleNamespace:
    return SimpleNamespace(
        rigid_objects=[SimpleNamespace(entity_id=name) for name in ("item", "tray")],
        containers=[
            ContainerAffordanceBinding(
                entity_id="landing",
                parent_id="tray",
                native_name="landing",
                release_clearance=clearance,
            )
        ],
    )


@pytest.mark.parametrize(
    "clearance, accepted", [(0.12, True), (0.05, False), (0.08, False)]
)
def test_world_yaw_accepts_only_supported_release_offsets(
    monkeypatch: pytest.MonkeyPatch, clearance: float, accepted: bool
) -> None:
    subject = SceneObjectRef("item")
    route = SimpleNamespace(
        pick_id="pick",
        place_id="place",
        pick_call=Pick(object=subject, resources={"primary": "left"}),
        place_call=Place(
            object=subject,
            inside=SceneAffordanceRef("landing"),
            resources={"primary": "left"},
        ),
    )
    factory = SimpleNamespace(
        _world_yaw_free_places=("place",),
        _pick_purposes={"pick": "ordinary"},
        _pick_continuations=(route,),
        _simulation=object(),
        task_program_registration=SimpleNamespace(scene_binding=_binding(clearance)),
    )
    geometry = object()
    monkeypatch.setattr(
        place_yaw, "_prepare_placement_geometry", lambda *args: geometry
    )
    if not accepted:
        with pytest.raises(ValueError, match="bound ordinary rigid inside route"):
            place_yaw.bind_routes(factory)
        return
    bound = place_yaw.bind_routes(factory)["place"]
    assert bound[1].release_clearance == clearance
    assert bound[-1] is geometry


@pytest.mark.parametrize(
    "clearance, accepted", [(0.12, True), (0.05, False), (0.08, False)]
)
def test_measured_place_preserves_supported_release_offsets(
    monkeypatch: pytest.MonkeyPatch, clearance: float, accepted: bool
) -> None:
    policy = MeasuredRigidPolicy(
        object(), _binding(clearance), (("place", "item", "landing"),)
    )
    request = SimpleNamespace(
        invocation_id="place",
        goal=SimpleNamespace(xpos=SceneEntityPose("landing")),
        skill_options=SimpleNamespace(max_approach_retract_z=None),
        binding=SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(task_state_key="left")
        ),
    )
    context = SimpleNamespace(
        get_held_object=lambda key: SimpleNamespace(semantics=SceneObjectRef("item"))
    )

    class LandingReached(Exception):
        pass

    def stop_after_contract_check(*args):
        assert args[-1].release_clearance == clearance
        raise LandingReached

    monkeypatch.setattr(policy, "_landing", stop_after_contract_check)
    error = LandingReached if accepted else ValueError
    with pytest.raises(error):
        policy._place(None, request, context, None)
