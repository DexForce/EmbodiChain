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

from types import SimpleNamespace as NS

import pytest

from embodichain.gen_sim.task_engine._task_program import drawer_binding as binding
from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
    DrawerObservation,
)
from embodichain.lab.task_program.semantics import (
    Place,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneObjectRef,
)


def _route():
    part = binding.PrismaticBinding(
        "cabinet",
        "slide",
        "drawer",
        "body",
        "/drawer/handle",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (-0.2, 0.0),
        part_id="part_one",
    )
    return binding.DrawerRoute(
        "inside__cabinet__object__part_one",
        "object",
        part,
        (-0.1, -0.1, 0.0),
        (0.1, 0.1, 0.1),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_drawer_placements_share_geometry_not_upright_policy(
    monkeypatch, reverse
):
    route = _route()
    align = {
        "id": "first.align",
        "task_instance_id": "first",
        "call": {
            "call_id": "gen_sim.align_held",
            "arguments": {"object": "object", "target": "current_object_pose"},
        },
    }

    def place(group):
        return {
            "id": f"{group}.place",
            "task_instance_id": group,
            "call": {"kind": "place", "object": "object", "inside": route.affordance},
        }

    def transport(group):
        return {
            "id": f"{group}.transport",
            "task_instance_id": group,
            "call": {
                "kind": "registered",
                "call_id": binding.DRAWER_TRANSPORT_CALL,
                "arguments": {"object": "object", "target": route.affordance},
            },
        }

    first, second = [align, transport("first"), place("first")], [
        transport("second"),
        place("second"),
    ]
    scene = NS(articulations=({"uid": "cabinet"},))
    monkeypatch.setattr(
        binding,
        "graph_bindings",
        lambda graph, scene: {"cabinet::part_one": route.binding},
    )
    measured = []

    def region(*args):
        measured.append(True)
        return route.lower, route.upper

    monkeypatch.setattr(binding, "drawer_region", region)
    result = binding.drawer_routes(
        {"nodes": second + first if reverse else first + second}, scene
    )
    assert len(result) == len(measured) == 1
    assert set(result[0].upright_calls) == {"first.place", "first.transport"}
    assert result[0].upright is False
    assert binding.DrawerRoute.decode(result[0].payload()) == result[0]


def test_drawer_call_binding_and_recovery_keep_policy_local():
    route = _route()
    from dataclasses import replace

    route = replace(route, upright_calls=("first.place", "first.transport"))
    calls = {
        "first.place": Place(
            object=SceneObjectRef("object"), inside=SceneAffordanceRef(route.affordance)
        ),
        "first.transport": RegisteredSemanticCall(
            call_id=binding.DRAWER_TRANSPORT_CALL,
            arguments={"object": "object", "target": route.affordance},
        ),
        "second.place": Place(
            object=SceneObjectRef("object"), inside=SceneAffordanceRef(route.affordance)
        ),
    }
    program = NS(
        program_id="program",
        iter_segments=lambda: iter(
            NS(
                name=name,
                segment_id=f"segment{index}",
                calls=(NS(segment_call_index=0, call=call),),
            )
            for index, (name, call) in enumerate(calls.items())
        ),
    )
    scoped = binding.bind_drawer_routes((route,), program)[0]
    assert scoped.upright_calls == ("program/segment0:0", "program/segment1:0")
    observation = object.__new__(DrawerObservation)
    observation.route = scoped
    observation.vertices = object()
    observation.other_payloads = ()
    upright = observation.for_invocation("program/segment0:0")
    recovered = observation.for_invocation("program/segment0:workflow_recovery:1:0")
    ordinary = observation.for_invocation("program/segment2:0")
    assert upright.route.upright and recovered.route.upright
    assert ordinary.route.upright is False
    assert (
        observation.for_invocation(
            "program/segment2:workflow_recovery:1:0"
        ).route.upright
        is False
    )
    assert (
        observation.for_invocation(
            "program/segment01:workflow_recovery:1:0"
        ).route.upright
        is False
    )
    assert observation.route.upright is False
    assert upright.vertices is ordinary.vertices is observation.vertices
    assert binding.bind_drawer_routes((_route(),), program) == (_route(),)
    with pytest.raises(ValueError):
        binding.bind_drawer_routes(
            (replace(route, upright_calls=("missing",)),), program
        )
    with pytest.raises(ValueError):
        binding.bind_drawer_routes(
            (
                replace(
                    route,
                    affordance="inside__cabinet__other__part_one",
                    object_id="other",
                ),
            ),
            program,
        )


@pytest.mark.parametrize("value", [True, "first", ["a", "a"], [""], [3]])
def test_drawer_call_payload_rejects_invalid_scope(value):
    with pytest.raises(ValueError):
        binding.DrawerRoute.decode({**_route().payload(), "upright_calls": value})
