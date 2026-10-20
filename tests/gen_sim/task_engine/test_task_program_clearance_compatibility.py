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

"""Provider-free generated-recipe compatibility for released-hand recovery."""

from __future__ import annotations

from typing import Any

import pytest

from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine._task_program.clearance_recovery import (
    bind_clearance_calls,
)
from embodichain.lab.sim.atomic_actions import Affordance
from embodichain.lab.task_program import CompiledTaskProgram, TaskProgramCompiler
from embodichain.lab.task_program.language.decoder import decode_task_program
from embodichain.lab.task_program.semantics import (
    Place,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneEntityManifest,
    SceneManifest,
    SceneObjectRef,
)


def _selector(reference: str | None = None) -> dict[str, Any]:
    return {
        "kind": "none" if reference is None else "scene_ref",
        "step_id": "",
        "reference": reference or "",
        "quantifier": "one",
        "count": 0,
    }


def _generated_graph(
    task_type: str, *, preceding_handover: bool = False
) -> dict[str, Any]:
    target = "table" if task_type == "E4" else "tray"
    step = {
        "id": "orient",
        "task_type": task_type,
        "object": _selector("can"),
        "target": _selector(target),
        "relation": "on" if task_type == "E4" else "inside",
        "required_arm": "none" if task_type == "E4" else "right_arm",
        "transfer_arm": "left_arm" if task_type == "E4" else "none",
        "receive_arm": "right_arm" if task_type == "E4" else "none",
        "orientation_goal": "upright",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "place" if task_type == "E4" else "none",
        "depends_on": [],
    }
    steps = [step]
    bindings = {
        "step_01.object": ["can"],
        "step_01.target": [target],
    }
    if preceding_handover:
        handover = {
            **step,
            "id": "handover",
            "task_type": "E4",
            "target": _selector(),
            "relation": "none",
            "required_arm": "none",
            "transfer_arm": "left_arm",
            "receive_arm": "right_arm",
            "orientation_goal": "none",
            "terminal_behavior": "hold",
        }
        steps = [handover, {**step, "depends_on": ["handover"]}]
        bindings = {
            "step_01.object": ["can"],
            "step_02.object": ["can"],
            "step_02.target": [target],
        }
    draft = InstructionDraftResult(
        intent={"steps": steps},
        model="provider_free_fixture",
        attempts=1,
        latency_seconds=0.0,
        normalizations=(),
    )
    candidate = TaskAgent(interpreter=lambda *_args, **_kwargs: draft).generate(
        "upright_compatibility", "test-instruction", candidate_count=1
    )["candidates"][0]
    return SemanticTaskPlanner().plan(
        candidate,
        {
            "schema_version": ROLE_BINDINGS_SCHEMA,
            "task_id": "upright_compatibility",
            "candidate_id": candidate["candidate_id"],
            "reference_bindings": bindings,
            "role_bindings": {},
        },
        [
            {"runtime_uid": "can", "init_pos": [0.0, -0.2, 0.8]},
            {"runtime_uid": "tray", "init_pos": [0.0, 0.0, 0.72]},
            {"runtime_uid": "table", "init_pos": [0.0, 0.0, 0.72]},
        ],
    )


def _compile_calls(
    calls: list[dict[str, Any]], *, targets: dict[str, Any] | None = None
) -> CompiledTaskProgram:
    # The full strict decoder/compiler owns semantic types and invocation identity.
    config = decode_task_program(
        {
            "program_id": "clearance_compatibility",
            "integration": {
                "robot_profile": "fixture",
                "scene_registry": "fixture",
                "runtime_preset": "trajectory_verified",
            },
            "targets": targets or {},
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": f"call_{index}",
                        "steps": {"kind": "invoke", "call": call},
                    }
                    for index, call in enumerate(calls)
                ],
            },
        }
    )
    tray = SceneObjectRef("tray")
    manifest = SceneManifest(
        (
            SceneEntityManifest(ref=SceneObjectRef("can")),
            SceneEntityManifest(ref=SceneObjectRef("table")),
            SceneEntityManifest(ref=tray),
            SceneEntityManifest(
                ref=SceneAffordanceRef("inside__tray__can"),
                parent=tray,
                native_name="inside",
                affordance_payload_type=Affordance,
            ),
        )
    )
    return TaskProgramCompiler(manifest).compile(config)


def _invocation_id(program: CompiledTaskProgram, segment: Any) -> str:
    return (
        f"{program.program_id}/{segment.segment_id}:"
        f"{segment.calls[0].segment_call_index}"
    )


@pytest.mark.parametrize(
    "task_type,preceding_handover",
    [("E4", False), ("E1", False), ("E1", True)],
)
def test_generated_upright_cleanup_binds_its_actual_release_and_park_tail(
    task_type: str, preceding_handover: bool
) -> None:
    graph = _generated_graph(task_type, preceding_handover=preceding_handover)
    raw_calls = [node["call"] for node in graph["nodes"]]
    program = _compile_calls(raw_calls, targets=graph["targets"])
    segments = tuple(program.iter_segments())
    clearance_index = next(
        index
        for index, segment in enumerate(segments)
        if getattr(segment.calls[0].call, "call_id", None) == "gen_sim.clear_released"
    )
    release = segments[clearance_index - 1].calls[0].call
    parks = segments[clearance_index + 1 :]
    if task_type == "E4":
        assert type(release) is RegisteredSemanticCall
        assert release.call_id == "simulation.place_relative"
        assert [dict(segment.calls[0].call.resources) for segment in parks] == [
            {"primary": "left"},
            {"primary": "right"},
        ]
    else:
        assert type(release) is Place
        assert release.object.entity_id == "can"
        assert release.inside.entity_id == "inside__tray__can"
        assert [dict(segment.calls[0].call.resources) for segment in parks] == (
            [{"primary": "right"}, {"primary": "left"}]
            if preceding_handover
            else [{"primary": "right"}]
        )

    routes = bind_clearance_calls(program)

    assert len(routes) == 1
    route = routes[0]
    assert route.release_call == release
    assert route.clearance_invocation_id == _invocation_id(
        program, segments[clearance_index]
    )
    assert route.park_invocation_ids == tuple(
        _invocation_id(program, segment) for segment in parks
    )
    assert route.park_calls == tuple(segment.calls[0].call for segment in parks)


def _registered_call(
    call_id: str, *, resource: str = "right", object_id: str = "can"
) -> dict[str, Any]:
    return {
        "kind": "registered",
        "call_id": call_id,
        "arguments": (
            {"object": object_id, "reference": "table", "relation": "on"}
            if call_id == "simulation.place_relative"
            else (
                {"object": object_id, "target": "staging"}
                if call_id == "gen_sim.clear_released"
                else {}
            )
        ),
        "resources": {"primary": resource},
    }


@pytest.mark.parametrize(
    "failure",
    [
        "non_park",
        "non_park_after_source",
        "three_parks",
        "duplicate_resource",
        "missing_owner",
        "object",
        "arm",
        "curated_object",
        "curated_arm",
    ],
)
def test_cleanup_tail_extension_preserves_exact_release_and_bounded_park_scope(
    failure: str,
) -> None:
    release = _registered_call("simulation.place_relative")
    clear = _registered_call("gen_sim.clear_released")
    park = _registered_call("simulation.park")
    if failure == "non_park":
        tail = [_registered_call("ordinary.move_end_effector"), park]
    elif failure == "non_park_after_source":
        tail = [
            _registered_call("simulation.park", resource="left"),
            _registered_call("ordinary.move_end_effector"),
            park,
        ]
    elif failure == "three_parks":
        tail = [
            _registered_call("simulation.park", resource="left"),
            _registered_call("simulation.park", resource="auxiliary"),
            park,
        ]
    elif failure == "duplicate_resource":
        tail = [park, park]
    elif failure == "missing_owner":
        tail = [
            _registered_call("simulation.park", resource="left"),
            _registered_call("simulation.park", resource="auxiliary"),
        ]
    else:
        tail = [park]
        if failure.startswith("curated_"):
            release = {
                "kind": "place",
                "object": "tray" if failure == "curated_object" else "can",
                "inside": "inside__tray__can",
                "resources": {
                    "primary": "left" if failure == "curated_arm" else "right"
                },
            }
        else:
            release = _registered_call(
                "simulation.place_relative",
                object_id="other" if failure == "object" else "can",
                resource="left" if failure == "arm" else "right",
            )
    program = _compile_calls([release, clear, *tail])

    with pytest.raises(ValueError):
        bind_clearance_calls(program)
