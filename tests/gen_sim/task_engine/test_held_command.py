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

from importlib import import_module
from types import SimpleNamespace

import pytest

from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    EffectMonitorRef,
    SkillPolicyPreset,
)
from embodichain.lab.task_program.semantics.calls import (
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
    SemanticPose,
)
from embodichain.lab.task_program.semantics.scene import SceneObjectRef

TRANSPORT = "simulation.move_held_object"
ALIGN = "gen_sim.align_held"
ITEM = "vessel"
RESOURCE = "left"
PRESET = "verified"
RIGIDS = frozenset({ITEM, "other"})
Entry = tuple[str, str, str, str]


def _module():
    return import_module("embodichain.gen_sim.task_engine._task_program.held_command")


def _entries(predecessor: str = "pick") -> list[Entry]:
    return [
        ("acquire", predecessor, ITEM, RESOURCE),
        ("transport", TRANSPORT, ITEM, RESOURCE),
    ]


def _graph(entries: list[Entry]) -> dict:
    nodes = []
    for name, kind, entity, resource in entries:
        call = {"resources": {"primary": resource}}
        if kind in {"pick", "place"}:
            call.update(kind=kind, object=entity)
        else:
            call.update(kind="registered", call_id=kind, arguments={"object": entity})
        nodes.append(
            {
                "id": name,
                "call": call,
                "depends_on": [nodes[-1]["id"]] if nodes else [],
                "task_instance_id": "recipe",
            }
        )
    return {"nodes": nodes}


def _call(kind: str, entity: str, resource: str) -> SemanticCallSpec:
    resources = {"primary": resource}
    if kind == "pick":
        return Pick(object=SceneObjectRef(entity), resources=resources)
    if kind == "place":
        return Place(
            object=SceneObjectRef(entity),
            at=SemanticPose((0, 0, 1), (0, 0, 0, 1)),
            resources=resources,
        )
    return RegisteredSemanticCall(
        call_id=kind, arguments={"object": entity}, resources=resources
    )


def _program(entries: list[Entry]) -> SimpleNamespace:
    segments = [
        SimpleNamespace(
            name=name,
            segment_id="segment-" + name,
            calls=(
                SimpleNamespace(
                    segment_call_index=0, call=_call(kind, entity, resource)
                ),
            ),
            parallel_block=None,
            repeat_frames=(),
        )
        for name, kind, entity, resource in entries
    ]
    return SimpleNamespace(
        program_id="compiled", iter_segments=lambda: iter(segments), segments=segments
    )


def _profile(
    predecessor: str = "pick",
    *,
    assurance: EffectAssurance = EffectAssurance.VERIFIED,
    missing: str | None = None,
    invalid: str | None = None,
    revision: str = "1",
) -> SimpleNamespace:
    monitor = EffectMonitorRef("builtin.composite_effect", "1")
    monitors = {predecessor: monitor, TRANSPORT: monitor}
    if missing is not None:
        monitors.pop(missing, None)
    if invalid is not None:
        monitors[invalid] = EffectMonitorRef(
            "custom.monitor" if revision == "1" else "builtin.composite_effect",
            revision,
        )
    if assurance is EffectAssurance.PROJECTED:
        monitors = {}
    preset = SkillPolicyPreset(
        preset_id=PRESET,
        action_option_templates={},
        effect_assurance=assurance,
        effect_monitors=monitors,
    )
    return SimpleNamespace(presets={PRESET: preset})


@pytest.mark.parametrize(
    "predecessor",
    ["pick", "gen_sim.pick.group", "gen_sim.pick.handover_source", ALIGN, TRANSPORT],
)
def test_declarations_select_only_immediate_held_predecessor(predecessor):
    assert _module().held_command_declarations(_graph(_entries(predecessor))) == {
        "transport": "acquire"
    }


@pytest.mark.parametrize(
    "predecessor",
    [
        "place",
        "simulation.place_relative",
        "simulation.pour",
        "hand_over",
        "simulation.pick",
    ],
)
def test_declarations_do_not_grant_from_release_pour_or_handover(predecessor):
    assert _module().held_command_declarations(_graph(_entries(predecessor))) == {}


@pytest.mark.parametrize("changed", ["object", "resource"])
def test_declarations_do_not_cross_object_or_resource(changed):
    entries = _entries()
    entries[1] = (
        "transport",
        TRANSPORT,
        "other" if changed == "object" else ITEM,
        "right" if changed == "resource" else RESOURCE,
    )
    assert _module().held_command_declarations(_graph(entries)) == {}


def test_declarations_require_exact_registered_transport_consumer():
    graph = _graph(_entries())
    graph["nodes"][1]["call"] = {
        "kind": "move_held_object",
        "object": ITEM,
        "resources": {"primary": RESOURCE},
    }
    assert _module().held_command_declarations(graph) == {}


def test_declarations_track_repeated_object_by_call_occurrence():
    entries = _entries() + [("transport_again", TRANSPORT, ITEM, RESOURCE)]
    assert _module().held_command_declarations(_graph(entries)) == {
        "transport": "acquire",
        "transport_again": "transport",
    }


@pytest.mark.parametrize("barrier", ["place", "simulation.pour", "hand_over"])
def test_declarations_do_not_search_back_across_a_barrier(barrier):
    entries = _entries()
    entries.insert(1, ("barrier", barrier, ITEM, RESOURCE))
    assert _module().held_command_declarations(_graph(entries)) == {}


def test_declarations_preserve_prefix_when_suffix_changes():
    entries = _entries()
    expected = _module().held_command_declarations(_graph(entries))
    suffixes = [
        [("pour", "simulation.pour", ITEM, RESOURCE)],
        [("release", "place", ITEM, RESOURCE)],
        [
            ("other_pick", "pick", "other", "right"),
            ("other_move", TRANSPORT, "other", "right"),
        ],
    ]
    for suffix in suffixes:
        actual = _module().held_command_declarations(_graph(entries + suffix))
        assert {name: actual[name] for name in expected} == expected


@pytest.mark.parametrize("dependencies", [[], ["other"], ["acquire", "other"]])
def test_declarations_require_direct_predecessor_dependency(dependencies):
    graph = _graph(_entries())
    graph["nodes"][1]["depends_on"] = dependencies
    assert _module().held_command_declarations(graph) == {}


@pytest.mark.parametrize(
    "predecessor", ["pick", "gen_sim.pick.group", ALIGN, TRANSPORT]
)
def test_binding_uses_canonical_call_ids_and_verified_transport_monitor(predecessor):
    actual = _module().bind_held_command_sources(
        {"transport": "acquire"},
        _program(_entries(predecessor)),
        _profile(predecessor),
        PRESET,
        RIGIDS,
    )
    assert set(actual) == {"compiled/segment-transport:0"}
    proof = actual["compiled/segment-transport:0"]
    assert proof["predecessor_invocation_id"] == "compiled/segment-acquire:0"
    assert proof["predecessor_call_id"] == predecessor
    assert proof["entity_id"] == ITEM
    assert proof["resource_id"] == RESOURCE
    assert proof["preset_id"] == PRESET
    assert proof["attachment_monitor"] == {
        "monitor_id": "builtin.composite_effect",
        "revision": "1",
    }
    assert proof["retention_monitor"] == proof["attachment_monitor"]


def test_empty_binding_does_not_infer_undeclared_authority():
    assert (
        _module().bind_held_command_sources(
            {}, _program(_entries()), _profile(), PRESET, RIGIDS
        )
        == {}
    )


def test_binding_preserves_prefix_proof_when_suffix_changes():
    declarations = {"transport": "acquire"}
    expected = _module().bind_held_command_sources(
        declarations, _program(_entries()), _profile(), PRESET, RIGIDS
    )
    suffix = [
        ("other_pick", "pick", "other", "right"),
        ("other_move", TRANSPORT, "other", "right"),
    ]
    assert (
        _module().bind_held_command_sources(
            declarations, _program(_entries() + suffix), _profile(), PRESET, RIGIDS
        )
        == expected
    )


@pytest.mark.parametrize(
    "declarations",
    [
        None,
        [],
        (),
        True,
        {1: "acquire"},
        {"transport": 1},
        {"": "acquire"},
        {"transport": ""},
        {" transport": "acquire"},
        {"transport": "acquire "},
        {"unknown": "acquire"},
        {"transport": "unknown"},
    ],
)
def test_binding_rejects_malformed_or_missing_segment_names(declarations):
    with pytest.raises(ValueError):
        _module().bind_held_command_sources(
            declarations, _program(_entries()), _profile(), PRESET, RIGIDS
        )


@pytest.mark.parametrize("barrier", ["place", "simulation.pour", "hand_over"])
def test_binding_revalidates_adjacency_instead_of_trusting_declarations(barrier):
    entries = _entries()
    entries.insert(1, ("barrier", barrier, ITEM, RESOURCE))
    with pytest.raises(ValueError):
        _module().bind_held_command_sources(
            {"transport": "acquire"}, _program(entries), _profile(), PRESET, RIGIDS
        )


@pytest.mark.parametrize(
    "mismatch", ["object", "resource", "consumer", "source", "nonrigid"]
)
def test_binding_revalidates_call_and_resource_identity(mismatch):
    entries = _entries()
    rigid_ids = RIGIDS
    if mismatch == "nonrigid":
        rigid_ids = frozenset({"other"})
    elif mismatch == "source":
        entries[0] = ("acquire", "simulation.pour", ITEM, RESOURCE)
    else:
        entries[1] = (
            "transport",
            ALIGN if mismatch == "consumer" else TRANSPORT,
            "other" if mismatch == "object" else ITEM,
            "right" if mismatch == "resource" else RESOURCE,
        )
    with pytest.raises(ValueError):
        _module().bind_held_command_sources(
            {"transport": "acquire"}, _program(entries), _profile(), PRESET, rigid_ids
        )


@pytest.mark.parametrize("duplicate", ["name", "canonical_id"])
def test_binding_rejects_duplicate_compiled_identity(duplicate):
    program = _program(_entries())
    if duplicate == "name":
        program.segments[1].name = program.segments[0].name
    else:
        program.segments[1].segment_id = program.segments[0].segment_id
    with pytest.raises(ValueError):
        _module().bind_held_command_sources(
            {"transport": "acquire"}, program, _profile(), PRESET, RIGIDS
        )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("scope", ["parallel", "repeat"])
def test_binding_rejects_parallel_and_repeated_occurrences(index, scope):
    program = _program(_entries())
    if scope == "parallel":
        program.segments[index].parallel_block = SimpleNamespace(block_id="parallel")
    else:
        program.segments[index].repeat_frames = (SimpleNamespace(repeat_id="repeat"),)
    with pytest.raises(ValueError):
        _module().bind_held_command_sources(
            {"transport": "acquire"}, program, _profile(), PRESET, RIGIDS
        )


@pytest.mark.parametrize("missing", ["pick", TRANSPORT])
def test_binding_does_not_grant_without_both_effect_monitors(missing):
    assert (
        _module().bind_held_command_sources(
            {"transport": "acquire"},
            _program(_entries()),
            _profile(missing=missing),
            PRESET,
            RIGIDS,
        )
        == {}
    )


def test_alignment_monitor_cannot_authorize_transport():
    profile = _profile(missing=TRANSPORT)
    preset = profile.presets[PRESET]
    profile.presets[PRESET] = SkillPolicyPreset(
        preset_id=PRESET,
        action_option_templates={},
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors={
            **dict(preset.effect_monitors),
            ALIGN: EffectMonitorRef("builtin.composite_effect", "1"),
        },
    )
    assert (
        _module().bind_held_command_sources(
            {"transport": "acquire"}, _program(_entries()), profile, PRESET, RIGIDS
        )
        == {}
    )


@pytest.mark.parametrize("call_id", ["pick", TRANSPORT])
@pytest.mark.parametrize("revision", ["1", "2"])
def test_custom_or_wrong_revision_monitor_does_not_grant(call_id, revision):
    assert (
        _module().bind_held_command_sources(
            {"transport": "acquire"},
            _program(_entries()),
            _profile(invalid=call_id, revision=revision),
            PRESET,
            RIGIDS,
        )
        == {}
    )


@pytest.mark.parametrize("scope", ["projected", "missing_preset", "missing_profile"])
def test_binding_does_not_grant_from_projected_or_missing_profile(scope):
    profile = _profile(
        assurance=(
            EffectAssurance.PROJECTED
            if scope == "projected"
            else EffectAssurance.VERIFIED
        )
    )
    assert (
        _module().bind_held_command_sources(
            {"transport": "acquire"},
            _program(_entries()),
            SimpleNamespace() if scope == "missing_profile" else profile,
            "unknown" if scope == "missing_preset" else PRESET,
            RIGIDS,
        )
        == {}
    )


@pytest.mark.parametrize("revision", [1, 2])
def test_binding_versions_only_the_new_continuity_proof(revision):
    source = _module().bind_held_command_sources(
        {"transport": "acquire"},
        _program(_entries()),
        _profile(),
        PRESET,
        RIGIDS,
        revision=revision,
    )["compiled/segment-transport:0"]
    if revision == 2:
        assert source["command_continuity_revision"] == 2
    else:
        assert "command_continuity_revision" not in source


@pytest.mark.parametrize("revision", [None, True, False, 0, 3, "2", 2.0])
def test_binding_rejects_unknown_continuity_revision(revision):
    with pytest.raises(ValueError, match="revision"):
        _module().bind_held_command_sources(
            {"transport": "acquire"},
            _program(_entries()),
            _profile(),
            PRESET,
            RIGIDS,
            revision=revision,
        )
