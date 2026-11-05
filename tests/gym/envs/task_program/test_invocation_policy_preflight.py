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

"""Tests for config-created Task Program environment integrations."""

from __future__ import annotations
from copy import deepcopy
from .test_configured_integration import _integration_payload
from embodichain.lab.task_program import TaskProgramIntegrationCfg, decode_task_program
from embodichain.lab.task_program.integrations import (
    compile_configured_task_program,
    task_program_fingerprint,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_configured_task_program_integration,
)


def _post_occurrence_program(*, alias=False):
    payload = _integration_payload("repeated_pick_place")
    selection = TaskProgramIntegrationCfg(
        robot_profile=payload["robot_profile"]["profile_id"],
        scene_registry=payload["scene"]["registry_id"],
        runtime_preset=payload["robot_profile"]["presets"][0]["preset_id"],
    )
    if alias:
        next(
            value
            for value in payload["scene"]["rigid_objects"]
            if value["entity_id"] == "cube"
        )["simulation_uid"] = "native_cube_alias"
    declaration = dict(
        program_id="post_occurrence",
        targets={},
        program=dict(
            kind="sequence",
            items=[
                dict(
                    kind="segment",
                    name=name,
                    steps=dict(
                        kind="invoke",
                        call=dict(
                            kind="registered",
                            call_id="test.turn",
                            arguments={"object": "cube"},
                        ),
                    ),
                    post=[
                        dict(
                            kind="wait_stable",
                            entity="native_cube_alias" if alias else "cube",
                            preset="turned",
                        )
                    ],
                )
                for name in ("action", "cleanup")
            ],
        ),
    )
    return compile_configured_task_program(
        declaration, scene_binding=payload["scene"], selection=selection
    )


def test_post_policy_bindings_keep_reused_named_presets_and_independent_occurrences():
    from embodichain.lab.task_program.integrations import (
        BoundPostPolicyPresets,
        PostPolicyPresetBinding,
    )

    program = _post_occurrence_program()
    segments = tuple(program.iter_segments())
    fingerprint = task_program_fingerprint(program)
    bindings = tuple(
        PostPolicyPresetBinding(
            fingerprint, f"{program.program_id}/{segment.segment_id}:post:0", target
        )
        for segment, target in zip(segments, ("turned", "terminal"), strict=True)
    )
    bound = BoundPostPolicyPresets(program, bindings)
    assert [
        bound.preset_for(segment.post_policies[0], segment=segment)
        for segment in segments
    ] == ["turned", "terminal"]
    assert [segment.post_policies[0].cfg.preset for segment in segments] == [
        "turned",
        "turned",
    ]
    bound.validate_program(program)


def test_post_policy_bindings_reject_wrong_program_duplicate_unknown_and_changed_occurrence():
    from dataclasses import replace
    import pytest
    from embodichain.lab.task_program.integrations import (
        BoundPostPolicyPresets,
        PostPolicyPresetBinding,
    )

    program = _post_occurrence_program()
    segment = next(program.iter_segments())
    binding = PostPolicyPresetBinding(
        task_program_fingerprint(program),
        f"{program.program_id}/{segment.segment_id}:post:0",
        "terminal",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        BoundPostPolicyPresets(
            program, [replace(binding, program_fingerprint="f" * 64)]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        BoundPostPolicyPresets(program, [binding, binding])
    with pytest.raises(ValueError, match="Unknown"):
        BoundPostPolicyPresets(
            program, [replace(binding, occurrence_id="foreign:post:0")]
        )
    bound = BoundPostPolicyPresets(program, [binding])
    changed = replace(segment, name="changed")
    with pytest.raises(ValueError, match="bound program occurrence"):
        bound.preset_for(changed.post_policies[0], segment=changed)
    other = tuple(program.iter_segments())[1]
    with pytest.raises(ValueError, match="does not belong"):
        bound.preset_for(other.post_policies[0], segment=segment)


def test_post_policy_binding_preserves_canonical_scene_aliases_on_unselected_policies():
    from embodichain.lab.task_program.integrations import (
        BoundPostPolicyPresets,
        PostPolicyPresetBinding,
    )

    program = _post_occurrence_program(alias=True)
    segments = tuple(program.iter_segments())
    assert all(
        segment.post_policies[0].cfg.entity == "native_cube_alias"
        and segment.post_policies[0].entity.entity_id == "cube"
        for segment in segments
    )
    binding = PostPolicyPresetBinding(
        task_program_fingerprint(program),
        f"{program.program_id}/{segments[-1].segment_id}:post:0",
        "terminal",
    )
    bound = BoundPostPolicyPresets(program, [binding])
    assert bound.preset_for(segments[0].post_policies[0], segment=segments[0]) is None
    assert (
        bound.preset_for(segments[-1].post_policies[0], segment=segments[-1])
        == "terminal"
    )
