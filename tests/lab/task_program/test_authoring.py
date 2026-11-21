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

"""Behavioral coverage for shorthand, shared targets and source identities."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from embodichain.lab.task_program import decode_task_program, TaskProgramCompiler
from embodichain.lab.task_program.language.authoring import normalize_task_program
from embodichain.lab.task_program.language.decoder import TaskProgramDecodeError
from embodichain.lab.task_program.language.semantic_contracts import (
    SemanticCallContract,
    validate_semantic_call_payload,
)
from embodichain.lab.task_program.integrations._program_identity import (
    task_program_fingerprint,
)
from embodichain.lab.task_program.semantics.calls import (
    MoveHeldObjectCall,
    SemanticRelativePose,
)
from embodichain.lab.task_program.semantics.integration import SceneManifest
from embodichain.lab.sim.atomic_skills.goals import resolve_pose_goal
from test_program_compiler import _scene_registry, _integration


def _data():
    return dict(
        program_id="authoring",
        integration=_integration().to_dict(),
        targets={
            "destination": dict(
                kind="pose",
                frame="world",
                position=[0.1, 0.2, 0.3],
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        },
        program=[
            dict(
                segment="present",
                steps=[dict(move_held_object=dict(object="cube", at="destination"))],
                post=[dict(hold=dict(object="cube", at="destination", duration_s=0.2))],
            )
        ],
    )


@pytest.mark.parametrize("with_contract", [False, True])
def test_shorthand_and_canonical_share_compiled_identity_without_observation(
    with_contract,
):
    data = _data()
    if with_contract:
        data["program"][0].update(
            instruction="Move the held cube to its presentation pose.",
            contract={
                "outcomes": [
                    {"kind": "call_effect", "ref": 0, "assurance": "projected"}
                ]
            },
        )
    original = deepcopy(data)
    normalized = normalize_task_program(data)
    registry, provider = _scene_registry()
    compiler = TaskProgramCompiler(SceneManifest.from_registry(registry))
    short = compiler.compile(decode_task_program(data), strict_contracts=with_contract)
    canonical = compiler.compile(
        decode_task_program(normalized.declaration), strict_contracts=with_contract
    )
    assert task_program_fingerprint(short) == task_program_fingerprint(canonical)
    segment = next(short.iter_segments())
    assert type(segment.calls[0].call) is MoveHeldObjectCall
    assert segment.calls[0].call.object.entity_id == "cube"
    assert segment.post_policies[0].cfg.duration_s == 0.2
    assert provider.calls == 0
    assert data == original
    if with_contract:
        from embodichain.lab.task_program.compiler.program import (
            TaskProgramCompileError,
        )

        for field in ("instruction", "contract"):
            incomplete = deepcopy(data)
            del incomplete["program"][0][field]
            with pytest.raises(TaskProgramCompileError) as error:
                compiler.compile(decode_task_program(incomplete), strict_contracts=True)
            assert error.value.path == ("program", 0, field)


@pytest.mark.parametrize("frame", ["parent", "world_axes"])
def test_relative_target_resolves_rotated_reference(frame):
    from types import SimpleNamespace

    target = SemanticRelativePose("tray", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), frame)
    reference = torch.eye(4).repeat(2, 1, 1)
    reference[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    reference[:, :3, 3] = torch.tensor([2.0, 3.0, 4.0])
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={"tray": SimpleNamespace(pose=reference, confidence=1.0)}
        ),
        robot=SimpleNamespace(qpos=torch.zeros(2, 1)),
        batch_size=2,
    )
    actual = resolve_pose_goal(target.to_goal(), context, name="destination")
    expected = (2.0, 4.0, 4.0) if frame == "parent" else (3.0, 3.0, 4.0)
    torch.testing.assert_close(actual[:, :3, 3], torch.tensor(expected).expand(2, -1))
    torch.testing.assert_close(
        actual[0, :3, :3], reference[0, :3, :3] if frame == "parent" else torch.eye(3)
    )


def test_shorthand_reports_authored_reference_path():
    data = _data()
    data["program"][0]["steps"][0]["move_held_object"]["at"] = "missing"
    with pytest.raises(TaskProgramDecodeError) as error:
        decode_task_program(data)
    assert error.value.code == "unknown_target"
    assert error.value.path == ("program", 0, "steps", 0, "move_held_object", "at")


def test_shorthand_does_not_infer_object_or_execute_unknown_fields():
    data = _data()
    del data["program"][0]["steps"][0]["move_held_object"]["object"]
    with pytest.raises(TaskProgramDecodeError, match="object"):
        decode_task_program(data)
    data = _data()
    data["program"][0]["steps"][0]["pick"] = {"object": "cube"}
    with pytest.raises(TaskProgramDecodeError, match="exactly one"):
        decode_task_program(data)


def test_demo_shorthand_requires_explicit_registered_alias_and_exact_roles():
    data = _data()
    data["program"] = [{"demo_skill": {"skill": "trace", "roles": {"tool": "cube"}}}]
    with pytest.raises(TaskProgramDecodeError, match="Unknown Demo Skill"):
        normalize_task_program(data)
    result = normalize_task_program(data, demo_skills={"trace": "task.trace"})
    assert result.declaration["program"]["call"] == dict(
        kind="registered",
        call_id="task.trace",
        arguments={"tool": "cube"},
        resources={},
    )
    contract = {
        "task.trace": SemanticCallContract(
            {"tool": "entity"}, {}, preconditions=("primary_free",)
        )
    }
    validate_semantic_call_payload(
        result.declaration["program"]["call"], contracts=contract
    )
    with pytest.raises(ValueError, match="arguments"):
        validate_semantic_call_payload(
            dict(kind="registered", call_id="task.trace", arguments={}),
            contracts=contract,
        )


def test_planning_contract_includes_explicit_transport_object_and_builtin_options():
    payload = dict(
        kind="registered",
        call_id="simulation.move_held_object",
        arguments=dict(target="destination", object="cube", reference="tray"),
    )
    validate_semantic_call_payload(payload)
    legacy = {payload["call_id"]: SemanticCallContract({"target": "target"}, {})}
    with pytest.raises(ValueError, match="optional keys"):
        validate_semantic_call_payload(payload, contracts=legacy)
    validate_semantic_call_payload(dict(kind="pick", object="cube", grasp="cube_grasp"))
    validate_semantic_call_payload(
        dict(
            kind="hand_over",
            object="cube",
            final_target=dict(kind="target_ref", target="delivery"),
            resources=dict(source="left", destination="right"),
        )
    )


def test_relative_target_canonicalizes_aliases_and_rejects_unobserved_frames():
    from embodichain.lab.task_program import TaskProgramCompileError

    data = _data()
    data["targets"]["destination"] = dict(
        kind="scene_relative",
        entity="sim_cube",
        frame="parent",
        position=[0.0, 0.0, 0.1],
        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    registry, provider = _scene_registry()
    compiler = TaskProgramCompiler(SceneManifest.from_registry(registry))
    compiled = compiler.compile(decode_task_program(data))
    assert next(compiled.iter_segments()).calls[0].call.at.entity_id == "cube"
    data["targets"]["destination"]["entity"] = "cube_grasp"
    with pytest.raises(TaskProgramCompileError, match="affordance frames"):
        compiler.compile(decode_task_program(data))
    assert provider.calls == 0


def test_authoring_diagnostics_do_not_change_serialized_program_payload():
    import json
    from embodichain.lab.task_program import loads_task_program_json

    config = decode_task_program(_data())
    payload = config.to_dict()
    assert set(payload) == {
        "program_id",
        "instruction",
        "integration",
        "targets",
        "program",
    }
    restored = loads_task_program_json(json.dumps(payload))
    registry, _ = _scene_registry()
    compiler = TaskProgramCompiler(SceneManifest.from_registry(registry))
    assert task_program_fingerprint(
        compiler.compile(config)
    ) == task_program_fingerprint(compiler.compile(restored))


def test_unknown_shorthand_action_points_to_the_authored_action_name():
    data = _data()
    data["program"] = [{"unsupported_action": {"object": "cube"}}]
    with pytest.raises(TaskProgramDecodeError) as error:
        decode_task_program(data)
    assert error.value.path == ("program", 0, "unsupported_action")
