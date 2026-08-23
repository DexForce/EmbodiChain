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

"""Tests for provider-free Expert Program compilation and lazy expansion."""

from __future__ import annotations

from itertools import islice

import pytest
import torch

from embodichain.lab.gym.envs.expert_program import (
    CyclicPoseTargetCfg,
    CompiledProgram,
    ExpertProgramCfg,
    ExpertProgramCompileError,
    ExpertProgramCompiler,
    ExpertProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from embodichain.lab.sim.atomic_actions import Affordance, EntityState
from embodichain.lab.sim.skills.calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
    SemanticPose,
)
from embodichain.lab.sim.skills.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)


class _NeverObserveProvider:
    """Record and reject every attempted dynamic scene observation."""

    def __init__(self) -> None:
        self.calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        self.calls += 1
        raise AssertionError("Expert Program compilation must not observe state.")


def _scene_registry() -> tuple[SceneRegistry, _NeverObserveProvider]:
    """Return static identities backed by a provider that must stay unused."""
    provider = _NeverObserveProvider()
    cube = SceneObjectRef("cube")
    tray = SceneObjectRef("tray")
    arm = SceneArticulationRef("arm")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube,
                state_provider=provider,
                aliases=("sim_cube",),
            ),
            SceneEntityRegistration(ref=tray, state_provider=provider),
            SceneEntityRegistration(ref=arm, state_provider=provider),
            SceneEntityRegistration(
                ref=SceneLinkRef("arm_tcp"),
                state_provider=provider,
                parent=arm,
                native_name="tcp",
            ),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("cube_grasp"),
                aliases=("legacy_grasp",),
                parent=cube,
                native_name="grasp",
                affordance=Affordance(),
                relative_pose=torch.eye(4),
            ),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("tray_top"),
                parent=tray,
                native_name="top",
                affordance=Affordance(),
                relative_pose=torch.eye(4),
            ),
        )
    )
    return registry, provider


def _integration() -> ExpertProgramIntegrationCfg:
    """Return one static integration selection."""
    return ExpertProgramIntegrationCfg(
        robot_profile="auto",
        scene_registry="env",
        runtime_preset="safe",
    )


def _pose(x: float, y: float = 0.0, z: float = 0.2) -> PoseCfg:
    """Build one target pose with an identity WXYZ quaternion."""
    return PoseCfg(
        position=(x, y, z),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _program(
    node: InvokeCfg | SequenceCfg | RepeatCfg | SegmentCfg,
    *,
    targets: dict[str, CyclicPoseTargetCfg] | None = None,
    program_id: str = "test_program",
) -> ExpertProgramCfg:
    """Build one valid Expert Program around a supplied node."""
    return ExpertProgramCfg(
        schema_version=2,
        program_id=program_id,
        integration=_integration(),
        program=node,
        targets={} if targets is None else targets,
    )


def _assert_pose_equal(actual: SemanticPose, expected: SemanticPose) -> None:
    """Compare owned pose tensor values."""
    assert torch.allclose(actual.position, expected.position)
    assert torch.allclose(actual.quaternion_wxyz, expected.quaternion_wxyz)


def _assert_semantic_call_equal(
    actual: SemanticCallSpec,
    expected: SemanticCallSpec,
) -> None:
    """Compare exact semantic call values whose public classes use eq=False."""
    assert type(actual) is type(expected)
    assert dict(actual.resources) == dict(expected.resources)
    if type(actual) is Pick and type(expected) is Pick:
        assert actual.object == expected.object
        assert actual.grasp == expected.grasp
    elif type(actual) is Place and type(expected) is Place:
        assert actual.object == expected.object
        assert actual.on == expected.on
        assert actual.inside == expected.inside
        assert (actual.at is None) == (expected.at is None)
        if actual.at is not None and expected.at is not None:
            _assert_pose_equal(actual.at, expected.at)
    elif type(actual) is HandOver and type(expected) is HandOver:
        assert actual.object == expected.object
        assert actual.receiver == expected.receiver
        assert (actual.final_target is None) == (expected.final_target is None)
        if actual.final_target is not None and expected.final_target is not None:
            _assert_pose_equal(actual.final_target, expected.final_target)
    elif (
        type(actual) is RegisteredSemanticCall
        and type(expected) is RegisteredSemanticCall
    ):
        assert actual.call_id == expected.call_id
        assert actual.arguments == expected.arguments
    else:  # pragma: no cover - exact supported union is exhausted above
        raise AssertionError(f"Unsupported call type {type(actual).__name__}.")


def test_compiler_matches_direct_python_semantic_calls_and_sequence_order() -> None:
    registry, provider = _scene_registry()
    target = _pose(0.5, 0.1)
    config = _program(
        SequenceCfg(
            items=(
                InvokeCfg(
                    call=PickCfg(
                        object="sim_cube",
                        grasp="legacy_grasp",
                        resources={"primary": "left_actor"},
                    )
                ),
                InvokeCfg(call=PlaceCfg(object="sim_cube", on="tray_top")),
                InvokeCfg(
                    call=HandOverCfg(
                        object="sim_cube",
                        receiver="right_actor",
                        final_target=TargetRefCfg(target="handover_pose"),
                    )
                ),
                InvokeCfg(
                    call=RegisteredSemanticCallCfg(
                        call_id="example.inspect",
                        arguments={
                            "labels": ["front", "back"],
                            "options": {"confidence": 0.9},
                        },
                    )
                ),
            )
        ),
        targets={"handover_pose": CyclicPoseTargetCfg(values=(target,))},
    )

    compiled = ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    segments = list(compiled)

    expected = (
        Pick(
            object=SceneObjectRef("cube"),
            grasp=SceneAffordanceRef("cube_grasp"),
            resources={"primary": "left_actor"},
        ),
        Place(
            object=SceneObjectRef("cube"),
            on=SceneAffordanceRef("tray_top"),
        ),
        HandOver(
            object=SceneObjectRef("cube"),
            receiver="right_actor",
            final_target=SemanticPose(target.position, target.quaternion_wxyz),
        ),
        RegisteredSemanticCall(
            call_id="example.inspect",
            arguments={
                "labels": ("front", "back"),
                "options": {"confidence": 0.9},
            },
        ),
    )
    assert len(segments) == len(expected)
    assert all(segment.implicit for segment in segments)
    assert [segment.segment_index for segment in segments] == list(range(4))
    assert [segment.calls[0].call_index for segment in segments] == list(range(4))
    assert len({segment.segment_id for segment in segments}) == 4
    for segment, expected_call in zip(segments, expected, strict=True):
        _assert_semantic_call_equal(segment.calls[0].call, expected_call)
    assert provider.calls == 0


def test_repeat_expands_independent_segments_with_cyclic_targets() -> None:
    registry, provider = _scene_registry()
    poses = (_pose(0.45, -0.2), _pose(0.45, 0.0), _pose(0.45, 0.2))
    body = SegmentCfg(
        name="move_cube",
        steps=SequenceCfg(
            items=(
                InvokeCfg(call=PickCfg(object="cube")),
                InvokeCfg(
                    call=PlaceCfg(
                        object="cube",
                        at=TargetRefCfg(target="drop_pose"),
                    )
                ),
            )
        ),
        post=(WaitStablePostCfg(entity="cube"),),
        validators=(ObjectNearTargetValidatorCfg(object="cube", target="drop_pose"),),
    )
    config = _program(
        RepeatCfg(count=3, body=body),
        targets={"drop_pose": CyclicPoseTargetCfg(values=poses)},
        program_id="repeated_cube",
    )

    compiled = ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    segments = list(compiled)
    second_pass = list(compiled)

    assert [segment.segment_id for segment in segments] == [
        segment.segment_id for segment in second_pass
    ]
    assert len(segments) == 3
    assert len({segment.segment_id for segment in segments}) == 3
    assert [segment.segment_index for segment in segments] == [0, 1, 2]
    assert [call.call_index for segment in segments for call in segment.calls] == list(
        range(6)
    )
    assert all(not segment.implicit for segment in segments)
    assert segments == second_pass
    for index, (segment, pose) in enumerate(zip(segments, poses, strict=True)):
        assert len(segment.repeat_frames) == 1
        assert segment.repeat_frames[0].path == ("program",)
        assert segment.repeat_frames[0].iteration_index == index
        assert segment.repeat_frames[0].count == 3
        place = segment.calls[1]
        assert type(place.call) is Place
        assert place.call.at is not None
        _assert_pose_equal(
            place.call.at,
            SemanticPose(pose.position, pose.quaternion_wxyz),
        )
        assert place.target_selections[0].value_index == index
        validator = segment.validators[0]
        _assert_pose_equal(validator.target_pose, place.call.at)
        assert validator.target_selection == place.target_selections[0]
        assert segment.post_policies[0].entity == SceneObjectRef("cube")
    assert provider.calls == 0


def test_repeat_expansion_is_bounded_and_never_observes_scene_providers() -> None:
    registry, provider = _scene_registry()
    config = _program(
        RepeatCfg(
            count=1_000,
            body=InvokeCfg(call=PickCfg(object="sim_cube")),
        )
    )

    compiled = ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    iterator = iter(compiled)

    assert provider.calls == 0
    first_two = list(islice(iterator, 2))
    assert [segment.segment_index for segment in first_two] == [0, 1]
    assert [segment.repeat_frames[0].iteration_index for segment in first_two] == [
        0,
        1,
    ]
    assert provider.calls == 0


def test_compiled_program_builds_cross_segment_analysis_windows_provider_free() -> None:
    registry, provider = _scene_registry()
    config = _program(
        SequenceCfg(
            items=(
                SegmentCfg(
                    name="pick",
                    steps=InvokeCfg(call=PickCfg(object="cube")),
                ),
                SegmentCfg(
                    name="place",
                    steps=InvokeCfg(
                        call=PlaceCfg(
                            object="cube",
                            at=TargetRefCfg(target="drop_pose"),
                        )
                    ),
                ),
            )
        ),
        targets={"drop_pose": CyclicPoseTargetCfg(values=(_pose(0.5),))},
    )

    compiled = ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    preflight = compiled.preflight_analyses()
    execution = compiled.sequential_execution_analysis(0)

    assert type(compiled) is CompiledProgram
    assert compiled.segment_count == 2
    assert len(tuple(compiled.iter_segments())) == 2
    assert len(preflight) == 1
    assert preflight[0].kind == "sequential_stretch"
    assert [type(call) for call in preflight[0].calls] == [Pick, Place]
    assert execution.kind == "sequential_suffix"
    assert execution.execution_prefix_length == 1
    assert [type(call) for call in execution.calls] == [Pick, Place]
    assert provider.calls == 0


def test_compilation_rechecks_expanded_call_bound_after_config_mutation() -> None:
    registry, provider = _scene_registry()
    inner = RepeatCfg(count=1, body=InvokeCfg(call=PickCfg(object="cube")))
    config = _program(RepeatCfg(count=1, body=inner))
    assert type(config.program) is RepeatCfg
    assert type(config.program.body) is RepeatCfg
    config.program.count = 1_000
    config.program.body.count = 1_000
    with pytest.raises(ExpertProgramCompileError) as error:
        ExpertProgramCompiler.from_scene_registry(registry).compile(config)

    assert error.value.code == "expanded_call_limit"
    assert provider.calls == 0


def test_wait_stable_accepts_every_canonical_scene_entity_subtype() -> None:
    registry, provider = _scene_registry()
    config = _program(
        SegmentCfg(
            name="link_settle",
            steps=InvokeCfg(call=PickCfg(object="cube")),
            post=(WaitStablePostCfg(entity="arm_tcp"),),
        )
    )

    segment = next(
        iter(ExpertProgramCompiler.from_scene_registry(registry).compile(config))
    )

    assert segment.post_policies[0].entity == SceneLinkRef("arm_tcp")
    assert provider.calls == 0


def test_compiled_program_owns_source_and_each_emitted_mutable_config() -> None:
    registry, _ = _scene_registry()
    target = CyclicPoseTargetCfg(values=(_pose(0.4), _pose(0.5)))
    registered = RegisteredSemanticCallCfg(
        call_id="example.inspect",
        arguments={"settings": {"enabled": True}},
    )
    repeat = RepeatCfg(
        count=2,
        body=SegmentCfg(
            name="inspect_and_place",
            steps=SequenceCfg(
                items=(
                    InvokeCfg(call=registered),
                    InvokeCfg(
                        call=PlaceCfg(
                            object="cube",
                            at=TargetRefCfg(target="drop_pose"),
                        )
                    ),
                )
            ),
            post=(WaitStablePostCfg(entity="cube", preset="rigid_object"),),
            validators=(
                ObjectNearTargetValidatorCfg(
                    object="cube",
                    target="drop_pose",
                    position_tolerance=0.03,
                ),
            ),
        ),
    )
    config = _program(repeat, targets={"drop_pose": target})
    compiled = ExpertProgramCompiler.from_scene_registry(registry).compile(config)

    compiled_source = config.program
    assert type(compiled_source) is RepeatCfg
    source_segment = compiled_source.body
    assert type(source_segment) is SegmentCfg
    source_steps = source_segment.steps
    assert type(source_steps) is SequenceCfg
    source_registered = source_steps.items[0].call
    assert type(source_registered) is RegisteredSemanticCallCfg
    compiled_source.count = 1
    config.targets["drop_pose"].values = (_pose(9.0),)
    source_registered.arguments["settings"]["enabled"] = False
    source_segment.post[0].preset = "changed"
    source_segment.validators[0].position_tolerance = 9.0

    first_pass = list(compiled)
    assert len(first_pass) == 2
    first_registered = first_pass[0].calls[0].call
    assert type(first_registered) is RegisteredSemanticCall
    assert first_registered.arguments["settings"]["enabled"] is True
    first_place = first_pass[0].calls[1].call
    assert type(first_place) is Place and first_place.at is not None
    assert first_place.at.position[0].item() == pytest.approx(0.4)
    assert first_pass[0].post_policies[0].cfg.preset == "rigid_object"
    assert first_pass[0].validators[0].cfg.position_tolerance == pytest.approx(0.03)

    second_pass = list(compiled)
    assert second_pass[0].post_policies[0].cfg.preset == "rigid_object"
    assert second_pass[0].validators[0].cfg.position_tolerance == pytest.approx(0.03)


def test_compiler_rejects_nested_segment_at_exact_path() -> None:
    registry, _ = _scene_registry()
    config = _program(
        SegmentCfg(
            name="outer",
            steps=SegmentCfg(
                name="inner",
                steps=InvokeCfg(call=PickCfg(object="cube")),
            ),
        )
    )

    with pytest.raises(ExpertProgramCompileError) as error:
        ExpertProgramCompiler.from_scene_registry(registry).compile(config)

    assert error.value.code == "nested_segment"
    assert error.value.path == ("program", "steps")


def test_compiler_reports_typed_scene_mismatch_at_reference_site() -> None:
    registry, _ = _scene_registry()
    config = _program(
        InvokeCfg(call=PickCfg(object="tray_top")),
    )

    with pytest.raises(ExpertProgramCompileError) as error:
        ExpertProgramCompiler.from_scene_registry(registry).compile(config)

    assert error.value.code == "scene_reference_type_mismatch"
    assert error.value.path == ("program", "call", "object")


def test_compiler_rechecks_mutated_repeat_and_target_bounds() -> None:
    registry, _ = _scene_registry()
    repeat = RepeatCfg(count=1, body=InvokeCfg(call=PickCfg(object="cube")))
    target = CyclicPoseTargetCfg(values=(_pose(0.4),))
    config = _program(repeat, targets={"drop_pose": target})
    assert type(config.program) is RepeatCfg
    config.program.count = 0

    with pytest.raises(ExpertProgramCompileError) as repeat_error:
        ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    assert repeat_error.value.code == "invalid_repeat_count"
    assert repeat_error.value.path == ("program", "count")

    config.program.count = 1
    config.targets["drop_pose"].values = ()
    with pytest.raises(ExpertProgramCompileError) as target_error:
        ExpertProgramCompiler.from_scene_registry(registry).compile(config)
    assert target_error.value.code == "empty_target_values"
    assert target_error.value.path == ("targets", "drop_pose", "values")
