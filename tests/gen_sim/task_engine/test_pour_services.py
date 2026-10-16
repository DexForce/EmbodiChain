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
"""Provider-free tests of explicit pour service declarations and native adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program.configured import decode_task_lowerer
from embodichain.gen_sim.task_engine._task_program.pour_geometry import staging_pose
from embodichain.gen_sim.task_engine._task_program.pour_services import (
    PourStageRoute,
    PourTransportFactory,
    RetainedPourFactory,
    gravity_from_simulation,
    native_vertices,
)
from embodichain.gen_sim.task_engine._task_program.services import (
    _AbsolutePoseTarget,
    _MoveHeldObjectLowerer,
    _MoveHeldObjectLowererFactory,
    _MoveHeldObjectRoute,
    _SceneEntityTarget,
)
from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
    PourGoal,
    PourOptions,
    SceneEntityPose,
)
from embodichain.lab.sim.atomic_actions.goals import (
    collect_scene_dependencies,
    resolve_pose_goal,
)
from embodichain.lab.sim.atomic_actions.state import EntityState
from embodichain.lab.task_program.integrations._configured_services import (
    _PourLowererFactory,
)
from embodichain.lab.task_program.semantics import (
    GRASP_AFFORDANCE_CAPABILITY,
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
    SemanticEffectKind,
)
from embodichain.lab.task_program.semantics.scene import _SimulationEntityStateProvider


def _simulation(**kwargs):
    return SimpleNamespace(
        physics_backend=kwargs.get("physics_backend", "default"),
        num_envs=kwargs.get("num_envs", 1),
        sim_config=SimpleNamespace(
            physics_cfg=SimpleNamespace(gravity=kwargs.get("gravity", (0, 0, -9.81)))
        ),
        get_rigid_object=Mock(
            side_effect=AssertionError(
                "Never resolve simulation UIDs by canonical name"
            )
        ),
    )


def _registry():
    raw = torch.tensor(
        [[[-0.02, -0.01, -0.03], [0.02, 0.01, 0.03], [0.02, -0.01, -0.03]]]
    )
    scale = torch.tensor([2.0, 3.0, 4.0])
    source = SimpleNamespace(
        uid="native_source_37",
        num_instances=1,
        get_vertices=Mock(return_value=raw * scale),
    )
    receiver = SimpleNamespace(
        uid="native_receiver_81",
        num_instances=1,
        get_vertices=Mock(return_value=raw * 5),
    )
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("source"),
                state_provider=_SimulationEntityStateProvider(source),
                aliases=("source_alias",),
            ),
            SceneEntityRegistration(
                ref=SceneObjectRef("receiver"),
                state_provider=_SimulationEntityStateProvider(receiver),
            ),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("source_grasp"),
                parent=SceneObjectRef("source"),
                native_name="grasp",
                relative_pose=torch.eye(4),
                affordance_revision="test-1",
                affordance_capabilities=(GRASP_AFFORDANCE_CAPABILITY,),
                affordance=AxisAlignAffordance(
                    mesh_vertices=(raw * scale)[0],
                    mesh_triangles=torch.tensor([[0, 1, 2]]),
                    internal_axis=torch.tensor([1.0, 0.0, 0.0]),
                ),
            ),
        )
    )
    return registry, source, receiver, (raw * scale)[0].numpy()


def _factory():
    return PourTransportFactory(
        routes=(
            _MoveHeldObjectRoute("source", "stage", _SceneEntityTarget("receiver")),
            _MoveHeldObjectRoute(
                "source", "return", _AbsolutePoseTarget((0.1, 0.2, 0.3), (0, 0, 0, 1))
            ),
        ),
        staging_routes=(PourStageRoute("source", "stage", "receiver", 0.025),),
    )


def _create(factory):
    registry, source, receiver, scaled = _registry()
    robot = object()
    simulation = _simulation()
    lowerer = factory.create(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    return lowerer, registry, simulation, source, receiver, scaled


def _call(target="stage", **kwargs):
    return RegisteredSemanticCall(
        call_id="simulation.move_held_object",
        arguments={"object": "source", "target": target, **kwargs},
    )


def _context():
    source_pose, receiver_pose = torch.eye(4, dtype=torch.float64), torch.eye(
        4, dtype=torch.float64
    )
    source_pose[:3, :3] = torch.as_tensor(
        Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    )
    source_pose[:3, 3] = torch.tensor([0.1, 0.2, 0.9])
    receiver_pose[:3, 3] = torch.tensor([0.5, -0.2, 0.7])
    return SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=torch.zeros((1, 7), dtype=torch.float64)),
        scene=SimpleNamespace(
            entities={
                "source": EntityState(source_pose.unsqueeze(0)),
                "receiver": EntityState(receiver_pose),
            }
        ),
    )


def test_native_vertices_are_scaled_local_owned_and_resolved_via_registry():
    registry, source, _, expected = _registry()
    result = native_vertices(registry, "source_alias")
    source.get_vertices.assert_called_once_with(env_ids=[0], scale=True)
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result[0, 0] = 9
    source.get_vertices.return_value.add_(10)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros((0, 3)),
        torch.zeros((1, 0, 3)),
        torch.zeros((2, 3, 3)),
        torch.zeros((1, 3, 3), dtype=torch.long),
        torch.full((1, 3, 3), float("nan")),
    ],
)
def test_native_geometry_missing_or_invalid_fails_closed(value):
    registry, source, _, _ = _registry()
    source.get_vertices.return_value = value
    with pytest.raises(ValueError, match="vertices"):
        native_vertices(registry, "source")


def test_native_provider_and_instance_count_are_required():
    registry, source, _, _ = _registry()
    source.num_instances = 2
    with pytest.raises(ValueError, match="one native instance"):
        native_vertices(registry, "source")
    source.get_vertices = None
    with pytest.raises(TypeError, match="native rigid-object"):
        native_vertices(registry, "source")


@pytest.mark.parametrize(
    "values",
    [
        {"physics_backend": "newton"},
        {"num_envs": 2},
        {"gravity": None},
        {"gravity": (0, 0, 0)},
        {"gravity": (0, float("nan"), -1)},
        {"gravity": (0, -1)},
    ],
)
def test_gravity_scope_or_missing_measurement_fails_closed(values):
    with pytest.raises(ValueError):
        gravity_from_simulation(_simulation(**values))


def test_actual_gravity_is_not_replaced_by_a_default():
    assert gravity_from_simulation(_simulation(gravity=(0, 0, -3.2))) == (0, 0, -3.2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stage_uses_current_observation_and_same_shared_geometry_without_mutation(
    dtype,
):
    factory = _factory()
    lowerer, _, simulation, source, receiver, scaled = _create(factory)
    context = _context()
    context.robot.qpos = context.robot.qpos.to(dtype=dtype)
    context.scene.entities = {
        key: replace(state, pose=state.pose.to(dtype=dtype))
        for key, state in context.scene.entities.items()
    }
    before = {key: state.pose.clone() for key, state in context.scene.entities.items()}
    result = lowerer.lower(
        _call(reference="receiver"),
        context=context,
        bound=None,
        option_template=MoveHeldObjectOptions(),
    )
    expected = staging_pose(
        scaled,
        receiver.get_vertices.return_value[0].numpy(),
        before["source"][0].numpy(),
        before["receiver"].numpy(),
        clearance_m=0.025,
        gravity_world=(0, 0, -9.81),
    )
    assert type(result.goal) is HeldObjectPoseGoal
    assert type(result.goal.object_target_pose) is SceneEntityPose
    resolved = resolve_pose_goal(
        result.goal.object_target_pose, context, name="pour_staging"
    )
    assert resolved.shape == (1, 4, 4)
    assert resolved.dtype == torch.float32
    np.testing.assert_array_equal(resolved[0].numpy(), expected.astype(np.float32))
    assert torch.equal(resolved[0, :3, :3], before["source"][0, :3, :3].float())
    assert collect_scene_dependencies(result.goal) == ("receiver",)
    for key in before:
        torch.testing.assert_close(context.scene.entities[key].pose, before[key])
    assert result.registered_effect.held_objects[0].object_id == "source"
    assert result.registered_effect.held_objects[0].acquire_before_segment is None
    assert lowerer.staging_routes == factory.staging_routes
    assert lowerer.gravity_world == (0, 0, -9.81)
    with pytest.raises(TypeError):
        lowerer.scene_vertices["other"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        lowerer.scene_vertices["source"][0, 0] = 8
    source.get_vertices.assert_called_once_with(env_ids=[0], scale=True)
    simulation.get_rigid_object.assert_not_called()


def test_stage_goal_follows_receiver_translation_but_preserves_entry_world_R():
    lowerer, *_ = _create(_factory())
    context = _context()
    context.robot.qpos = context.robot.qpos.float()
    context.scene.entities = {
        key: replace(state, pose=state.pose.float())
        for key, state in context.scene.entities.items()
    }
    result = lowerer.lower(
        _call(reference="receiver"),
        context=context,
        bound=None,
        option_template=MoveHeldObjectOptions(),
    )
    goal = result.goal.object_target_pose
    assert goal.entity_id == "receiver"
    assert goal.relative_pose is None
    before = resolve_pose_goal(goal, context, name="pour_staging")
    receiver = context.scene.entities["receiver"]
    delta = torch.tensor([0.07, -0.04, 0.02])
    moved = receiver.pose.clone()
    moved[:3, 3] += delta
    moved[:3, :3] = torch.as_tensor(
        Rotation.from_euler("xyz", [0.5, 0.4, -0.3]).as_matrix(), dtype=torch.float32
    )
    context.scene.entities["receiver"] = replace(receiver, pose=moved)
    after = resolve_pose_goal(goal, context, name="pour_staging")
    torch.testing.assert_close(
        after[0, :3, 3] - before[0, :3, 3], delta, rtol=0, atol=1e-7
    )
    assert torch.equal(after[0, :3, :3], before[0, :3, :3])
    assert torch.equal(
        after[0, :3, :3], context.scene.entities["source"].pose[0, :3, :3]
    )
    assert collect_scene_dependencies(result.goal) == ("receiver",)


def test_stage_scene_reference_is_canonical_not_a_configured_alias():
    registry, *_ = _registry()
    registry = SceneRegistry(
        tuple(
            (
                replace(item, aliases=(*item.aliases, "receiver_alias"))
                if item.ref.entity_id == "receiver"
                else item
            )
            for item in registry.registrations
        )
    )
    original = _factory()
    factory = replace(
        original,
        routes=(
            _MoveHeldObjectRoute(
                "source", "stage", _SceneEntityTarget("receiver_alias")
            ),
            original.routes[1],
        ),
        staging_routes=(PourStageRoute("source", "stage", "receiver_alias", 0.025),),
    )
    robot = object()
    lowerer = factory.create(
        simulation=_simulation(),
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    context = _context()
    result = lowerer.lower(
        _call(reference="receiver_alias"),
        context=context,
        bound=None,
        option_template=MoveHeldObjectOptions(),
    )
    goal = result.goal.object_target_pose
    assert goal.entity_id == "receiver"
    assert collect_scene_dependencies(result.goal) == ("receiver",)
    assert "receiver_alias" not in context.scene.entities
    assert torch.isfinite(resolve_pose_goal(goal, context, name="pour_staging")).all()
    context.scene.entities.pop("receiver")
    with pytest.raises(KeyError, match="receiver"):
        resolve_pose_goal(goal, context, name="pour_staging")


def test_nonstage_lowering_is_identical_to_the_original_and_never_reads_context():
    factory = _factory()
    lowerer, *_ = _create(factory)
    original = _MoveHeldObjectLowerer(factory.routes)
    context = Mock()
    kwargs = dict(context=context, bound=None, option_template=MoveHeldObjectOptions())
    expected, actual = original.lower(_call("return"), **kwargs), lowerer.lower(
        _call("return"), **kwargs
    )
    torch.testing.assert_close(
        actual.goal.object_target_pose, expected.goal.object_target_pose
    )
    assert actual.registered_effect == expected.registered_effect
    assert context.mock_calls == []


def test_stage_and_pour_lookahead_defer_only_candidate_dependent_targets():
    lowerer, *_ = _create(_factory())
    kwargs = dict(
        picked_object=SceneObjectRef("source"), bound=None, previous_target=None
    )
    assert lowerer.pick_lookahead_targets(_call(), **kwargs) == ()
    assert len(lowerer.pick_lookahead_targets(_call("return"), **kwargs)) == 1
    assert (
        lowerer.pick_lookahead_targets(
            _call(), **dict(kwargs, picked_object=SceneObjectRef("other"))
        )
        is None
    )
    retained, *_ = _create(RetainedPourFactory("source"))
    call = RegisteredSemanticCall(
        call_id="simulation.pour", arguments={"object": "source"}
    )
    assert retained.pick_lookahead_targets(call, **kwargs) == ()
    assert (
        retained.pick_lookahead_targets(
            call, **dict(kwargs, picked_object=SceneObjectRef("other"))
        )
        is None
    )


@pytest.mark.parametrize("change", ["batch", "row", "source", "receiver"])
def test_stage_rejects_unsupported_batches_and_missing_observations(change):
    lowerer, *_ = _create(_factory())
    context = _context()
    if change == "batch":
        context.batch_size = 2
    elif change == "row":
        context.env_ids = torch.tensor([2])
    else:
        context.scene.entities[change] = EntityState(torch.eye(4), confidence=0)
    with pytest.raises(ValueError):
        lowerer.lower(
            _call(),
            context=context,
            bound=None,
            option_template=MoveHeldObjectOptions(),
        )


def test_retained_pour_reuses_the_existing_goal_and_exact_phase_effect_contract():
    lowerer, *_ = _create(RetainedPourFactory("source"))
    call = RegisteredSemanticCall(
        call_id="simulation.pour", arguments={"object": "source"}
    )
    options = PourOptions()
    before = deepcopy(options)
    context = Mock()
    result = lowerer.lower(call, context=context, bound=None, option_template=options)
    assert type(result.goal) is PourGoal
    effect = result.registered_effect
    assert effect.effect_kind is SemanticEffectKind.ATTACH
    (held,) = effect.held_objects
    assert held.expectation_id == held.slot_id == "primary"
    assert held.object_id == "source" and held.relation is HeldObjectRelation.ATTACHED
    assert held.acquire_before_segment == "pour" and held.retain_during_segments == (
        "pour",
    )
    assert asdict(options) == asdict(before)
    assert context.mock_calls == []
    assert not hasattr(lowerer, "object_to_eef")
    assert not hasattr(lowerer, "candidates")
    with pytest.raises(TypeError, match="exact PourOptions"):
        lowerer.lower(
            call, context=context, bound=None, option_template=MoveHeldObjectOptions()
        )
    with pytest.raises(ValueError, match="arguments"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id="simulation.pour", arguments={"object": "receiver"}
            ),
            context=context,
            bound=None,
            option_template=options,
        )


def _declaration():
    return {
        "kind": "pour_transport",
        "routes": [
            {
                "object_id": "source",
                "target_id": "stage",
                "pose": {"kind": "scene_entity", "entity_id": "receiver"},
            }
        ],
        "staging_routes": [
            {
                "object_id": "source",
                "target_id": "stage",
                "receiver_id": "receiver",
                "clearance_m": 0.02,
            }
        ],
    }


def test_closed_decoder_factory_identity_and_legacy_kinds_remain_distinct():
    factory = decode_task_lowerer(_declaration(), path="test")
    assert type(factory) is PourTransportFactory
    assert factory.call_id == "simulation.move_held_object"
    assert factory.revision != _MoveHeldObjectLowererFactory.revision
    retained = decode_task_lowerer(
        {"kind": "retained_pour", "object_id": "source"}, path="test"
    )
    assert type(retained) is RetainedPourFactory
    legacy = decode_task_lowerer({"kind": "pour", "object_id": "source"}, path="test")
    assert type(legacy) is _PourLowererFactory and legacy.revision == "1"
    normal = _declaration()
    normal["kind"] = "move_held_object"
    normal.pop("staging_routes")
    assert (
        type(decode_task_lowerer(normal, path="test")) is _MoveHeldObjectLowererFactory
    )
    with pytest.raises(FrozenInstanceError):
        retained.object_id = "different"
    first, *_ = _create(factory)
    second, *_ = _create(factory)
    assert first is not second


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown",
        "stage_unknown",
        "empty",
        "duplicate",
        "different_source",
        "different_receiver",
        "missing",
        "negative",
        "nan",
        "boolean",
    ],
)
def test_decoder_rejects_unknown_missing_ambiguous_or_invalid_stage_declarations(
    invalid,
):
    data = _declaration()
    stage = data["staging_routes"][0]
    if invalid == "unknown":
        data["class_type"] = "arbitrary.Module"
    elif invalid == "stage_unknown":
        stage["fallback"] = True
    elif invalid == "empty":
        data["staging_routes"] = []
    elif invalid == "duplicate":
        data["staging_routes"].append(deepcopy(stage))
    elif invalid == "different_source":
        stage["object_id"] = "different"
    elif invalid == "different_receiver":
        stage["receiver_id"] = "different"
    elif invalid == "missing":
        data.pop("staging_routes")
    else:
        stage["clearance_m"] = {"negative": -1, "nan": float("nan"), "boolean": True}[
            invalid
        ]
    with pytest.raises((ValueError, TypeError)):
        decode_task_lowerer(data, path="test")


def test_factory_rejects_another_robot_and_unknown_retained_fields():
    registry, *_ = _registry()
    with pytest.raises(ValueError, match="exact robot"):
        _factory().create(
            simulation=_simulation(),
            robot=object(),
            scene_registry=registry,
            engine=SimpleNamespace(robot=object()),
        )
    with pytest.raises(ValueError):
        decode_task_lowerer(
            {"kind": "retained_pour", "object_id": "source", "G": []}, path="test"
        )
