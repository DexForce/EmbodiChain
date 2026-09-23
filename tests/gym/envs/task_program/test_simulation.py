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

"""Tests for explicit Task Program simulation bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.task_program.integrations.simulation import (
    AntipodalGraspAffordanceBinding,
    ContainerAffordanceBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    RigidizedArticulationAntipodalGraspBinding,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationRigidizedArticulationObjectBinding,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    SupportSurfaceAffordanceBinding,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AxisAlignAffordance,
    CARTESIAN_POSE_CAPABILITY,
    GRASP_CAPABILITY,
    PickUpOptions,
)
from embodichain.lab.task_program.semantics import (
    ContainerAffordance,
    EffectAssurance,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    RobotResource,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneDynamics,
    SceneLinkRef,
    SceneObjectRef,
    SkillPolicyPreset,
    SupportSurfaceAffordance,
)
from embodichain.lab.task_program.semantics.integration import SceneManifest
from embodichain.lab.task_program.semantics.profiles import ResourceEndpoint

_BATCH_SIZE = 2


class _RigidObject:
    """Minimal selected rigid object with a batched triangle mesh."""

    def __init__(self) -> None:
        self.pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
        self.vertices = torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=torch.float32,
        ).repeat(_BATCH_SIZE, 1, 1)
        self.triangles = torch.tensor(
            (((0, 1, 2),),),
            dtype=torch.int32,
        ).repeat(_BATCH_SIZE, 1, 1)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self.pose

    def get_vertices(
        self,
        env_ids: list[int],
        *,
        scale: bool,
    ) -> torch.Tensor:
        assert scale is True
        return self.vertices[env_ids]

    def get_triangles(self, env_ids: list[int]) -> torch.Tensor:
        return self.triangles[env_ids]


class _Articulation:
    """Minimal articulation exposing exact joint and link lookup surfaces."""

    joint_names = ("drawer_slide",)
    link_names = ("drawer_handle",)

    def __init__(self) -> None:
        self.pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
        self.qpos = torch.tensor(((0.1,), (0.2,)), dtype=torch.float32)
        self.link_pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
        self.link_pose[:, 0, 3] = torch.tensor((0.3, 0.4))

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self.pose

    def get_qpos(self, *, target: bool) -> torch.Tensor:
        assert target is False
        return self.qpos

    def get_link_pose(
        self,
        link_name: str,
        *,
        env_ids: list[int],
        to_matrix: bool,
    ) -> torch.Tensor:
        assert link_name == "drawer_handle"
        assert to_matrix is True
        return self.link_pose[env_ids]


class _Simulation:
    """Explicit native-UID lookup fixture."""

    def __init__(self) -> None:
        self.rigid_object = _RigidObject()
        self.articulation = _Articulation()

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        return self.rigid_object if uid == "native_cube" else None

    def get_articulation(self, uid: str) -> _Articulation | None:
        return self.articulation if uid == "native_drawer" else None


class _RigidizedArticulation:
    """Floating articulation whose only joint is locked by coincident limits."""

    uid = "cube_articulation"
    pk_chain = None
    joint_names = ("top_turn",)
    link_names = ("lower_two_layers",)

    def __init__(
        self,
        *,
        fixed_base: bool = False,
        limits: tuple[float, float] = (0.0, 0.0),
        vertices: torch.Tensor | None = None,
    ) -> None:
        self.cfg = SimpleNamespace(root_props=SimpleNamespace(fixed_base=fixed_base))
        self.pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
        self.qpos = torch.zeros((_BATCH_SIZE, 1), dtype=torch.float32)
        self.limits = torch.tensor(limits, dtype=torch.float32).reshape(1, 1, 2)
        self.limits = self.limits.repeat(_BATCH_SIZE, 1, 1)
        self.vertices = (
            torch.tensor(
                ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
                dtype=torch.float32,
            )
            if vertices is None
            else vertices
        )
        self.triangles = torch.tensor(((0, 1, 2),), dtype=torch.int64)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self.pose

    def get_link_pose(self, link_name: str, *, to_matrix: bool) -> torch.Tensor:
        assert link_name == "lower_two_layers"
        assert to_matrix is True
        return self.pose

    def get_link_vert_face(self, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert link_name == "lower_two_layers"
        return self.vertices, self.triangles

    def get_qpos(self, *, target: bool = False) -> torch.Tensor:
        del target
        return self.qpos

    def get_joint_drive(self) -> tuple[torch.Tensor, ...]:
        stiffness = torch.ones_like(self.qpos)
        zeros = torch.zeros_like(self.qpos)
        return (stiffness, zeros, zeros, zeros, zeros, zeros)

    def get_qpos_limits(self) -> torch.Tensor:
        return self.limits


class _RigidizedSimulation:
    """Record which native lookup path the rigidized binding uses."""

    def __init__(self, articulation: _RigidizedArticulation | None = None) -> None:
        self.articulation = articulation or _RigidizedArticulation()
        self.articulation_lookups: list[str] = []
        self.rigid_object_lookups: list[str] = []

    def get_articulation(self, uid: str) -> _RigidizedArticulation | None:
        self.articulation_lookups.append(uid)
        return self.articulation if uid == "cube_articulation" else None

    def get_rigid_object(self, uid: str) -> None:
        self.rigid_object_lookups.append(uid)
        return None


class _Robot:
    """Minimal robot control-part lookup fixture."""

    control_parts = {"arm": object(), "hand": object()}

    def get_joint_ids(self, *, name: str) -> list[int]:
        return {"arm": [0, 1], "hand": [2]}[name]


@dataclass(frozen=True, slots=True)
class _MobileEndpoint(ResourceEndpoint):
    """Test-only non-joint endpoint declaration."""

    controller_id: str


def _scene_binding() -> SimulationSceneBinding:
    """Build one cube-and-drawer binding using only typed declarations."""
    return SimulationSceneBinding(
        registry_id="tabletop",
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id="cube",
                simulation_uid="native_cube",
                aliases=("perceived_cube",),
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="cube",
                default_grasp_affordance="cube_grasp",
            ),
        ),
        articulations=(
            SimulationArticulationBinding(
                entity_id="drawer",
                simulation_uid="native_drawer",
                semantic_type="drawer",
            ),
        ),
        links=(
            SimulationArticulationLinkBinding(
                entity_id="drawer_handle_link",
                articulation_id="drawer",
                native_link_name="drawer_handle",
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id="cube_grasp",
                object_id="cube",
                native_name="mesh_antipodal",
                revision="cube-grasp-v1",
            ),
        ),
        support_surfaces=(
            SupportSurfaceAffordanceBinding(
                entity_id="cube_support_target",
                parent_id="cube",
                native_name="support_target",
                object_target_pose=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.25,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                minimum_confidence=0.7,
                is_default=True,
            ),
        ),
        containers=(
            ContainerAffordanceBinding(
                entity_id="drawer_inside_target",
                parent_id="drawer_handle_link",
                native_name="inside_target",
                object_target_pose=(
                    1.0,
                    0.0,
                    0.0,
                    0.1,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                minimum_confidence=0.8,
                is_default=True,
            ),
        ),
    )


def _rigidized_scene_binding() -> SimulationSceneBinding:
    """Build one provider-free Rubik's Cube binding."""
    return SimulationSceneBinding(
        registry_id="rubiks_scene",
        rigidized_articulations=(
            SimulationRigidizedArticulationObjectBinding(
                entity_id="rubiks_cube",
                simulation_uid="cube_articulation",
                locked_qpos={"top_turn": 0.0},
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="rubiks_cube",
                default_grasp_affordance="rubiks_cube_grasp",
            ),
        ),
        rigidized_articulation_grasps=(
            RigidizedArticulationAntipodalGraspBinding(
                entity_id="rubiks_cube_grasp",
                object_id="rubiks_cube",
                grasp_link="lower_two_layers",
                native_name="lower_two_layers",
                revision="1",
            ),
        ),
    )


def test_rigidized_articulation_binding_declares_object_and_grasp() -> None:
    manifest = _rigidized_scene_binding().declare()

    object_entry = manifest.lookup(SceneObjectRef("rubiks_cube"))
    grasp_entry = manifest.lookup(SceneAffordanceRef("rubiks_cube_grasp"))
    assert object_entry.semantic_type == "rubiks_cube"
    assert grasp_entry.parent == SceneObjectRef("rubiks_cube")
    assert grasp_entry.affordance_payload_type is AntipodalAffordance


@pytest.mark.parametrize(
    ("locked_qpos", "exception", "message"),
    [
        ({}, ValueError, "must not be empty"),
        ({"top_turn": True}, TypeError, "finite number"),
        ({"top_turn": float("inf")}, ValueError, "finite"),
    ],
)
def test_rigidized_articulation_binding_rejects_invalid_locked_qpos(
    locked_qpos: dict[str, float],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        SimulationRigidizedArticulationObjectBinding(
            entity_id="rubiks_cube",
            simulation_uid="cube_articulation",
            locked_qpos=locked_qpos,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "exception"),
    [
        ("joint_position_tolerance", 0.0, ValueError),
        ("joint_position_tolerance", float("inf"), ValueError),
        ("joint_position_tolerance", 10.0, ValueError),
        ("link_transform_tolerance", -1.0, ValueError),
        ("link_transform_tolerance", True, TypeError),
        ("link_transform_tolerance", 10.0, ValueError),
    ],
)
def test_rigidized_articulation_binding_rejects_invalid_tolerance(
    field_name: str,
    value: float,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception, match=field_name):
        SimulationRigidizedArticulationObjectBinding(
            entity_id="rubiks_cube",
            simulation_uid="cube_articulation",
            locked_qpos={"top_turn": 0.0},
            **{field_name: value},
        )


def test_rigidized_articulation_binding_owns_locked_qpos() -> None:
    locked_qpos = {"top_turn": 0.0}
    binding = SimulationRigidizedArticulationObjectBinding(
        entity_id="rubiks_cube",
        simulation_uid="cube_articulation",
        locked_qpos=locked_qpos,
    )

    locked_qpos["top_turn"] = 1.0

    assert dict(binding.locked_qpos) == {"top_turn": 0.0}
    with pytest.raises(TypeError):
        binding.locked_qpos["top_turn"] = 1.0  # type: ignore[index]


def test_rigidized_articulation_grasp_requires_rigidized_parent() -> None:
    grasp = _rigidized_scene_binding().rigidized_articulation_grasps[0]

    with pytest.raises(KeyError, match="unbound rigidized articulation"):
        SimulationSceneBinding(
            registry_id="rubiks_scene",
            rigidized_articulation_grasps=(grasp,),
        ).declare()


def test_ordinary_antipodal_grasp_rejects_rigidized_parent() -> None:
    root = _rigidized_scene_binding().rigidized_articulations[0]

    with pytest.raises(KeyError, match="unbound object"):
        SimulationSceneBinding(
            registry_id="rubiks_scene",
            rigidized_articulations=(root,),
            antipodal_grasps=(
                AntipodalGraspAffordanceBinding(
                    entity_id="rubiks_cube_grasp",
                    object_id="rubiks_cube",
                    native_name="mesh_antipodal",
                    revision="1",
                ),
            ),
        ).declare()


def test_rigidized_articulation_ids_share_global_namespace() -> None:
    root = _rigidized_scene_binding().rigidized_articulations[0]

    with pytest.raises(ValueError, match="must be unique"):
        SimulationSceneBinding(
            registry_id="rubiks_scene",
            rigidized_articulations=(root,),
            rigid_objects=(
                SimulationRigidObjectBinding(
                    entity_id="rubiks_cube",
                    simulation_uid="native_cube",
                ),
            ),
        )


def test_rigidized_articulation_binding_builds_from_native_articulation() -> None:
    simulation = _RigidizedSimulation()

    registry = _rigidized_scene_binding().build(simulation)  # type: ignore[arg-type]

    root = registry.lookup(SceneObjectRef("rubiks_cube"))
    grasp = registry.lookup(SceneAffordanceRef("rubiks_cube_grasp"))
    assert root.semantic_type == "rubiks_cube"
    assert isinstance(grasp.affordance, AntipodalAffordance)
    assert grasp.parent == SceneObjectRef("rubiks_cube")
    assert simulation.articulation_lookups
    assert set(simulation.articulation_lookups) == {"cube_articulation"}
    assert simulation.rigid_object_lookups == []


def test_rigidized_articulation_remains_a_placement_parent() -> None:
    binding = _rigidized_scene_binding()
    support = SupportSurfaceAffordanceBinding(
        entity_id="cube_support",
        parent_id="rubiks_cube",
        native_name="support",
        is_default=True,
    )
    configured = replace(binding, support_surfaces=(support,))

    declared = configured.declare().lookup(SceneAffordanceRef("cube_support"))
    live = configured.build(_RigidizedSimulation()).lookup(
        SceneAffordanceRef("cube_support")
    )

    assert declared.parent == SceneObjectRef("rubiks_cube")
    assert live.parent == SceneObjectRef("rubiks_cube")


def test_rigidized_articulation_binding_rejects_fixed_root() -> None:
    simulation = _RigidizedSimulation(_RigidizedArticulation(fixed_base=True))

    with pytest.raises(ValueError, match="floating"):
        _rigidized_scene_binding().build(simulation)  # type: ignore[arg-type]


def test_rigidized_articulation_binding_requires_coincident_limits() -> None:
    simulation = _RigidizedSimulation(_RigidizedArticulation(limits=(-1.0, 1.0)))

    with pytest.raises(ValueError, match="coincident"):
        _rigidized_scene_binding().build(simulation)  # type: ignore[arg-type]


def test_rigidized_articulation_limit_lock_uses_fixed_safety_epsilon() -> None:
    simulation = _RigidizedSimulation(_RigidizedArticulation(limits=(-4.0e-4, 4.0e-4)))

    with pytest.raises(ValueError, match="coincident"):
        _rigidized_scene_binding().build(simulation)  # type: ignore[arg-type]


def test_rigidized_articulation_binding_fails_before_returning_invalid_geometry() -> (
    None
):
    vertices = torch.tensor(
        ((float("nan"), 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
        dtype=torch.float32,
    )
    simulation = _RigidizedSimulation(_RigidizedArticulation(vertices=vertices))

    with pytest.raises(ValueError, match="finite"):
        _rigidized_scene_binding().build(simulation)  # type: ignore[arg-type]


def _profile_binding() -> SimulationRobotSkillProfileBinding:
    """Build one manipulation profile declaration."""
    return SimulationRobotSkillProfileBinding(
        profile_id="test_robot",
        resources=(
            ControlPartResourceBinding(
                resource_id="manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part="hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="parallel_gripper",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="parallel_gripper",
                control_part="hand",
                commands={"open": (0.0,), "grasp": (1.0,)},
            ),
        ),
        defaults={"pick_up": {"primary": "manipulator"}},
        presets=(
            SkillPolicyPreset(
                "safe",
                action_option_templates={"pick": PickUpOptions()},
                effect_assurance=EffectAssurance.PROJECTED,
            ),
        ),
        default_preset="safe",
    )


def test_scene_binding_builds_existing_registry_contracts() -> None:
    simulation = _Simulation()

    registry = _scene_binding().build(simulation)  # type: ignore[arg-type]
    snapshot = registry.make_scene_provider().snapshot(
        timestamp=0.0,
        env_ids=torch.tensor((0, 1), dtype=torch.long),
    )

    assert registry.resolve("perceived_cube") == SceneObjectRef("cube")
    assert registry.resolve("drawer") == SceneArticulationRef("drawer")
    assert registry.resolve("drawer_handle_link") == SceneLinkRef("drawer_handle_link")
    grasp_ref = registry.resolve_affordance(
        "cube",
        capability=GRASP_AFFORDANCE_CAPABILITY,
    )
    assert grasp_ref == SceneAffordanceRef("cube_grasp")
    grasp = registry.lookup(grasp_ref).affordance
    assert isinstance(grasp, AntipodalAffordance)
    assert grasp.mesh_vertices is not None and grasp.mesh_vertices.shape == (3, 3)
    assert torch.equal(
        snapshot.articulation_joints[("drawer", "drawer_slide")].position,
        simulation.articulation.qpos,
    )
    assert torch.equal(
        snapshot.entities["drawer_handle_link"].pose,
        simulation.articulation.link_pose,
    )
    support_ref = registry.resolve_affordance(
        "cube",
        capability=PLACE_ON_AFFORDANCE_CAPABILITY,
    )
    support = registry.lookup(support_ref).affordance
    assert type(support) is SupportSurfaceAffordance
    assert support.minimum_confidence == pytest.approx(0.7)
    assert torch.equal(
        snapshot.entities[support_ref.entity_id].pose[:, 2, 3],
        torch.full((_BATCH_SIZE,), 0.25),
    )
    container_ref = registry.resolve_affordance(
        "drawer_handle_link",
        capability=PLACE_IN_AFFORDANCE_CAPABILITY,
    )
    container = registry.lookup(container_ref).affordance
    assert type(container) is ContainerAffordance
    assert container.minimum_confidence == pytest.approx(0.8)
    assert torch.allclose(
        snapshot.entities[container_ref.entity_id].pose[:, 0, 3],
        torch.tensor((0.4, 0.5)),
    )


def test_scene_manifest_uses_live_float32_pose_precision() -> None:
    """Static preflight and live registry agree for decimal placement poses."""
    binding = _scene_binding()
    support = replace(
        binding.support_surfaces[0],
        object_target_pose=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.16,
            0.0,
            0.0,
            1.0,
            0.03,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    configured = replace(binding, support_surfaces=(support,))

    declared = configured.declare()
    live = SceneManifest.from_registry(configured.build(_Simulation()))

    assert declared.entries == live.entries


def test_axis_aligned_grasp_binding_preserves_mesh_and_local_axis() -> None:
    """Configured pouring geometry remains a valid antipodal grasp payload."""
    binding = _scene_binding()
    axis_grasp = replace(
        binding.antipodal_grasps[0],
        internal_axis=(1.0, 0.0, 0.0),
    )
    configured = replace(binding, antipodal_grasps=(axis_grasp,))

    manifest_grasp = configured.declare().lookup(
        "cube_grasp",
        expected_type=SceneAffordanceRef,
    )
    live_grasp = configured.build(_Simulation()).lookup("cube_grasp").affordance

    assert manifest_grasp.affordance_payload_type is AxisAlignAffordance
    assert type(live_grasp) is AxisAlignAffordance
    assert torch.equal(live_grasp.internal_axis, torch.tensor((1.0, 0.0, 0.0)))
    assert live_grasp.mesh_vertices is not None


def test_scene_binding_fails_closed_on_missing_native_entity() -> None:
    binding = _scene_binding()
    missing = replace(
        binding.rigid_objects[0],
        simulation_uid="missing_cube",
    )

    with pytest.raises(KeyError, match="missing_cube"):
        replace(binding, rigid_objects=(missing,)).build(  # type: ignore[arg-type]
            _Simulation()
        )


def test_scene_binding_fails_closed_on_missing_native_link() -> None:
    binding = _scene_binding()
    missing = replace(binding.links[0], native_link_name="missing_handle")

    with pytest.raises(KeyError, match="missing_handle"):
        replace(binding, links=(missing,)).build(  # type: ignore[arg-type]
            _Simulation()
        )


def test_robot_profile_binding_builds_existing_profile_contracts() -> None:
    profile = _profile_binding().build(_Robot())  # type: ignore[arg-type]

    resource = profile.resources["manipulator"]
    motion = resource.endpoints["motion"]
    grasp = resource.endpoints["grasp"]
    assert motion.control_part == "arm"
    assert motion.capabilities == frozenset({CARTESIAN_POSE_CAPABILITY})
    assert grasp.control_part == "hand"
    assert grasp.command_profile == "parallel_gripper"
    command = profile.command_profiles["parallel_gripper"].commands["grasp"]
    assert torch.equal(command.positions, torch.tensor((1.0,)))
    assert profile.defaults["pick_up"].resources == {"primary": "manipulator"}


def test_profile_binding_owns_direct_core_resource() -> None:
    endpoint = _MobileEndpoint(
        controller_id="base_controller",
        capabilities=frozenset({"motion.base.velocity"}),
    )
    resource = RobotResource(
        resource_id="mobile_base",
        endpoints={"motion": endpoint},
    )
    binding = SimulationRobotSkillProfileBinding(
        profile_id="mobile_robot",
        resources=(resource,),
    )

    profile = binding.build(object())  # type: ignore[arg-type]
    built_resource = profile.resources["mobile_base"]
    built_endpoint = built_resource.endpoints["motion"]

    assert isinstance(built_endpoint, _MobileEndpoint)
    assert built_endpoint is not endpoint
    assert built_endpoint.controller_id == "base_controller"
    assert built_resource.members == ()


def test_whole_body_control_part_remains_supported_and_strict() -> None:
    class WholeBodyRobot:
        control_parts = {"whole_body": object()}

        def get_joint_ids(self, *, name: str) -> list[int]:
            assert name == "whole_body"
            return [0, 1, 2, 3]

    binding = SimulationRobotSkillProfileBinding(
        profile_id="whole_body_robot",
        resources=(
            ControlPartResourceBinding(
                resource_id="body",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="whole_body",
                        capabilities=frozenset({"motion.whole_body"}),
                    ),
                ),
            ),
        ),
    )

    profile = binding.build(WholeBodyRobot())  # type: ignore[arg-type]
    endpoint = profile.resources["body"].endpoints["motion"]

    assert endpoint.control_part == "whole_body"
    assert endpoint.capabilities == frozenset({"motion.whole_body"})


def test_robot_profile_binding_fails_closed_on_missing_control_part() -> None:
    binding = _profile_binding()
    resource = binding.resources[0]
    missing_endpoint = replace(
        resource.endpoints[0],
        control_part="missing_arm",
    )

    with pytest.raises(KeyError, match="missing_arm"):
        replace(
            binding,
            resources=(
                replace(
                    resource,
                    endpoints=(missing_endpoint, resource.endpoints[1]),
                ),
            ),
        ).build(
            _Robot()
        )  # type: ignore[arg-type]


def test_robot_profile_binding_rejects_wrong_command_width() -> None:
    binding = _profile_binding()
    invalid = replace(
        binding.command_presets[0],
        commands={"open": (0.0, 0.0), "grasp": (1.0, 1.0)},
    )

    with pytest.raises(ValueError, match="has 2 positions.*has 1 joints"):
        replace(binding, command_presets=(invalid,)).build(  # type: ignore[arg-type]
            _Robot()
        )
