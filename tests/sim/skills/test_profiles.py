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

"""Tests for generic robot resources and declarative skill profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    BUILTIN_ACTION_TYPES,
    CARTESIAN_POSE_CAPABILITY,
    ControlCommand,
    ControlPartCommandProfile,
    DisjointSlotEndpoints,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    GRASP_COMMAND,
    INVERSE_KINEMATICS_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
    JointPositionCommand,
    JointPositionGoal,
    MotionPolicy,
    OPEN_COMMAND,
    PickUpOptions,
    ResolvedActionRequest,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from embodichain.lab.sim.atomic_actions.bindings import (
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.tracking import (
    JOINT_POSITION_CHANNEL,
    EndpointTrackingFeedbackAddress,
    JointPositionTrackingMetric,
    TrackingPolicy,
)
from embodichain.lab.sim.skills import (
    AmbiguousSkillBindingError,
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTACT_EFFECT_CHANNEL,
    ControlPartEndpoint,
    ControlPartEndpointAdapter,
    ControlPartEvidenceAddress,
    EffectEvidenceSourceRef,
    EffectMonitorRef,
    EndpointResolution,
    FORCE_EFFECT_CHANNEL,
    JOINT_STATE_EFFECT_CHANNEL,
    POSE_RELATION_EFFECT_CHANNEL,
    ProfileValidationError,
    ResourceBinding,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
    UnsupportedSkillError,
)

_JOINT_IDS = {
    "left_arm": [0, 1],
    "left_hand": [2],
    "right_arm": [3, 4],
    "right_hand": [5],
    "base": [6, 7],
    "torso": [8],
    "full_body": [0, 1, 3, 4, 6, 7, 8],
}

_MOTION_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
        INVERSE_KINEMATICS_CAPABILITY,
        JOINT_POSITION_CAPABILITY,
    }
)


def _command_profiles() -> dict[str, ControlPartCommandProfile]:
    return {
        hand: ControlPartCommandProfile.joint_positions(
            open=torch.tensor([0.0]),
            grasp=torch.tensor([1.0]),
        )
        for hand in ("left_hand", "right_hand")
    }


def _engine(
    *,
    control_profiles: dict[str, ControlPartCommandProfile] | None = None,
    load_builtins: bool = True,
) -> AtomicActionEngine:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 9
    robot.control_parts = {name: object() for name in _JOINT_IDS}
    robot.get_qpos.return_value = torch.zeros(2, 9)
    robot.get_qvel.return_value = torch.zeros(2, 9)
    robot.get_joint_ids.side_effect = lambda name: list(_JOINT_IDS[name])
    robot.get_solver.side_effect = lambda name=None: (
        object() if name in {"left_arm", "right_arm"} else None
    )
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    return AtomicActionEngine(
        generator,
        control_profiles=control_profiles,
        load_builtins=load_builtins,
    )


def _resources(*, include_right: bool = True) -> dict[str, RobotResource]:
    resources = {
        "left_arm": RobotResource(
            "left_arm",
            endpoints={"control": ControlPartEndpoint("left_arm")},
        ),
        "left_hand": RobotResource(
            "left_hand",
            endpoints={"control": ControlPartEndpoint("left_hand")},
        ),
        "left_actor": RobotResource(
            "left_actor",
            endpoints={
                "motion": ControlPartEndpoint(
                    "left_arm", capabilities=_MOTION_CAPABILITIES
                ),
                "grasp": ControlPartEndpoint(
                    "left_hand", capabilities=frozenset({GRASP_CAPABILITY})
                ),
            },
            members=("left_arm", "left_hand"),
        ),
        "base": RobotResource(
            "base",
            endpoints={
                "motion": ControlPartEndpoint(
                    "base", capabilities=frozenset({"motion.base.se2"})
                )
            },
        ),
        "torso": RobotResource(
            "torso",
            endpoints={"control": ControlPartEndpoint("torso")},
        ),
    }
    if include_right:
        resources.update(
            {
                "right_arm": RobotResource(
                    "right_arm",
                    endpoints={"control": ControlPartEndpoint("right_arm")},
                ),
                "right_hand": RobotResource(
                    "right_hand",
                    endpoints={"control": ControlPartEndpoint("right_hand")},
                ),
                "right_actor": RobotResource(
                    "right_actor",
                    endpoints={
                        "motion": ControlPartEndpoint(
                            "right_arm", capabilities=_MOTION_CAPABILITIES
                        ),
                        "grasp": ControlPartEndpoint(
                            "right_hand",
                            capabilities=frozenset({GRASP_CAPABILITY}),
                        ),
                    },
                    members=("right_arm", "right_hand"),
                ),
            }
        )
    whole_body_members = ["base", "torso", "left_arm"]
    if include_right:
        whole_body_members.append("right_arm")
    if include_right:
        resources["whole_body"] = RobotResource(
            "whole_body",
            endpoints={
                "motion": ControlPartEndpoint(
                    "full_body", capabilities=frozenset({"motion.whole_body"})
                )
            },
            members=tuple(whole_body_members),
        )
    return resources


def _profile(
    *,
    defaults: dict[str, ResourceBinding] | None = None,
    resources: dict[str, RobotResource] | None = None,
    command_profiles: dict[str, ControlPartCommandProfile] | None = None,
) -> RobotSkillProfile:
    return RobotSkillProfile(
        profile_id="test_robot",
        resources=_resources() if resources is None else resources,
        command_profiles=(
            _command_profiles() if command_profiles is None else command_profiles
        ),
        defaults={} if defaults is None else defaults,
    )


class _WholeBodyAction(AtomicAction[JointPositionGoal, ActionOptions]):
    skill_id: ClassVar[str] = "whole_body_reach"
    GoalType: ClassVar[type] = JointPositionGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                "body",
                endpoints=(
                    SkillEndpointRequirement(
                        "motion",
                        capabilities=frozenset({"motion.whole_body"}),
                    ),
                ),
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        raise NotImplementedError


class _NavigateAction(AtomicAction[JointPositionGoal, ActionOptions]):
    skill_id: ClassVar[str] = "navigate"
    GoalType: ClassVar[type] = JointPositionGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                "body",
                endpoints=(
                    SkillEndpointRequirement(
                        "motion",
                        capabilities=frozenset({"motion.base.se2"}),
                    ),
                ),
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _BaseVelocityEndpoint(ResourceEndpoint):
    """Future non-joint endpoint used to prove the resource API stays generic."""

    controller_id: str
    claim_id: str | None = None


@dataclass(frozen=True, slots=True)
class _MutableMetadataEndpoint(ResourceEndpoint):
    """Endpoint with mutable metadata used to verify ownership snapshots."""

    controller_id: str
    aliases: list[str]


@dataclass(frozen=True, slots=True)
class _BaseVelocityTarget(RuntimeEndpointTarget):
    """Typed runtime destination for the test mobile controller."""

    controller_id: str

    @property
    def transport_id(self) -> str:
        """Return the fake base-velocity transport kind."""
        return "test.base_velocity"

    @property
    def target_id(self) -> str:
        """Return the addressed controller ID."""
        return self.controller_id


@dataclass(frozen=True, slots=True)
class _MutableRuntimeTarget(RuntimeEndpointTarget):
    """Target with nested mutable data used to prove snapshot ownership."""

    controller_id: str
    aliases: list[str]

    @property
    def transport_id(self) -> str:
        """Return the fake mutable-target transport kind."""
        return "test.mutable"

    @property
    def target_id(self) -> str:
        """Return the addressed controller ID."""
        return self.controller_id


@dataclass(frozen=True, slots=True)
class _TwistCommand(ControlCommand):
    """Test-only non-joint command for a mobile controller."""

    value: tuple[float, float, float]

    def snapshot(self) -> _TwistCommand:
        """Return an independently owned immutable command."""
        return _TwistCommand(tuple(self.value))

    def equivalent_to(self, other: ControlCommand) -> bool:
        """Return whether another twist command has the same value."""
        return isinstance(other, _TwistCommand) and self.value == other.value


class _BaseVelocityEndpointAdapter(ResourceEndpointAdapter):
    """Resolve the test mobile controller without profile-resolver changes."""

    adapter_id: ClassVar[str] = "test.base_velocity"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = _BaseVelocityEndpoint

    def resolve(
        self,
        endpoint: ResourceEndpoint,
        *,
        engine: AtomicActionEngine,
    ) -> EndpointResolution:
        """Resolve one mobile controller to a generic exclusive claim."""
        del engine
        assert isinstance(endpoint, _BaseVelocityEndpoint)
        claim_id = (
            endpoint.controller_id if endpoint.claim_id is None else endpoint.claim_id
        )
        return EndpointResolution(
            runtime_target=_BaseVelocityTarget(endpoint.controller_id),
            command_profile_key=endpoint.controller_id,
            claim_tokens=frozenset({f"controller:{claim_id}"}),
        )


class _VelocityNavigateAction(AtomicAction[JointPositionGoal, ActionOptions]):
    """Semantic test skill consuming a non-core controller endpoint."""

    skill_id: ClassVar[str] = "navigate_velocity"
    GoalType: ClassVar[type] = JointPositionGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                "body",
                endpoints=(
                    SkillEndpointRequirement(
                        "motion",
                        capabilities=frozenset({"motion.base.velocity"}),
                        required_commands={"stop": _TwistCommand},
                    ),
                ),
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        raise NotImplementedError


def test_engine_skills_only_exposes_visible_explicit_installed_contracts() -> None:
    engine = _engine(control_profiles=_command_profiles())
    expected = {
        action_type.skill_id
        for action_type in BUILTIN_ACTION_TYPES
        if action_type.agent_visible
    }

    assert set(engine.skills) == expected
    assert "move_joints" in engine.actions
    assert "move_joints" not in engine.skills


def test_new_skill_subclass_must_redeclare_binding_contract() -> None:
    base_contract = BUILTIN_ACTION_TYPES[0].descriptor().binding_contract

    class Derived(BUILTIN_ACTION_TYPES[0]):
        skill_id: ClassVar[str] = "derived_without_explicit_contract"

    assert base_contract is not None
    assert Derived.descriptor().binding_contract is None


def test_profile_owns_input_mappings_and_command_tensors() -> None:
    resources = _resources()
    open_positions = torch.tensor([0.0])
    profiles = {
        "left_hand": ControlPartCommandProfile.joint_positions(open=open_positions)
    }
    profile = _profile(resources=resources, command_profiles=profiles)

    resources.clear()
    profiles.clear()
    open_positions.fill_(9.0)

    assert "left_actor" in profile.resources
    command = profile.command_profiles["left_hand"].commands[OPEN_COMMAND]
    assert isinstance(command, JointPositionCommand)
    assert command.positions.tolist() == [0.0]


def test_profile_owns_custom_endpoint_nested_payloads() -> None:
    source_aliases = ["base"]
    resource = RobotResource(
        "mobile_base",
        endpoints={
            "motion": _MutableMetadataEndpoint(
                "base_controller",
                aliases=source_aliases,
            )
        },
    )
    profile = RobotSkillProfile(
        "mobile",
        resources={"mobile_base": resource},
    )

    source_aliases.append("source_mutation")
    resource_endpoint = resource.endpoints["motion"]
    assert isinstance(resource_endpoint, _MutableMetadataEndpoint)
    resource_endpoint.aliases.append("resource_mutation")
    profile_endpoint = profile.resources["mobile_base"].endpoints["motion"]
    assert isinstance(profile_endpoint, _MutableMetadataEndpoint)

    assert profile_endpoint.aliases == ["base"]


def test_endpoint_resolution_requires_a_runtime_target() -> None:
    with pytest.raises(TypeError, match="runtime_target"):
        EndpointResolution(
            runtime_target=None,  # type: ignore[arg-type]
            exclusive=False,
        )


def test_endpoint_resolution_owns_runtime_target_snapshot() -> None:
    aliases = ["base"]
    target = _MutableRuntimeTarget("base_controller", aliases)

    resolution = EndpointResolution(runtime_target=target, exclusive=False)
    aliases.append("source_mutation")
    target.aliases.append("target_mutation")

    assert resolution.runtime_target is not target
    assert type(resolution.runtime_target) is _MutableRuntimeTarget
    assert resolution.runtime_target.aliases == ["base"]


def test_endpoint_resolution_owns_and_freezes_effect_sources() -> None:
    source = EffectEvidenceSourceRef(
        "test.provider",
        "1",
        ControlPartEvidenceAddress("left_arm", POSE_RELATION_EFFECT_CHANNEL),
    )
    sources = {POSE_RELATION_EFFECT_CHANNEL: source}

    resolution = EndpointResolution(
        runtime_target=_BaseVelocityTarget("base_controller"),
        task_state_key="mobile_actor",
        effect_sources=sources,
        exclusive=False,
    )
    sources.clear()

    assert resolution.task_state_key == "mobile_actor"
    assert tuple(resolution.effect_sources) == (POSE_RELATION_EFFECT_CHANNEL,)
    assert resolution.effect_sources[POSE_RELATION_EFFECT_CHANNEL] is not source
    assert resolution.effect_sources[POSE_RELATION_EFFECT_CHANNEL].address == (
        ControlPartEvidenceAddress("left_arm", POSE_RELATION_EFFECT_CHANNEL)
    )
    with pytest.raises(TypeError):
        resolution.effect_sources["new"] = source  # type: ignore[index]


@pytest.mark.parametrize("returns_self", [False, True])
def test_endpoint_resolution_rejects_invalid_target_snapshot(
    returns_self: bool,
) -> None:
    @dataclass(frozen=True, slots=True)
    class InvalidSnapshotTarget(RuntimeEndpointTarget):
        controller_id: str

        @property
        def transport_id(self) -> str:
            return "test.invalid_snapshot"

        @property
        def target_id(self) -> str:
            return self.controller_id

        def snapshot(self) -> RuntimeEndpointTarget:
            if returns_self:
                return self
            return _BaseVelocityTarget(self.controller_id)

    with pytest.raises(TypeError, match="same target type"):
        EndpointResolution(
            runtime_target=InvalidSnapshotTarget("base_controller"),
            exclusive=False,
        )


def test_resource_graph_rejects_unknown_member_and_cycle() -> None:
    with pytest.raises(ValueError, match="unknown members"):
        RobotSkillProfile(
            "unknown_member",
            resources={
                "group": RobotResource("group", members=("missing",)),
            },
        )

    with pytest.raises(ValueError, match="contains a cycle"):
        RobotSkillProfile(
            "cycle",
            resources={
                "a": RobotResource("a", members=("b",)),
                "b": RobotResource("b", members=("a",)),
            },
        )


def test_identifier_sets_do_not_accept_one_string_as_characters() -> None:
    with pytest.raises(TypeError, match="not a string"):
        ControlPartEndpoint("left_arm", capabilities="motion.cartesian_pose")
    with pytest.raises(TypeError, match="not a string"):
        RobotResource(
            "left_arm",
            endpoints={"control": ControlPartEndpoint("left_arm")},
            members="left_arm",
        )
    with pytest.raises(TypeError, match="iterable of endpoint IDs"):
        DisjointSlotEndpoints("motion")


def test_slot_constraint_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown endpoints"):
        SkillResourceSlot(
            "primary",
            endpoints=(SkillEndpointRequirement("motion"),),
            constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
        )


def test_bind_rejects_unknown_control_part() -> None:
    resources = _resources()
    resources["camera_gimbal"] = RobotResource(
        "camera_gimbal",
        endpoints={"motion": ControlPartEndpoint("missing")},
    )

    with pytest.raises(ProfileValidationError, match="unknown control part 'missing'"):
        _profile(resources=resources).bind(
            _engine(control_profiles=_command_profiles())
        )


def test_resource_graph_accepts_extensible_endpoint_before_adapter_installation() -> (
    None
):
    resource = RobotResource(
        "mobile_base",
        endpoints={
            "motion": _BaseVelocityEndpoint(
                "base_controller",
                capabilities=frozenset({"motion.base.velocity"}),
            )
        },
    )
    profile = RobotSkillProfile("mobile", resources={"mobile_base": resource})

    assert (
        profile.resources["mobile_base"].endpoints["motion"]
        == resource.endpoints["motion"]
    )
    with pytest.raises(ProfileValidationError, match="ResourceEndpointAdapter"):
        profile.bind(_engine(control_profiles={}, load_builtins=False))


def test_custom_endpoint_adapter_resolves_commands_and_physical_claim() -> None:
    resource = RobotResource(
        "mobile_base",
        endpoints={
            "motion": _BaseVelocityEndpoint(
                "base_velocity",
                capabilities=frozenset({"motion.base.velocity"}),
            )
        },
    )
    profile = RobotSkillProfile(
        "mobile",
        resources={"mobile_base": resource},
        command_profiles={
            "base_velocity": ControlPartCommandProfile(
                commands={"stop": _TwistCommand((0.0, 0.0, 0.0))}
            )
        },
    )
    engine = _engine(control_profiles={}, load_builtins=False)
    engine.register(_VelocityNavigateAction())

    bound = engine.bind_skill_profile(
        profile,
        endpoint_adapters={_BaseVelocityEndpoint: _BaseVelocityEndpointAdapter()},
    )
    resolved = bound.resolve("navigate_velocity")
    endpoint = resolved.resources["body"].endpoints["motion"]
    binding_endpoint = resolved.action_binding.endpoint("body", "motion")

    assert endpoint.adapter_id == "test.base_velocity"
    assert isinstance(endpoint.runtime_target, _BaseVelocityTarget)
    assert isinstance(endpoint.commands["stop"], _TwistCommand)
    assert resolved.claim.claim_tokens == frozenset({"controller:base_velocity"})
    assert resolved.action_binding.owner_id == engine.binding_owner_id
    assert binding_endpoint.resource_id == "mobile_base"
    assert binding_endpoint.require_target(_BaseVelocityTarget).controller_id == (
        "base_velocity"
    )
    assert isinstance(binding_endpoint.command("stop"), _TwistCommand)


def test_custom_endpoint_joint_claim_survives_action_binding_lowering() -> None:
    class JointClaimAdapter(_BaseVelocityEndpointAdapter):
        """Attach robot-joint ownership to a non-joint runtime target."""

        adapter_id: ClassVar[str] = "test.base_velocity_joint_claim"

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del engine
            assert isinstance(endpoint, _BaseVelocityEndpoint)
            return EndpointResolution(
                runtime_target=_BaseVelocityTarget(endpoint.controller_id),
                command_profile_key=endpoint.controller_id,
                joint_ids=(6, 7),
            )

    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={
                    "motion": _BaseVelocityEndpoint(
                        "base_velocity",
                        capabilities=frozenset({"motion.base.velocity"}),
                    )
                },
            )
        },
        command_profiles={
            "base_velocity": ControlPartCommandProfile(
                commands={"stop": _TwistCommand((0.0, 0.0, 0.0))}
            )
        },
    )
    engine = _engine(control_profiles={}, load_builtins=False)
    engine.register(_VelocityNavigateAction())
    bound = engine.bind_skill_profile(
        profile,
        endpoint_adapters={_BaseVelocityEndpoint: JointClaimAdapter()},
    )

    binding_endpoint = bound.resolve("navigate_velocity").action_binding.endpoint(
        "body", "motion"
    )

    assert binding_endpoint.joint_ids == (6, 7)


def test_custom_endpoint_joint_claim_must_fit_robot_dof() -> None:
    class OutOfRangeJointClaimAdapter(_BaseVelocityEndpointAdapter):
        adapter_id: ClassVar[str] = "test.out_of_range_joint_claim"

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del engine
            assert isinstance(endpoint, _BaseVelocityEndpoint)
            return EndpointResolution(
                runtime_target=_BaseVelocityTarget(endpoint.controller_id),
                joint_ids=(9,),
            )

    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={
                    "motion": _BaseVelocityEndpoint(
                        "base_velocity",
                        capabilities=frozenset({"motion.base.velocity"}),
                    )
                },
            )
        },
    )

    with pytest.raises(ProfileValidationError, match="outside robot DOF 9"):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={
                _BaseVelocityEndpoint: OutOfRangeJointClaimAdapter(),
            },
        )


def test_engine_constructor_forwards_custom_endpoint_adapters() -> None:
    source = _engine(control_profiles={}, load_builtins=False)
    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={
                    "motion": _BaseVelocityEndpoint(
                        "base_velocity",
                        capabilities=frozenset({"motion.base.velocity"}),
                    )
                },
            )
        },
    )

    engine = AtomicActionEngine(
        source.motion_generator,
        load_builtins=False,
        skill_profile=profile,
        endpoint_adapters={_BaseVelocityEndpoint: _BaseVelocityEndpointAdapter()},
    )

    assert engine.skill_profile is not None
    assert engine.skill_profile.resources["mobile_base"].claim.claim_tokens == (
        frozenset({"controller:base_velocity"})
    )


def test_custom_endpoint_claim_tokens_protect_distinct_leaf_aliases() -> None:
    endpoint = _BaseVelocityEndpoint(
        "base_velocity",
        capabilities=frozenset({"motion.base.velocity"}),
    )
    profile = RobotSkillProfile(
        "aliased_mobile",
        resources={
            "base_a": RobotResource("base_a", endpoints={"motion": endpoint}),
            "base_b": RobotResource("base_b", endpoints={"motion": endpoint}),
        },
    )

    with pytest.raises(ProfileValidationError, match="adapter claims"):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: _BaseVelocityEndpointAdapter()},
        )


def test_distinct_physical_leaves_cannot_share_one_runtime_target() -> None:
    profile = RobotSkillProfile(
        "duplicate_runtime_target",
        resources={
            "base_a": RobotResource(
                "base_a",
                endpoints={
                    "motion": _BaseVelocityEndpoint("shared", claim_id="base_a")
                },
            ),
            "base_b": RobotResource(
                "base_b",
                endpoints={
                    "motion": _BaseVelocityEndpoint("shared", claim_id="base_b")
                },
            ),
        },
    )

    with pytest.raises(ProfileValidationError, match="share runtime targets"):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: _BaseVelocityEndpointAdapter()},
        )


def test_endpoint_adapter_cannot_omit_runtime_target() -> None:
    class MissingRuntimeTargetAdapter(ResourceEndpointAdapter):
        adapter_id: ClassVar[str] = "test.missing_runtime_target"
        endpoint_type: ClassVar[type[ResourceEndpoint]] = _BaseVelocityEndpoint

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del endpoint, engine
            return EndpointResolution(
                runtime_target=None,  # type: ignore[arg-type]
                exclusive=False,
            )

    profile = RobotSkillProfile(
        "missing_runtime_target",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={"motion": _BaseVelocityEndpoint("base_velocity")},
            )
        },
    )

    with pytest.raises(
        ProfileValidationError,
        match="test.missing_runtime_target.*mobile_base.*motion.*runtime_target",
    ):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: MissingRuntimeTargetAdapter()},
        )


def test_exclusive_custom_endpoint_requires_a_physical_claim() -> None:
    class EmptyClaimAdapter(ResourceEndpointAdapter):
        adapter_id: ClassVar[str] = "test.empty_claim"
        endpoint_type: ClassVar[type[ResourceEndpoint]] = _BaseVelocityEndpoint

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del endpoint, engine
            return EndpointResolution(
                runtime_target=_BaseVelocityTarget("base_velocity")
            )

    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={"motion": _BaseVelocityEndpoint("base_velocity")},
            )
        },
    )

    with pytest.raises(
        ProfileValidationError,
        match="test.empty_claim.*mobile_base.*motion.*joint_ids or claim_tokens",
    ):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: EmptyClaimAdapter()},
        )


def test_nonexclusive_custom_endpoint_may_omit_a_physical_claim() -> None:
    class VirtualEndpointAdapter(ResourceEndpointAdapter):
        adapter_id: ClassVar[str] = "test.virtual"
        endpoint_type: ClassVar[type[ResourceEndpoint]] = _BaseVelocityEndpoint

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del endpoint, engine
            return EndpointResolution(
                runtime_target=_BaseVelocityTarget("virtual"),
                exclusive=False,
            )

    profile = RobotSkillProfile(
        "virtual",
        resources={
            "virtual_channel": RobotResource(
                "virtual_channel",
                endpoints={"motion": _BaseVelocityEndpoint("virtual")},
            )
        },
    )

    bound = profile.bind(
        _engine(control_profiles={}, load_builtins=False),
        endpoint_adapters={_BaseVelocityEndpoint: VirtualEndpointAdapter()},
    )

    assert not bound.resources["virtual_channel"].endpoints["motion"].exclusive


def test_endpoint_adapter_registration_validates_declared_type() -> None:
    class MissingMetadataAdapter(ResourceEndpointAdapter):
        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del endpoint, engine
            return EndpointResolution(
                runtime_target=_BaseVelocityTarget("base_velocity"),
                exclusive=False,
            )

    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={"motion": _BaseVelocityEndpoint("base_velocity")},
            )
        },
    )

    with pytest.raises(TypeError, match="must declare.*endpoint_type"):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: MissingMetadataAdapter()},
        )


def test_builtin_control_part_adapter_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="cannot be overridden"):
        _profile().bind(
            _engine(control_profiles=_command_profiles()),
            endpoint_adapters={ControlPartEndpoint: ControlPartEndpointAdapter()},
        )


def test_endpoint_adapter_must_return_endpoint_resolution() -> None:
    class WrongReturnAdapter(ResourceEndpointAdapter):
        adapter_id: ClassVar[str] = "test.wrong_return"
        endpoint_type: ClassVar[type[ResourceEndpoint]] = _BaseVelocityEndpoint

        def resolve(
            self,
            endpoint: ResourceEndpoint,
            *,
            engine: AtomicActionEngine,
        ) -> EndpointResolution:
            del endpoint, engine
            return object()  # type: ignore[return-value]

    profile = RobotSkillProfile(
        "mobile",
        resources={
            "mobile_base": RobotResource(
                "mobile_base",
                endpoints={"motion": _BaseVelocityEndpoint("base_velocity")},
            )
        },
    )

    with pytest.raises(ProfileValidationError, match="expected EndpointResolution"):
        profile.bind(
            _engine(control_profiles={}, load_builtins=False),
            endpoint_adapters={_BaseVelocityEndpoint: WrongReturnAdapter()},
        )


def test_bind_rejects_overlapping_physical_leaves() -> None:
    resources = _resources()
    resources["left_arm_alias"] = RobotResource(
        "left_arm_alias",
        endpoints={"control": ControlPartEndpoint("left_arm")},
    )

    with pytest.raises(ProfileValidationError, match="overlap on robot joints"):
        _profile(resources=resources).bind(
            _engine(control_profiles=_command_profiles())
        )


def test_bind_rejects_composite_endpoint_outside_member_claim() -> None:
    resources = _resources()
    resources["bad_composite"] = RobotResource(
        "bad_composite",
        endpoints={"motion": ControlPartEndpoint("right_arm")},
        members=("left_arm",),
    )

    with pytest.raises(ProfileValidationError, match="not claimed by its members"):
        _profile(resources=resources).bind(
            _engine(control_profiles=_command_profiles())
        )


def test_bind_rejects_profile_commands_not_installed_on_engine() -> None:
    with pytest.raises(ProfileValidationError, match="is not installed"):
        _profile().bind(_engine(control_profiles={}))


def test_explicit_endpoint_command_profile_must_exist() -> None:
    profile = RobotSkillProfile(
        "missing_commands",
        resources={
            "arm": RobotResource(
                "arm",
                endpoints={
                    "motion": ControlPartEndpoint(
                        "left_arm",
                        command_profile="missing_profile",
                    )
                },
            )
        },
    )

    with pytest.raises(ProfileValidationError, match="required command profile"):
        profile.bind(_engine(control_profiles={}, load_builtins=False))


def test_bind_rejects_engine_command_payload_that_differs_from_profile() -> None:
    engine_profiles = _command_profiles()
    engine_profiles["left_hand"] = ControlPartCommandProfile.joint_positions(
        open=torch.tensor([0.5]),
        grasp=torch.tensor([1.0]),
    )

    with pytest.raises(ProfileValidationError, match="not semantically equivalent"):
        _profile().bind(_engine(control_profiles=engine_profiles))


def test_profile_rejects_conflicting_endpoint_command_profiles() -> None:
    resources = {
        "hand": RobotResource(
            "hand",
            endpoints={
                "first": ControlPartEndpoint(
                    "left_hand",
                    command_profile="first_hand",
                ),
                "second": ControlPartEndpoint(
                    "left_hand",
                    command_profile="second_hand",
                ),
            },
        )
    }
    with pytest.raises(ValueError, match="non-equivalent 'grasp'"):
        RobotSkillProfile(
            "conflicting_commands",
            resources=resources,
            command_profiles={
                "first_hand": ControlPartCommandProfile.joint_positions(
                    grasp=torch.tensor([0.5])
                ),
                "second_hand": ControlPartCommandProfile.joint_positions(
                    grasp=torch.tensor([1.0])
                ),
            },
        )


def test_bind_rejects_joint_command_with_wrong_endpoint_dof() -> None:
    profiles = _command_profiles()
    profiles["left_hand"] = ControlPartCommandProfile.joint_positions(
        open=torch.zeros(2),
        grasp=torch.ones(2),
    )

    with pytest.raises(ProfileValidationError, match="2 joints, expected 1"):
        _profile(command_profiles=profiles).bind(_engine(control_profiles=profiles))


def test_profile_commands_must_be_broadcastable_across_environments() -> None:
    profiles = _command_profiles()
    profiles["left_hand"] = ControlPartCommandProfile.joint_positions(
        open=torch.zeros(2, 1),
        grasp=torch.ones(2, 1),
    )

    with pytest.raises(ProfileValidationError, match="must be one-dimensional"):
        _profile(command_profiles=profiles).bind(_engine(control_profiles=profiles))


def test_bind_rejects_unverified_standard_solver_capability() -> None:
    engine = _engine(control_profiles=_command_profiles())
    engine.robot.get_solver.side_effect = lambda name=None: None

    with pytest.raises(ProfileValidationError, match="has no configured solver"):
        _profile().bind(engine)


def test_unique_capability_binding_lowers_to_exact_action_binding() -> None:
    profile = _profile(resources=_resources(include_right=False))
    engine = _engine(control_profiles=_command_profiles())
    bound = profile.bind(engine)

    resolved = bound.resolve("pick_up")
    motion = resolved.action_binding.endpoint("primary", "motion")
    grasp = resolved.action_binding.endpoint("primary", "grasp")

    assert resolved.resource_ids == {"primary": "left_actor"}
    assert resolved.action_binding.owner_id == engine.binding_owner_id
    assert resolved.action_binding.endpoint_keys == (
        ("primary", "motion"),
        ("primary", "grasp"),
    )
    assert motion.require_target(JointPositionTarget).control_part == "left_arm"
    assert grasp.require_target(JointPositionTarget).control_part == "left_hand"
    assert motion.task_state_key == "left_actor"
    assert grasp.task_state_key == "left_actor"
    motion_tracking = motion.tracking_channel(JOINT_POSITION_CHANNEL)
    assert motion_tracking.source.provider_id == "planning_context.robot"
    assert motion_tracking.source.revision == "1"
    assert motion_tracking.projector.projector_id == "joint_position_payload"
    assert motion_tracking.projector.revision == "1"
    assert isinstance(
        motion_tracking.source.address,
        EndpointTrackingFeedbackAddress,
    )
    assert (
        motion_tracking.source.address.target.address_fingerprint
        == motion.target.address_fingerprint
    )
    resource = resolved.resources["primary"]
    motion_sources = resource.endpoints["motion"].effect_sources
    grasp_sources = resource.endpoints["grasp"].effect_sources
    assert set(motion_sources) == {
        POSE_RELATION_EFFECT_CHANNEL,
        JOINT_STATE_EFFECT_CHANNEL,
    }
    assert set(grasp_sources) == {
        POSE_RELATION_EFFECT_CHANNEL,
        JOINT_STATE_EFFECT_CHANNEL,
        CONTACT_EFFECT_CHANNEL,
        CONSTRAINT_EFFECT_CHANNEL,
        FORCE_EFFECT_CHANNEL,
    }
    assert motion_sources[POSE_RELATION_EFFECT_CHANNEL].address == (
        ControlPartEvidenceAddress("left_arm", POSE_RELATION_EFFECT_CHANNEL)
    )
    assert grasp_sources[CONSTRAINT_EFFECT_CHANNEL].address == (
        ControlPartEvidenceAddress("left_hand", CONSTRAINT_EFFECT_CHANNEL)
    )
    assert resolved.claim.leaf_resource_ids == frozenset({"left_arm", "left_hand"})
    assert resolved.claim.joint_ids == (0, 1, 2)


def test_ambiguous_binding_requires_complete_per_skill_default() -> None:
    engine = _engine(control_profiles=_command_profiles())
    bound = _profile().bind(engine)

    with pytest.raises(AmbiguousSkillBindingError, match="2 valid resource bindings"):
        bound.resolve("pick_up")

    selected = _profile(
        defaults={
            "pick_up": ResourceBinding({"primary": "right_actor"}),
        }
    ).bind(engine)
    assert selected.resolve("pick_up").resource_ids == {"primary": "right_actor"}


@pytest.mark.parametrize(
    "default",
    [
        ResourceBinding({}),
        ResourceBinding({"primary": "left_actor", "stale": "right_actor"}),
    ],
)
def test_profile_rejects_partial_or_extra_default_slots(
    default: ResourceBinding,
) -> None:
    with pytest.raises(ProfileValidationError, match="must cover exactly"):
        _profile(defaults={"pick_up": default}).bind(
            _engine(control_profiles=_command_profiles())
        )


def test_explicit_slot_selection_overrides_profile_default() -> None:
    bound = _profile(
        defaults={
            "pick_up": ResourceBinding({"primary": "left_actor"}),
        }
    ).bind(_engine(control_profiles=_command_profiles()))

    resolved = bound.resolve("pick_up", {"primary": "right_actor"})

    assert resolved.resource_ids == {"primary": "right_actor"}


def test_missing_required_command_filters_skill_and_reports_reason() -> None:
    profiles = _command_profiles()
    profiles = {
        name: ControlPartCommandProfile.joint_positions(grasp=torch.tensor([1.0]))
        for name in profiles
    }
    bound = _profile(command_profiles=profiles).bind(_engine(control_profiles=profiles))

    assert "pick_up" not in bound.skills
    with pytest.raises(UnsupportedSkillError, match="missing command 'open'"):
        bound.resolve("pick_up")


def test_one_participant_cannot_use_overlapping_required_endpoints() -> None:
    resources = _resources(include_right=False)
    resources["left_actor"] = RobotResource(
        "left_actor",
        endpoints={
            "motion": ControlPartEndpoint(
                "left_arm", capabilities=_MOTION_CAPABILITIES
            ),
            "grasp": ControlPartEndpoint(
                "left_arm", capabilities=frozenset({GRASP_CAPABILITY})
            ),
        },
        members=("left_arm",),
    )
    profiles = _command_profiles()
    profiles["left_arm"] = ControlPartCommandProfile.joint_positions(
        open=torch.zeros(2),
        grasp=torch.ones(2),
    )
    bound = _profile(resources=resources, command_profiles=profiles).bind(
        _engine(control_profiles=profiles)
    )

    assert "pick_up" not in bound.skills
    with pytest.raises(UnsupportedSkillError, match="overlap on joints"):
        bound.resolve("pick_up")


def test_coupled_endpoint_views_are_allowed_without_disjoint_constraint() -> None:
    class CoupledWholeBodyAction(AtomicAction[JointPositionGoal, ActionOptions]):
        skill_id: ClassVar[str] = "coupled_whole_body"
        GoalType: ClassVar[type] = JointPositionGoal
        binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
            slots=(
                SkillResourceSlot(
                    "body",
                    endpoints=(
                        SkillEndpointRequirement(
                            "motion",
                            capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                        ),
                        SkillEndpointRequirement(
                            "posture",
                            capabilities=frozenset({"control.posture"}),
                        ),
                    ),
                ),
            )
        )

        def _plan(
            self,
            request: ResolvedActionRequest[JointPositionGoal, ActionOptions],
            context: PlanningContext,
        ) -> ActionPlan:
            raise NotImplementedError

    resources = _resources()
    resources["coupled_body"] = RobotResource(
        "coupled_body",
        endpoints={
            "motion": ControlPartEndpoint(
                "full_body",
                capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
            ),
            "posture": ControlPartEndpoint(
                "full_body",
                capabilities=frozenset({"control.posture"}),
            ),
        },
        members=("base", "torso", "left_arm", "right_arm"),
    )
    engine = _engine(control_profiles=_command_profiles(), load_builtins=False)
    engine.register(CoupledWholeBodyAction())
    bound = _profile(resources=resources).bind(engine)

    assert bound.resolve("coupled_whole_body").resource_ids == {"body": "coupled_body"}


def test_disjoint_slot_constraint_rejects_one_actor_for_two_participants() -> None:
    profile = _profile(resources=_resources(include_right=False))
    bound = profile.bind(_engine(control_profiles=_command_profiles()))

    assert "hand_over" not in bound.skills
    with pytest.raises(UnsupportedSkillError, match="violate constraints"):
        bound.resolve("hand_over")


def test_composite_claim_conflicts_with_nested_arm_but_not_hand() -> None:
    bound = _profile().bind(_engine(control_profiles=_command_profiles()))

    whole_body = bound.resources["whole_body"].claim
    left_actor = bound.resources["left_actor"].claim
    left_hand = bound.resources["left_hand"].claim

    assert whole_body.conflicts_with(left_actor)
    assert not whole_body.conflicts_with(left_hand)


def test_generic_profile_supports_base_and_whole_body_without_arm_tool_fields() -> None:
    engine = _engine(control_profiles=_command_profiles(), load_builtins=False)
    engine.register(_WholeBodyAction())
    engine.register(_NavigateAction())
    bound = _profile().bind(engine)

    whole_body = bound.resolve("whole_body_reach")
    navigation = bound.resolve("navigate")

    assert set(bound.skills) == {"navigate", "whole_body_reach"}
    assert whole_body.resource_ids == {"body": "whole_body"}
    assert (
        whole_body.action_binding.endpoint("body", "motion")
        .require_target(JointPositionTarget)
        .control_part
        == "full_body"
    )
    assert whole_body.claim.leaf_resource_ids == frozenset(
        {"base", "torso", "left_arm", "right_arm"}
    )
    assert navigation.resource_ids == {"body": "base"}
    assert (
        navigation.action_binding.endpoint("body", "motion")
        .require_target(JointPositionTarget)
        .control_part
        == "base"
    )


def test_presets_are_versioned_snapshots_and_validate_planner() -> None:
    preset = SkillPolicyPreset(
        "safe",
        action_option_templates={
            "pick": PickUpOptions(pre_grasp_distance=0.08),
        },
        motion_policy=MotionPolicy(sample_count=80),
        tracking_policy=TrackingPolicy.joint_position(
            in_flight_max_abs_error=0.125,
            terminal_max_abs_error=0.125,
        ),
        required_planner="stub_planner",
    )
    profile = RobotSkillProfile(
        "presets",
        resources=_resources(),
        command_profiles=_command_profiles(),
        presets={"safe": preset},
        default_preset="safe",
        skill_presets={"pick_up": "safe"},
    )
    bound = profile.bind(_engine(control_profiles=_command_profiles()))

    first = bound.preset(skill_id="pick_up")
    second = bound.preset()

    assert first is not second
    assert first.schema_version == 2
    assert first.required_planner == "stub_planner"
    assert first.motion_policy.sample_count == 80
    assert first.tracking_policy is not second.tracking_policy
    assert first.action_option_templates["pick"] is not (
        second.action_option_templates["pick"]
    )
    assert (
        first.action_option_templates["pick"].pre_grasp_distance  # type: ignore[attr-defined]
        == 0.08
    )
    first_tracking = first.tracking_policy.in_flight
    assert first_tracking is not None
    assert isinstance(first_tracking.metrics[0], JointPositionTrackingMetric)
    assert first_tracking.metrics[0].tolerance == 0.125
    mutable_runner = first.runner_cfg
    mutable_runner.command_timeout = 99.0
    assert bound.preset().runner_cfg.command_timeout == 1.0
    with pytest.raises(KeyError, match="not an installed"):
        bound.preset(skill_id="typo")
    with pytest.raises(KeyError, match="not an installed"):
        bound.preset("safe", skill_id="typo")
    with pytest.raises(ValueError, match=r"supported versions are \[2\]"):
        SkillPolicyPreset("legacy", action_option_templates={}, schema_version=1)
    with pytest.raises(ValueError, match="required_planner"):
        SkillPolicyPreset("invalid", action_option_templates={}, required_planner="")

    incompatible = RobotSkillProfile(
        "bad_preset",
        resources=_resources(),
        command_profiles=_command_profiles(),
        presets={
            "other": SkillPolicyPreset(
                "other",
                action_option_templates={},
                required_planner="other_planner",
            )
        },
    )
    with pytest.raises(ProfileValidationError, match="requires planner"):
        incompatible.bind(_engine(control_profiles=_command_profiles()))


def test_policy_preset_defaults_exact_builtin_effect_monitor_refs() -> None:
    preset = SkillPolicyPreset("safe", action_option_templates={})

    assert set(preset.effect_monitors) == {
        "pick",
        "place",
        "hand_over",
    }
    for monitor_ref in preset.effect_monitors.values():
        assert monitor_ref.monitor_id == COMPOSITE_EFFECT_MONITOR_ID
        assert monitor_ref.revision == COMPOSITE_EFFECT_MONITOR_REVISION
        assert dict(monitor_ref.params) == {}


def test_policy_preset_distinguishes_explicit_empty_effect_monitor_mapping() -> None:
    preset = SkillPolicyPreset(
        "unmonitored",
        action_option_templates={},
        effect_monitors={},
    )

    assert dict(preset.effect_monitors) == {}
    assert dict(preset.snapshot().effect_monitors) == {}


def test_policy_preset_owns_and_snapshots_effect_monitor_refs() -> None:
    source_params = {
        "consecutive_samples": 3,
        "metadata": ["strict", {"source": "profile"}],
    }
    source_ref = EffectMonitorRef("test.monitor", "2", source_params)
    source_mapping = {"pick": source_ref}
    preset = SkillPolicyPreset(
        "custom",
        action_option_templates={},
        effect_monitors=source_mapping,
    )

    source_params["consecutive_samples"] = 99
    source_params["metadata"][1]["source"] = "mutated"  # type: ignore[index]
    source_mapping["pick"] = EffectMonitorRef("replacement", "1")
    first = preset.effect_monitors
    snapshot = preset.snapshot()
    second = snapshot.effect_monitors

    assert first["pick"] is not source_ref
    assert first["pick"].monitor_id == "test.monitor"
    assert first["pick"].params["consecutive_samples"] == 3
    assert first["pick"].params["metadata"] == (
        "strict",
        {"source": "profile"},
    )
    assert second["pick"] is not first["pick"]
    assert second["pick"].params == first["pick"].params
    with pytest.raises(TypeError):
        first["place"] = source_ref  # type: ignore[index]
    with pytest.raises(TypeError):
        first["pick"].params["consecutive_samples"] = 4  # type: ignore[index]


def test_policy_preset_owns_and_freezes_action_option_templates() -> None:
    direction = torch.tensor([0.0, 1.0, 0.0])
    source = PickUpOptions(
        pick_object_part="top",
        approach_direction=direction,
    )
    source_mapping = {"pick": source}
    preset = SkillPolicyPreset(
        "custom",
        action_option_templates=source_mapping,
    )

    direction.fill_(9.0)
    source.approach_direction.fill_(8.0)
    source_mapping.clear()
    first = preset.action_option_templates
    second = preset.snapshot().action_option_templates
    selected = preset.action_option_template("pick")

    assert type(first["pick"]) is PickUpOptions
    assert first["pick"] is not source
    assert second["pick"] is not first["pick"]
    assert selected is not first["pick"]
    assert first["pick"].pick_object_part == "top"  # type: ignore[attr-defined]
    torch.testing.assert_close(
        first["pick"].approach_direction,  # type: ignore[attr-defined]
        torch.tensor([0.0, 1.0, 0.0]),
    )
    with pytest.raises(TypeError):
        first["place"] = PickUpOptions()  # type: ignore[index]
    with pytest.raises(KeyError, match="no action-option template"):
        preset.action_option_template("place")


def test_policy_preset_requires_typed_action_option_templates() -> None:
    with pytest.raises(TypeError, match="action_option_templates"):
        SkillPolicyPreset("missing")  # type: ignore[call-arg]

    assert not SkillPolicyPreset(
        "empty", action_option_templates={}
    ).action_option_templates

    with pytest.raises(TypeError, match="ActionOptions"):
        SkillPolicyPreset(
            "invalid",
            action_option_templates={"pick": object()},  # type: ignore[dict-item]
        )


def test_policy_preset_rejects_deepcopy_with_nested_mutable_aliases() -> None:
    @dataclass(frozen=True, slots=True)
    class AliasingOptions(ActionOptions):
        values: list[int]

        def __deepcopy__(self, memo: dict[int, object]) -> AliasingOptions:
            del memo
            return type(self)(self.values)

    with pytest.raises(TypeError, match="without shared mutable objects"):
        SkillPolicyPreset(
            "invalid",
            action_option_templates={"vendor.alias": AliasingOptions([1])},
        )


def test_policy_preset_rejects_deepcopy_with_shared_tensor_storage() -> None:
    @dataclass(frozen=True, slots=True)
    class TensorViewOptions(ActionOptions):
        values: torch.Tensor

        def __deepcopy__(self, memo: dict[int, object]) -> TensorViewOptions:
            del memo
            return type(self)(self.values.view_as(self.values))

    with pytest.raises(TypeError, match="tensor storage"):
        SkillPolicyPreset(
            "invalid",
            action_option_templates={
                "vendor.tensor_alias": TensorViewOptions(torch.ones(2))
            },
        )


def test_profile_owns_named_grounding_provider_selections() -> None:
    selections = {"hand_over": "dual_center"}
    profile = RobotSkillProfile(
        "grounding",
        resources=_resources(),
        command_profiles=_command_profiles(),
        grounding_providers=selections,
    )

    selections["hand_over"] = "source_mutation"

    assert profile.grounding_providers == {"hand_over": "dual_center"}
    with pytest.raises(TypeError):
        profile.grounding_providers["pick"] = "invalid"  # type: ignore[index]
    with pytest.raises(ValueError, match="grounding_providers"):
        RobotSkillProfile(
            "invalid_grounding",
            resources=_resources(),
            grounding_providers={"hand_over": " provider"},
        )


def test_profile_rejects_default_for_uninstalled_skill() -> None:
    with pytest.raises(ProfileValidationError, match="not installed"):
        _profile(defaults={"missing": ResourceBinding({"primary": "left_actor"})}).bind(
            _engine(control_profiles=_command_profiles())
        )


def test_engine_can_install_profile_as_authoritative_command_source() -> None:
    source_engine = _engine(control_profiles=_command_profiles())
    profile = _profile(defaults={"pick_up": ResourceBinding({"primary": "left_actor"})})

    engine = AtomicActionEngine(source_engine.motion_generator, skill_profile=profile)

    assert engine.skill_profile is not None
    assert engine.skill_profile.resolve("pick_up").resource_ids == {
        "primary": "left_actor"
    }
    assert set(engine.control_profiles) == {"left_hand", "right_hand"}


def test_engine_rejects_endpoint_adapters_without_skill_profile() -> None:
    source = _engine(control_profiles={}, load_builtins=False)

    with pytest.raises(ValueError, match="requires skill_profile"):
        AtomicActionEngine(
            source.motion_generator,
            load_builtins=False,
            endpoint_adapters={_BaseVelocityEndpoint: _BaseVelocityEndpointAdapter()},
        )


def test_bound_profile_rejects_stale_engine_skill_catalog() -> None:
    engine = _engine(control_profiles=_command_profiles())
    bound = _profile().bind(engine)
    action_type = BUILTIN_ACTION_TYPES[0]

    class Replacement(action_type):
        binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
            slots=(
                SkillResourceSlot(
                    "primary",
                    endpoints=(
                        SkillEndpointRequirement(
                            "motion",
                            capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                        ),
                    ),
                ),
            )
        )

    engine.register(Replacement(), replace=True)

    assert engine.skill_profile is None
    with pytest.raises(RuntimeError, match="changed after"):
        _ = bound.skills


def test_bound_profile_rejects_equal_descriptor_implementation_replacement() -> None:
    engine = _engine(control_profiles=_command_profiles())
    bound = _profile().bind(engine)
    action_type = BUILTIN_ACTION_TYPES[0]

    class EquivalentReplacement(action_type):
        binding_contract: ClassVar[SkillBindingContract] = action_type.binding_contract

    assert EquivalentReplacement.descriptor() == action_type.descriptor()

    engine.register(EquivalentReplacement(), replace=True)

    with pytest.raises(RuntimeError, match="changed after"):
        _ = bound.skills


__all__ = []
