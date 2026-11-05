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

"""Held planning adapters retain original observations and close rejected queries."""

from __future__ import annotations
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.planning_probes import (
    HeldStagingBinding,
    RetainedCommandAuthority,
)
from embodichain.lab.sim.atomic_actions.state import (
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.scene.robot_distal import RobotDistalMeshSnapshot
from embodichain.lab.task_program.integrations.simulation.held_planning import (
    SimulationHeldPlanningProvider,
)


def _inputs():
    stamp = PhysicalObservationStamp("owner", "default.contacts", 7, 2.0)
    current = [stamp]
    qpos = torch.zeros(1, 4)
    qvel = torch.full_like(qpos, 0.1)
    target = torch.full_like(qpos, 0.2)
    target_velocity = torch.zeros_like(qpos)
    root = torch.eye(4)[None]
    reads = []

    def position(*, target=False):
        reads.append(("qpos", target))
        return (desired if target else qpos).clone()

    def velocity(*, target=False):
        reads.append(("qvel", target))
        return (target_velocity if target else qvel).clone()

    desired = target
    robot = SimpleNamespace(
        uid="robot",
        joint_names=("a", "b", "c", "d"),
        get_qpos=position,
        get_qvel=velocity,
        get_local_pose=lambda **kw: root.clone(),
        get_joint_ids=lambda name: (0, 1) if name == "custom_active" else (),
    )
    simulation = SimpleNamespace(
        num_envs=1, get_robot=lambda name: robot if name == "robot" else None
    )
    observed = RobotObservation(
        timestamp=1.0, qpos=qpos.clone(), qvel=qvel.clone(), root_pose=root.clone()
    )
    context = PlanningContext(
        robot=observed,
        task=TaskState.empty(1, "cpu"),
        scene=SceneSnapshot(timestamp=1.0, version=3, entities={}),
        env_ids=torch.tensor([0]),
        world_source_stamp=stamp,
    )
    request = SimpleNamespace(
        invocation_id="program/segment:0",
        skill_id="move_held_object",
        binding=SimpleNamespace(
            endpoint=lambda *a: SimpleNamespace(
                require_target=lambda cls: JointPositionTarget("custom_active", (0, 1))
            )
        ),
    )
    authority = RetainedCommandAuthority(
        preset_id="verified",
        predecessor_call_id="pick",
        predecessor_invocation_id="program/source:0",
        entity_id="payload",
        resource_id="resource",
        attachment_monitor_id="builtin.composite_effect",
        attachment_monitor_revision="1",
        retention_monitor_id="builtin.composite_effect",
        retention_monitor_revision="1",
    )
    bindings = {
        request.invocation_id: HeldStagingBinding(
            rigid_entity_id="payload", articulation_scene=True
        )
    }
    provider = SimulationHeldPlanningProvider(
        robot,
        simulation,
        bindings=bindings,
        retained_authorities={request.invocation_id: authority},
        read_source_stamp=lambda: current[0],
    )
    return provider, request, context, current, reads, desired, bindings


def test_static_bindings_are_owned_and_command_reads_keep_actual_not_projected_state():
    provider, request, context, current, reads, desired, bindings = _inputs()
    bindings.clear()
    assert provider.binding(request).rigid_entity_id == "payload"
    assert provider.authority(request).predecessor_call_id == "pick"
    assert reads == []
    assert (
        provider.binding(
            SimpleNamespace(skill_id="move_held_object", invocation_id="unknown")
        )
        is None
    )
    projected = context.project(qpos=context.robot.qpos + 1, task=context.task)
    observation = provider.capture(request, projected)
    assert torch.equal(observation.qpos, context.robot.qpos)
    assert torch.equal(observation.qvel, context.robot.qvel)
    assert reads == [("qpos", False), ("qvel", False), ("qpos", True), ("qvel", True)]
    provider.verify_current(observation)
    desired[0, 0] = 0.3
    with pytest.raises(ValueError, match="changed"):
        provider.verify_current(observation)
    assert observation.target_qpos[0, 0] == pytest.approx(0.2)
    current[0] = replace(current[0], source_id="foreign")
    with pytest.raises(ValueError, match="source changed"):
        provider.capture(request, context)


def test_distal_capture_delegates_original_observation_and_rejects_source_drift(
    monkeypatch,
):
    from embodichain.lab.task_program.integrations.simulation import (
        held_planning as owner,
    )

    provider, request, context, current, reads, _, _ = _inputs()
    projected = context.project(qpos=context.robot.qpos + 1, task=context.task)
    values = {part: np.eye(4) for part in ("custom_active", "custom_passive")}
    snapshot = RobotDistalMeshSnapshot(
        active="custom_active",
        passive="custom_passive",
        tcp=values,
        roots=values,
        points={part: np.ones((2, 3)) for part in values},
        hand_owner_fk_error={part: 0.0 for part in values},
    )
    captured = []

    def capture(robot, **kwargs):
        assert robot is provider.robot
        captured.append(kwargs)
        return snapshot

    monkeypatch.setattr(owner, "capture_robot_distal_meshes", capture)
    assert provider.capture_distal_geometry(request, projected) is snapshot
    assert captured[0]["expected_qpos"] is context.robot.qpos
    assert captured[0]["expected_root_pose"] is context.robot.root_pose
    assert captured[0]["active_control_part"] == "custom_active"
    assert reads == []

    def drift(robot, **kwargs):
        current[0] = replace(current[0], simulation_time=3.0)
        return snapshot

    monkeypatch.setattr(owner, "capture_robot_distal_meshes", drift)
    with pytest.raises(ValueError, match="source changed"):
        provider.capture_distal_geometry(request, context)


def test_articulation_query_is_closed_when_post_capture_scope_changes(monkeypatch):
    from embodichain.lab.task_program.integrations.simulation import (
        held_planning as owner,
    )

    provider, request, context, current, reads, _, _ = _inputs()
    created = []

    class Query:
        def __init__(self, robot, simulation, *, control_part):
            assert robot is provider.robot and control_part == "custom_active"
            self.closed = False
            created.append(self)
            current[0] = replace(current[0], revision=8)

        def close(self):
            self.closed = True

    monkeypatch.setattr(owner, "RobotArticulationCollisionSnapshot", Query)
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare_articulation_query(request, context)
    assert len(created) == 1 and created[0].closed
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare_articulation_query(request, context)
    assert len(created) == 1


@pytest.mark.parametrize("mode", ["verified", "projected", "missing", "wrong_revision"])
def test_retained_authority_requires_both_original_verified_monitor_contracts(mode):
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        SkillPolicyPreset,
        RobotSkillProfile,
    )
    from embodichain.lab.task_program.integrations.simulation.held_planning import (
        resolve_retained_command_authorities,
    )

    monitors = {
        name: EffectMonitorRef("builtin.composite_effect", "1")
        for name in ("pick", "gen_sim.align_held")
    }
    if mode == "projected":
        monitors = {}
    if mode == "missing":
        monitors.pop("pick")
    if mode == "wrong_revision":
        monitors["pick"] = EffectMonitorRef("builtin.composite_effect", "2")
    preset = SkillPolicyPreset(
        "verified",
        action_option_templates={},
        effect_assurance=(
            EffectAssurance.PROJECTED
            if mode == "projected"
            else EffectAssurance.VERIFIED
        ),
        effect_monitors=monitors,
    )
    profile = RobotSkillProfile("profile", resources={}, presets={"verified": preset})
    sources = (
        ("program/segment:0", "program/source:0", "pick", "payload", "resource"),
    )
    resolved = resolve_retained_command_authorities(profile, "verified", sources)
    if mode == "verified":
        provider, request, *_ = _inputs()
        assert (
            resolved[request.invocation_id].metadata()
            == provider.authority(request).metadata()
        )
    else:
        assert resolved == {}
    with pytest.raises(ValueError, match="unique occurrence"):
        resolve_retained_command_authorities(profile, "verified", sources + sources)


def test_staging_program_validation_keeps_exact_predecessor_subject_and_resource():
    from embodichain.lab.task_program.semantics import (
        Pick,
        RegisteredSemanticCall,
        SceneObjectRef,
        RobotSkillProfile,
        SkillPolicyPreset,
        EffectAssurance,
    )
    from embodichain.lab.task_program.integrations.simulation.held_planning import (
        HeldStagingInvocationBinding,
        _validate_program_bindings,
    )
    from embodichain.lab.task_program.integrations.invocation_policy import (
        InvocationPolicyBinding,
    )

    preset = SkillPolicyPreset(
        "policy", action_option_templates={}, effect_assurance=EffectAssurance.PROJECTED
    )
    profile = RobotSkillProfile("profile", resources={}, presets={"policy": preset})
    first = Pick(
        object=SceneObjectRef("payload"), resources={"primary": "custom_resource"}
    )
    second = RegisteredSemanticCall(
        call_id="gen_sim.align_held",
        arguments={"object": "payload", "preserve_yaw": False, "target": "destination"},
        resources={"primary": "custom_resource"},
    )
    segments = [
        SimpleNamespace(
            segment_id=name,
            parallel_block=None,
            repeat_frames=(),
            calls=[SimpleNamespace(call=call, segment_call_index=0, source_path=())],
        )
        for name, call in (("first", first), ("second", second))
    ]
    program = SimpleNamespace(program_id="p", iter_segments=lambda: iter(segments))
    manifest = SimpleNamespace(
        robot_profile=profile,
        link_call=lambda *args, **kwargs: SimpleNamespace(preset_id="policy"),
    )
    policies = tuple(
        InvocationPolicyBinding("a" * 64, "p/" + name + ":0", "policy")
        for name in ("first", "second")
    )
    binding = HeldStagingInvocationBinding(
        "p/second:0",
        "payload",
        predecessor_invocation_id="p/first:0",
        predecessor_call_id="pick",
        resource_id="custom_resource",
    )
    _validate_program_bindings(program, manifest, policies, (binding,))
    for changed in (
        replace(binding, rigid_entity_id="other"),
        replace(binding, resource_id="other"),
        replace(binding, predecessor_invocation_id="p/missing:0"),
    ):
        with pytest.raises(ValueError):
            _validate_program_bindings(program, manifest, policies, (changed,))
    segments[0].repeat_frames = ("repeat",)
    with pytest.raises(ValueError, match="earlier serial"):
        _validate_program_bindings(program, manifest, policies, (binding,))


@pytest.mark.parametrize("source_verified", [False, True])
def test_retained_declaration_checks_the_actual_predecessor_preset(source_verified):
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        SkillPolicyPreset,
        RobotSkillProfile,
    )
    from embodichain.lab.task_program.integrations.invocation_policy import (
        InvocationPolicyBinding,
    )
    from embodichain.lab.task_program.integrations.simulation.held_planning import (
        HeldStagingInvocationBinding,
        _declared_authorities,
    )

    monitors = {
        name: EffectMonitorRef("builtin.composite_effect", "1")
        for name in ("pick", "gen_sim.align_held")
    }
    current = SkillPolicyPreset(
        "current",
        action_option_templates={},
        effect_assurance=EffectAssurance.VERIFIED,
        effect_monitors=monitors,
    )
    previous = SkillPolicyPreset(
        "before",
        action_option_templates={},
        effect_assurance=(
            EffectAssurance.VERIFIED if source_verified else EffectAssurance.PROJECTED
        ),
        effect_monitors=monitors if source_verified else {},
    )
    profile = RobotSkillProfile(
        "profile", resources={}, presets={"current": current, "before": previous}
    )
    bindings = (
        HeldStagingInvocationBinding(
            "p/current:0",
            "payload",
            predecessor_invocation_id="p/before:0",
            predecessor_call_id="pick",
            resource_id="resource",
        ),
    )
    policies = (
        InvocationPolicyBinding("a" * 64, "p/current:0", "current"),
        InvocationPolicyBinding("a" * 64, "p/before:0", "before"),
    )
    result = _declared_authorities(profile, policies, bindings)
    assert bool(result) is source_verified
    if source_verified:
        assert result["p/current:0"].preset_id == "current"
        assert result["p/current:0"].predecessor_invocation_id == "p/before:0"
