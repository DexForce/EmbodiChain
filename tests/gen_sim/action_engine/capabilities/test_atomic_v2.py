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

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapability,
    StateAtom,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.runtime.loader import load_execution_program
from embodichain.gen_sim.action_engine.runtime.models import GroundedAction
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

from ..task_fixtures import make_task_spec
from embodichain.gen_sim.action_engine.planning.linker import link_seed_graph
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionOptions,
    ActionPlan,
    EndEffectorPoseGoal,
    PlannerDiagnostics,
    RuntimeCommandFrame,
    TimedTrajectory,
    TimedCommandSequence,
    TrackingPolicy,
)


@dataclass(frozen=True, slots=True)
class _TestOptions(ActionOptions):
    marker: str = "test"


class _TestAction:
    skill_id = "test_retreat"
    end_effector_roles: tuple[str, ...] = ()


class _TestEngine:
    binding_owner_id = "test-engine"

    def bind_control_parts(self, _skill_id, _endpoints):
        return ActionBinding(owner_id=self.binding_owner_id)

    def plan(self, invocation, context):
        assert isinstance(invocation.skill_options, _TestOptions)
        positions = context.robot.qpos[:, None, :]
        trajectory = TimedTrajectory.from_uniform_step(
            positions,
            env_ids=context.env_ids,
            step_dt=context.require_control_dt(),
        )
        return ActionPlan(
            skill_id=invocation.skill_id,
            plan_success=torch.ones(context.batch_size, dtype=torch.bool),
            commands=TimedCommandSequence(
                frames=(
                    RuntimeCommandFrame(
                        commands=(),
                        active_mask=torch.ones(
                            context.batch_size,
                            dtype=torch.bool,
                        ),
                        env_ids=context.env_ids,
                        hold_duration=trajectory.dt[:, 0],
                    ),
                ),
                env_ids=context.env_ids,
            ),
            joint_trajectory=trajectory,
            recovery_policy=invocation.recovery_policy,
            tracking_policy=TrackingPolicy.timed(),
            planned_scene_version=context.scene.version,
            planned_collision_world_revision=(0,) * context.batch_size,
            diagnostics=PlannerDiagnostics(backend="test"),
        )


class _Robot:
    dof = 2
    uid = "test_robot"
    control_parts = {"left_arm": [0], "right_arm": [1]}

    def get_qpos(self):
        return torch.zeros((1, 2))

    def get_joint_ids(self, *, name: str):
        return self.control_parts.get(name, [])


class _Entity:
    def get_local_pose(self, *, to_matrix: bool):
        assert to_matrix
        return torch.eye(4).unsqueeze(0)


class _Sim:
    def get_rigid_object(self, _uid: str):
        return _Entity()


def test_axis_align_uses_its_tutorial_motion_policy_base() -> None:
    capability = build_atomic_capability_registry().get("AxisAlign")

    assert capability.motion_base == "AxisAlign"


def test_axis_align_retains_ownership_until_explicit_release() -> None:
    capability = build_atomic_capability_registry().get("AxisAlign")
    contract = capability.resolve_contract(
        {
            "object_uid": "can",
            "actor": {"mode": "required", "arm": "left_arm"},
        }
    )

    assert capability.state_effect == "hold"
    assert capability.verifier_hook is None
    assert contract.requires == (
        StateAtom("arm_free", arm="left_arm"),
        StateAtom("object_free", object_uid="can"),
    )
    assert contract.effects[-1].atom == StateAtom(
        "object_held",
        object_uid="can",
        arm="left_arm",
    )


def test_single_arm_release_contract_frees_the_object_and_arm() -> None:
    capability = build_atomic_capability_registry().get("MoveJoints")
    contract = capability.resolve_contract(
        {
            "object_uid": "can",
            "actor": {"mode": "required", "arm": "left_arm"},
            "control": "hand",
            "target_binding": {
                "kind": "joint_state",
                "source": "gripper_open",
                "single_release": True,
            },
        }
    )

    assert contract.requires == (
        StateAtom("object_held", object_uid="can", arm="left_arm"),
    )
    assert [(effect.op, effect.atom.predicate) for effect in contract.effects] == [
        ("delete", "object_held"),
        ("add", "arm_free"),
        ("add", "object_free"),
    ]
    assert contract.failure_policy == "task_required"


def test_single_arm_release_verifies_the_selected_gripper_is_open() -> None:
    capability = build_atomic_capability_registry().get("MoveJoints")
    executor = SimpleNamespace(
        env=SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=2,
            open_state=(0.0, 0.0),
            get_current_gripper_state_agent=lambda: (
                torch.tensor([[0.0, 0.0], [0.1, 0.1]]),
                torch.ones(2, 2),
            ),
        ),
        runtime_policy=SimpleNamespace(
            predicate_fallbacks={"gripper_state_tolerance": 1.0e-3}
        ),
    )
    outcome = SimpleNamespace(
        grounded=SimpleNamespace(motion_policy={"single_release": True})
    )

    verified = capability.verifier_hook(
        executor=executor,
        step=SimpleNamespace(),
        arm="left_arm",
        outcome=outcome,
        attempted=torch.tensor([True, True]),
    )

    assert verified.tolist() == [True, False]


def test_explicit_required_home_is_safety_required_for_any_task_type() -> None:
    capability = build_atomic_capability_registry().get("MoveJoints")
    base = {
        "atomic_action": "MoveJoints",
        "object_uid": "can",
        "actor": {"mode": "required", "arm": "right_arm"},
        "control": "arm",
        "role": "cleanup",
        "target_binding": {"kind": "joint_state", "source": "initial"},
    }

    generic = capability.resolve_contract(base)
    required_home = capability.resolve_contract(
        {
            **base,
            "task_type": "test_carrier_consumer",
            "target_binding": {
                **base["target_binding"],
                "operation": "custom_home",
                "required_home": True,
            },
        }
    )

    assert generic.failure_policy == "best_effort"
    assert required_home.failure_policy == "safety_required"


def test_coordinated_release_contract_uses_binding_not_task_number() -> None:
    capability = build_atomic_capability_registry().get("MoveJoints")
    node = {
        "atomic_action": "MoveJoints",
        "task_type": "test_carrier_consumer",
        "object_uid": "tray",
        "actor": {"mode": "required", "arm": "left_arm"},
        "control": "hand",
        "role": "primary",
        "sync_group": "release_pair",
        "target_binding": {
            "kind": "joint_state",
            "source": "gripper_open",
            "coordinated_release_role": "participant",
        },
    }

    contract = capability.resolve_contract(node)

    assert contract.requires == (
        StateAtom("object_coordinated_held", object_uid="tray"),
    )


def test_new_descriptor_reuses_loader_and_adapter_without_dispatch_changes() -> None:
    registry = build_atomic_capability_registry()
    calls = []

    def target_hook(**kwargs):
        calls.append("target")
        pose = kwargs["object_pose"].clone()
        return GroundedAction(
            action_class="TestRetreat",
            arm=kwargs["arm"],
            control="arm",
            target=EndEffectorPoseGoal(xpos=pose),
            cfg=kwargs["policy"],
            object_pose=pose,
            target_object_pose=pose,
            motion_policy=kwargs["policy"],
        )

    def config_hook(**_kwargs):
        calls.append("config")
        return _TestOptions()

    registry.register(
        AtomicCapability(
            "TestRetreat",
            _TestAction,
            _TestOptions,
            frozenset({"policy_pose"}),
            frozenset({"arm"}),
            "single_arm",
            "preserve",
            "eef_pose",
            motion_base="MoveEndEffector",
            target_materializer_hook=target_hook,
            config_materializer_hook=config_hook,
            contract_resolver_hook=registry.get(
                "MoveEndEffector"
            ).contract_resolver_hook,
        )
    )
    task, requirements = make_task_spec("E1")
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    graph = instantiate_seed_graph(task, bindings)
    graph = deepcopy(graph)
    graph["capability_catalog_hash"] = registry.catalog_hash()
    cleanup = next(
        node for node in graph["nodes"] if node["atomic_action"] == "MoveEndEffector"
    )
    cleanup["atomic_action"] = "TestRetreat"
    cleanup.pop("contract")
    for group in graph["task_groups"]:
        group.pop("contract")
    graph["metadata"].pop("action_contract_linker")
    graph = link_seed_graph(graph, registry=registry)

    program = load_execution_program(graph, registry=registry)
    assert any(
        action["atomic_action_class"] == "TestRetreat"
        for edge in program.edges
        for action in edge.actions
    )

    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        robot=_Robot(),
        sim=_Sim(),
        agent_robot_profile="dual_ur10",
        get_agent_arm_control_part=lambda is_left: (
            "left_arm" if is_left else "right_arm"
        ),
        get_agent_eef_control_part=lambda _is_left: None,
    )
    adapter = AtomicActionAdapter(
        env,
        grasp_policy={},
        capability_registry=registry,
    )
    adapter._atomic_engine = _TestEngine()
    step = next(
        step
        for step in program.semantic_steps
        if any(
            action["atomic_action_class"] == "TestRetreat"
            for edge_id in step.edge_ids
            for action in next(
                edge for edge in program.edges if edge.id == edge_id
            ).actions
        )
    )
    action = next(
        action
        for edge in program.edges
        for action in edge.actions
        if action["atomic_action_class"] == "TestRetreat"
    )
    grounder = ActionGrounder(
        program,
        env,
        lambda _uid: None,
        capability_registry=registry,
    )
    state = ExecutionState(last_qpos=torch.zeros((1, 2)))
    grounded = grounder.ground(action, step, arm="left_arm", state=state)
    outcome = adapter.plan(
        grounded,
        state,
    )
    assert outcome.success.tolist() == [True]
    assert calls == ["target", "config"]
