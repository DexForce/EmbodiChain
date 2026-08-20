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
