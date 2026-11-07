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

"""Registered idle intent reaches ordinary Atomic/Motion with frozen FCL outcomes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    MotionPolicy,
    MoveJointsOptions,
)
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.motion.mesh_collision import SimulationFreeMotionQuery
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.task_program.integrations.configured import (
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from sim.motion._mesh_collision_fixture import mesh_query_fixture

_REFERENCE = json.loads(
    (
        Path(__file__).parents[3]
        / "sim/atomic_actions/data/idle_motion_source_reference.json"
    ).read_text()
)


@pytest.mark.skipif(
    importlib.util.find_spec("fcl") is None,
    reason="planning-geometry is required for actual FCL qualification",
)
@pytest.mark.parametrize(
    "case",
    [row for row in _REFERENCE["cases"] if row["target"][1] == 0],
    ids=lambda case: case["name"],
)
def test_configured_idle_lowerer_replays_source_frames_through_public_engine(
    case: dict[str, object],
) -> None:
    robot, simulation = mesh_query_fixture()
    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, torch.device("cpu")
    generator.planner = NS(cfg=NS(planner_type="toppra"), collision_world_info=None)
    engine = AtomicActionEngine(
        generator, free_motion_query=SimulationFreeMotionQuery(robot, simulation)
    )
    context = engine.initial_context(control_dt=0.04)
    factory = _decode_registered_lowerer(
        dict(
            kind="idle_arm_clearance",
            routes=[
                dict(
                    route_id="route",
                    clear_segment="clear",
                    restore_segment="restore",
                    resource="custom_motion",
                    joint_name="arm_slide",
                    offset_rad=case["target"][0],
                )
            ],
        ),
        path="services",
    )
    lowerer = factory.create(
        simulation=simulation, robot=robot, scene_registry=None, engine=engine
    )
    call = RegisteredSemanticCall(
        call_id="gen_sim.clear_idle_arm",
        arguments={"route": "route"},
        resources={"primary": "custom_motion"},
    )
    bound = NS(
        binding=NS(
            resources={
                "primary": NS(
                    endpoints={
                        "motion": NS(runtime_target=JointPositionTarget("arm", (0, 1)))
                    }
                )
            }
        )
    )
    options = MoveJointsOptions(require_free_motion=True)
    lowered = lowerer.lower(call, context=context, bound=bound, option_template=options)
    invocation = engine.make_invocation(
        "move_joints",
        lowered.goal,
        control_parts={"primary": {"motion": "arm"}},
        motion_policy=MotionPolicy(
            sample_count=3,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            acceleration_limit=None,
        ),
        skill_options=options,
        invocation_id="idle",
    )
    torch.manual_seed(491)
    before = torch.get_rng_state().clone()
    plan = engine.plan(invocation, context)
    actual = dict(
        name=case["name"],
        target=case["target"],
        positions=plan.joint_trajectory.positions.tolist(),
        dt=plan.joint_trajectory.dt.tolist(),
        success=plan.plan_success.tolist(),
        active_masks=[frame.active_mask.tolist() for frame in plan.commands.frames],
        rng_unchanged=torch.equal(before, torch.get_rng_state()),
    )
    assert actual == case
