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

"""Configured Atomic Skill lowering and fixed-scene deployment contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.utils._component_composition import (
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations._configured_atomic_skills import (
    _AxisAlignLowerer,
    _MoveJointsLowerer,
    _OpenDoorLowerer,
    _PressLowerer,
    _TwistLowerer,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_action_options,
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    JointPositionGoal,
    MoveJointsOptions,
    ObjectSemantics,
    OpenDoorAffordance,
    OpenDoorGoal,
    OpenDoorOptions,
    PressAffordance,
    PressGoal,
    PressOptions,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
)
from embodichain.lab.scripts.run_env import (
    _create_parser,
    _resolve_expansion_request,
    resolve_collection_plan,
)
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.utils.utility import load_config

_ROOT = Path(__file__).resolve().parents[4]
_TASKS = _ROOT / "extensions/embodichain_tasks/configs/tasks/manipulation"
_DEPLOYMENTS = (
    ("repeated_pickup/task.ur5.yaml", "pick"),
    ("repeated_move_held_object/task.ur5.yaml", "simulation.move_held_object"),
    ("repeated_pick_place/task.ur5.yaml", "place"),
    ("repeated_pour/task.ur5.yaml", "simulation.pour"),
    ("repeated_press/task.ur5.yaml", "simulation.press"),
    ("repeated_slide/task.ur5.yaml", "simulation.articulation_link_slide"),
    ("repeated_twist/task.ur5.yaml", "simulation.twist"),
    ("repeated_handover/task.dual_ur5_dh_pgi_140_80.yaml", "hand_over"),
    ("repeated_move_joint/task.ur5.yaml", "simulation.move_joints"),
    ("repeated_axis_align/task.ur5.yaml", "simulation.axis_align"),
    ("repeated_open_door/task.ur5.yaml", "simulation.open_door"),
    ("repeated_push_object/task.ur5.yaml", "simulation.push_object"),
)


@pytest.mark.parametrize(
    "task", ["repeated_press", "repeated_move_held_object", "repeated_pour"]
)
def test_dense_motion_deployments_use_blending_at_the_original_control_rate(
    task: str,
) -> None:
    path = _TASKS / task / "task.ur5.yaml"
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    assert deployment.planner_config["type"] == "trapezoidal"
    assert preset.motion_policy.strategy == "motion_gen"
    options = preset.motion_policy.plan_opts
    assert isinstance(options, TrapezoidalPlanOptions)
    assert not options.stop_at_waypoints
    assert options.blend_tolerance == 0.01
    assert options.collinearity_tolerance == 0.0001
    assert physical.config["env"]["sim_steps_per_control"] == 1


@pytest.mark.parametrize("relative,semantic_id", _DEPLOYMENTS)
def test_atomic_deployment_preflights_scene_calls_and_expansion_schedule(
    relative, semantic_id
):
    path = _TASKS / relative
    payload = load_config(path)
    physical = _resolve_gym_components(payload, base_dir=path.parent)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    catalog = deployment.integration.registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    compiled = catalog.preflight(program)
    calls = [
        call.call for segment in compiled.iter_segments() for call in segment.calls
    ]
    assert semantic_id in {call.semantic_id for call in calls}
    args = _create_parser().parse_args(["--gym-config", str(path)])
    profile, _, _, _ = _resolve_expansion_request(args, payload)
    assert profile.source.source_id == program.program_id
    assert profile.source.call_count == len(calls)
    assert not profile.scene_randomization.enabled
    assert not profile.visual.enabled
    assert profile.trajectory.variants_per_family == 4
    assert profile.trajectory.spatial.operators == ("joint_residual", "via_points")


@pytest.mark.parametrize("relative,_semantic_id", _DEPLOYMENTS)
def test_default_camera_collection_bounds_parallel_buffers_without_losing_recipes(
    relative, _semantic_id
):
    """Default RGB collection batches all recipes instead of allocating 16 buffers."""
    path = _TASKS / relative
    args = _create_parser().parse_args(["--gym-config", str(path)])
    cfg, config, _ = build_env_cfg_from_args(args)
    profile, _, _, collection = _resolve_expansion_request(args, load_config(path))
    plan = resolve_collection_plan(args, config, expansion_collection=collection)

    assert cfg.num_envs == profile.execution.max_inflight == 2
    assert cfg.enable_sensor
    rgba_bytes = (
        cfg.num_envs
        * cfg.max_episode_steps
        * sum(sensor.width * sensor.height * 4 for sensor in cfg.sensor)
    )
    # The former 16 x 10,000 x 320 x 240 x 4 allocation needed 46 GiB.
    assert rgba_bytes < 8 * 1024**3
    recipe_count = (
        profile.affordance.branches_per_family * profile.trajectory.variants_per_family
    )
    repeats = 3 if relative.startswith(("repeated_handover/", "repeated_slide/")) else 1
    assert plan.target_episodes == recipe_count * repeats
    batches = [
        plan.selection.take(
            cursor,
            min(cfg.num_envs, plan.target_episodes - cursor),
            recipe_count=recipe_count,
        )
        for cursor in range(0, plan.target_episodes, cfg.num_envs)
    ]
    assert tuple(index for batch in batches for index in batch) == (
        tuple(range(recipe_count)) * repeats
    )


@pytest.mark.parametrize(
    "kind", ["press", "twist", "open_door", "axis_align", "move_joints"]
)
def test_new_option_decoders_reject_cross_skill_fields(kind):
    with pytest.raises(ValueError, match="unsupported"):
        _decode_action_options(
            {"kind": kind, "translation_distance": 0.2}, path="options"
        )


@pytest.mark.parametrize(
    "kind", ["press", "twist", "open_door", "axis_align", "move_joints"]
)
def test_registered_factories_reject_executable_configuration(kind):
    with pytest.raises(ValueError, match="unsupported"):
        _decode_registered_lowerer(
            {"kind": kind, "class_type": "anything.callable"}, path="lowerer"
        )


@pytest.mark.parametrize(
    "lowerer_type,affordance,options,goal_type",
    [
        (
            _PressLowerer,
            PressAffordance(press_position=(0.0, 0.0, 0.0)),
            PressOptions(),
            PressGoal,
        ),
        (
            _TwistLowerer,
            TwistAffordance(
                grasp_position=(0.0, 0.0, 0.0), axis_origin=(0.0, 0.0, 0.0)
            ),
            TwistOptions(),
            TwistGoal,
        ),
    ],
)
def test_link_calls_use_live_scene_targets_and_owned_options(
    lowerer_type, affordance, options, goal_type
):
    semantics = ObjectSemantics(
        entity_id="target", label="target", geometry={}, affordance=affordance
    )
    lowerer = lowerer_type(semantics)
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={"target": SimpleNamespace(pose=torch.eye(4).repeat(2, 1, 1))}
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(lowerer.call_id, {"target": "target"}),
        context=context,
        bound=None,
        option_template=options,
    )
    assert type(result.goal) is goal_type
    assert torch.equal(result.goal.target_pose, context.scene.entities["target"].pose)
    assert result.goal.target_pose is not context.scene.entities["target"].pose
    assert result.skill_options is None
    with pytest.raises(ValueError, match="configured target"):
        lowerer.lower(
            RegisteredSemanticCall(lowerer.call_id, {"target": "other"}),
            context=context,
            bound=None,
            option_template=options,
        )


def test_open_door_maps_absolute_angle_to_hinge_fraction():
    affordance = OpenDoorAffordance(
        mesh_vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=(0.0, 0.0, 0.0),
        joint_name="hinge",
        joint_limits=(0.0, 2.0),
    )
    lowerer = _OpenDoorLowerer(
        ObjectSemantics(
            entity_id="handle", label="door", geometry={}, affordance=affordance
        )
    )
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={"handle": SimpleNamespace(pose=torch.eye(4).repeat(2, 1, 1))}
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            lowerer.call_id, {"handle": "handle", "open_angle": 1.0}
        ),
        context=context,
        bound=None,
        option_template=OpenDoorOptions(),
    )
    assert type(result.goal) is OpenDoorGoal
    assert result.goal.open_fraction == 0.5
    with pytest.raises(ValueError):
        lowerer.lower(
            RegisteredSemanticCall(
                lowerer.call_id, {"handle": "handle", "open_angle": 3.0}
            ),
            context=context,
            bound=None,
            option_template=OpenDoorOptions(),
        )


def test_axis_align_keeps_the_configured_object_axis():
    semantics = ObjectSemantics(
        entity_id="cube",
        label="cube",
        geometry={},
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0])),
    )
    lowerer = _AxisAlignLowerer(semantics)
    result = lowerer.lower(
        RegisteredSemanticCall(lowerer.call_id, {"object": "cube"}),
        context=None,
        bound=None,
        option_template=AxisAlignOptions(),
    )
    assert type(result.goal) is AxisAlignGoal
    assert torch.equal(
        result.goal.semantics.affordance.internal_axis, torch.tensor([1.0, 0.0, 0.0])
    )
    assert result.skill_options is None


def test_move_joints_keeps_each_required_target_as_one_goal():
    lowerer = _MoveJointsLowerer()
    context = SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(2, 6)))
    result = lowerer.lower(
        RegisteredSemanticCall(lowerer.call_id, {"qpos": [0.0, 1.0, 2.0]}),
        context=context,
        bound=None,
        option_template=MoveJointsOptions(),
    )
    assert type(result.goal) is JointPositionGoal
    assert result.goal.target.shape == (3,)
    assert result.skill_options is None
    with pytest.raises(ValueError, match="finite real"):
        lowerer.lower(
            RegisteredSemanticCall(lowerer.call_id, {"qpos": [[0.0, 1.0, 2.0]]}),
            context=context,
            bound=None,
            option_template=MoveJointsOptions(),
        )


@pytest.mark.parametrize(
    "relative,point,closure",
    [
        ("repeated_pickup/task.ur5.yaml", "grasp", 0.036),
        ("repeated_press/task.ur5.yaml", "default", 0.04),
        ("repeated_twist/task.ur5.yaml", "default", 0.036),
        ("repeated_open_door/task.ur5.yaml", "grasp", 0.04),
    ],
)
def test_ur5_fixtures_match_atomic_tcp_grasp_geometry_and_closure(
    relative: str, point: str, closure: float
) -> None:
    from embodichain.lab.scripts._atomic_action.tcp_presets import sample_tool_tcp
    from embodichain.lab.scripts._atomic_action.support import (
        TUTORIAL_PARALLEL_JAW_MODEL,
    )

    path = _TASKS / relative
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    assert physical.config["robot"]["solver_cfg"]["arm"]["tcp"] == sample_tool_tcp(
        "ur5", point=point
    )
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    generator = (
        deployment.integration.adapter_factory.delegate._grasp_pose_generator_factories[
            "eef"
        ]()
    )
    assert generator.gripper_model == TUTORIAL_PARALLEL_JAW_MODEL
    assert physical.embodiment_skill_profile["command_presets"][0]["commands"][
        "grasp"
    ] == [closure]
