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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math

import pytest
import torch
import yaml

from embodichain.compute.geometry.cable import cable_endpoints
from embodichain.lab.gym.utils.composition import resolve_gym_components
from embodichain.lab.task_program.integrations import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations.simulation.cable import (
    _decode_cable_service,
    _CableRuntime,
    _CableCheckLowerer,
    _CableJointLowerer,
    _CableJointFactory,
    _CableMoveFactory,
    _CableMoveLowerer,
    _CablePickFactory,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.sim.atomic_skills import MoveEndEffectorOptions, MoveJointsOptions

pytestmark = pytest.mark.no_sim
_ROOT = Path(__file__).resolve().parents[4]
_TASK = (
    _ROOT
    / "extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable"
)


def _declaration():
    return yaml.safe_load((_TASK / "task_program/integration.yaml").read_text())[
        "runtime_services"
    ]["cable_material"]


def _deployment(*, compact=False):
    path = _TASK / ("task.yam.compact.yaml" if compact else "task.yam.yaml")
    raw = yaml.safe_load(path.read_text())
    physical = resolve_gym_components(raw, base_dir=path.parent)
    deployment = load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        evaluation=physical.evaluation,
        base_dir=path.parent,
    )
    physical.validate_scene_bindings(deployment.scene_binding)
    return deployment


def test_cable_deployment_preflights_all_calls_and_measured_final_acceptance():
    deployment = _deployment()
    registration = deployment.integration.registration
    compiled = registration.catalog.preflight(deployment.load_program())
    assert compiled.segment_count == 57
    assert registration.evaluation_final_refs == ("route_success",)
    assert registration.scene_binding.declare().entries == ()
    factories = registration.registered_semantic_lowerer_factories
    assert {factory.target_descriptor.skill_id for factory in factories} == {
        "pick_up",
        "move_end_effector",
        "move_joints",
    }
    for factory in factories:
        assert factory.argument_contract.required["target"]


def test_compact_cable_preflight_has_eight_calls_and_retains_physical_checks():
    deployment = _deployment(compact=True)
    registration = deployment.integration.registration
    compiled = registration.catalog.preflight(deployment.load_program())
    assert compiled.segment_count == 8
    cfg = next(
        factory.cfg
        for factory in registration.registered_semantic_lowerer_factories
        if factory.call_id == "simulation.cable_pickup"
    )
    original = _decode_cable_service(_declaration())
    assert (cfg.geometry, cfg.motion_parts, cfg.slip_limit) == (
        original.geometry,
        original.motion_parts,
        original.slip_limit,
    )
    # Timing changes must preserve every geometric target and physical gate.
    targets = {target.target_id: target for target in cfg.targets}
    assert set(targets) == {target.target_id for target in original.targets} | {
        "settle_contact"
    }
    for target in original.targets:
        assert (
            replace(
                targets[target.target_id],
                speed=target.speed,
                angular_speed=target.angular_speed,
                samples=target.samples,
            )
            == target
        )
    operations = cfg.operations
    assert len(operations) == 8
    phases = [phase for operation in operations for phase in operation.phases]
    assert len(phases) == 55
    assert sum(phase.kind == "grasp" for phase in phases) == 4
    assert sum(post.kind == "check" for phase in phases for post in phase.after) == 48
    assert sum(post.kind == "hold" for phase in phases for post in phase.after) == 45
    assert {
        name: targets[name].samples
        for name in (
            "settle_soft_grasp",
            "settle_contact",
            "settle_released_end",
            "settle_and_validate",
        )
    } == {
        "settle_soft_grasp": 30,
        "settle_contact": 15,
        "settle_released_end": 30,
        "settle_and_validate": 120,
    }
    for phase in phases:
        if targets[phase.target].kind == "lay_down":
            assert targets[phase.target].angular_speed == 0.5
            assert phase.after[0].target == "settle_contact"
    assert registration.evaluation_final_refs == ("route_success",)


@pytest.mark.parametrize(
    "operation_name,call_id",
    [
        ("right_pick_0", "simulation.cable_pickup"),
        ("right_place_0", "simulation.cable_place"),
    ],
)
def test_contact_catalog_accepts_a_single_declared_operation_kind(
    operation_name, call_id
):
    path = _TASK / "task.yam.compact.yaml"
    physical = resolve_gym_components(
        yaml.safe_load(path.read_text()), base_dir=path.parent
    )
    declaration = yaml.safe_load(
        (_TASK / "task_program/integration.compact.yaml").read_text()
    )
    operations = declaration["runtime_services"]["cable_material"]["operations"]
    declaration["runtime_services"]["cable_material"]["operations"] = {
        operation_name: operations[operation_name]
    }
    selection = deepcopy(physical.config["task_program"])
    selection["integration"] = {"inline": declaration}
    deployment = load_configured_task_program_deployment(
        task_program=selection,
        skill_profile=physical.embodiment_skill_profile,
        evaluation=physical.evaluation,
        base_dir=path.parent,
    )
    compact_ids = {
        factory.call_id
        for factory in deployment.integration.registration.registered_semantic_lowerer_factories
    } & {"simulation.cable_pickup", "simulation.cable_place"}
    assert compact_ids == {call_id}


@pytest.mark.parametrize(
    "change", ["unknown", "negative_coordinate", "bad_geometry", "executable"]
)
def test_cable_configuration_rejects_invalid_closed_payload(change):
    value = _declaration()
    if change == "unknown":
        value["teleport"] = True
    elif change == "negative_coordinate":
        value["targets"]["right_grasp_0.162"]["material_distance"] = -1
    elif change == "bad_geometry":
        value["geometry"]["strict_region"] = [[0, 0, 0]]
    else:
        value["targets"]["right_grasp_0.162"]["class_type"] = "os.system"
    with pytest.raises((ValueError, TypeError)):
        _decode_cable_service(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"material_distance": None},
        {"material_distance": -0.01},
        {"retract_distance": -0.001},
        {"retract_distance": float("inf")},
        {"position": [0.5, None, 0.78]},
        {"position": [None, None, None]},
        {"position": [None, None, float("nan")]},
        {"firm": True},
        {"kind": "material", "retract_distance": 0.005},
    ],
)
def test_material_lift_rejects_invalid_geometry_and_inapplicable_fields(changes):
    value = _declaration()
    target = {
        "kind": "material_lift",
        "material_distance": 0.162,
        "position": [None, None, 0.78],
    }
    target.update(changes)
    if target.get("material_distance") is None:
        target.pop("material_distance")
    value["targets"] = {"lift": target}
    with pytest.raises((ValueError, TypeError)):
        _decode_cable_service(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"material_distance": -1},
        {"pitch": 0.1},
        {"pitch": -math.pi / 2},
        {"minimum_tool_pitch": float("nan")},
        {"maximum_rotation": float("inf")},
        {"maximum_rotation": 0},
        {"maximum_rotation": math.pi},
        {"kind": "yaw_align", "pitch": -0.1},
    ],
)
def test_pitch_alignment_rejects_invalid_angles_and_inapplicable_fields(changes):
    value = _declaration()
    target = {"kind": "pitch_align", "material_distance": 0.162}
    target.update(changes)
    value["targets"] = {"pitch": target}
    with pytest.raises((ValueError, TypeError)):
        _decode_cable_service(value)


def test_cable_binding_rejects_missing_physical_uid_before_simulation():
    from embodichain.lab.gym.utils._component_composition import (
        _validate_scene_binding_targets,
    )

    with pytest.raises(ValueError, match="not declared"):
        _validate_scene_binding_targets(
            {"cables": [{"entity_id": "cable", "simulation_uid": "missing"}]},
            simulation={"cable_object": [{"uid": "cable"}]},
        )


def _runtime():
    cfg = _decode_cable_service(_declaration())
    qpos = torch.zeros(4, 3)
    origins = torch.tensor(
        [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]], dtype=torch.float32
    )
    state = torch.zeros(4, 50, 13)
    state[..., 4] = math.sqrt(0.5)
    state[..., 6] = math.sqrt(0.5)
    state[..., 0] = (torch.arange(50) + 0.5) * 0.012
    state[..., :3] += origins[1:, None]

    class Arenas:
        root_offsets = origins.numpy()

        def index(self, name):
            return int(name[5:])

    def fk(q, **_):
        result = torch.eye(4).repeat(len(q), 1, 1)
        result[:, :3, 3] = q
        return result

    robot = SimpleNamespace(
        body_data=SimpleNamespace(
            entities=[SimpleNamespace(arena_name=f"arena{i}") for i in range(1, 5)]
        ),
        get_joint_ids=lambda **_: [0, 1, 2],
        get_qpos=lambda **_: qpos,
        compute_fk=fk,
    )
    cable = SimpleNamespace(
        cfg=SimpleNamespace(
            points=[[i * 0.012, 0, 0] for i in range(51)], radius=0.0035
        ),
        data=SimpleNamespace(
            segment_state_w=state, batch=SimpleNamespace(arena_indices=(1, 2, 3, 4))
        ),
    )
    sim = SimpleNamespace(
        spawn_result=SimpleNamespace(arenas=Arenas()), get_cable_object=lambda _: cable
    )
    service = _CableRuntime(cfg, sim, robot, torch.tensor([3, 1]))
    return service, state, qpos


def test_material_observation_maps_selected_rows_and_arena_origins_once():
    service, _, _ = _runtime()
    torch.testing.assert_close(
        service.origins, torch.tensor([[40.0, 0, 0], [20.0, 0, 0]])
    )
    torch.testing.assert_close(
        service.point(0.162)[0] - service.origins,
        torch.tensor([[0.162, 0, 0], [0.162, 0, 0]]),
        atol=4e-6,
        rtol=0,
    )


@pytest.mark.parametrize("retract_distance", [0.0, 0.008])
def test_material_lift_retracts_from_measured_tool_along_horizontal_material_tangent(
    retract_distance,
):
    service, state, measured = _runtime()
    value = _declaration()
    target = {
        "kind": "material_lift",
        "material_distance": 0.162,
        "position": [None, None, 0.78],
    }
    if retract_distance:
        target["retract_distance"] = retract_distance
    value["targets"] = {"lift": target}
    target = _decode_cable_service(value).target("lift")
    measured[3] = torch.tensor([0.50, 0.02, 0.76])
    measured[1] = torch.tensor([0.60, 0.03, 0.77])
    commanded = measured + 0.01
    service.robot.get_qpos = lambda *, target=False: commanded if target else measured
    original_fk = service.robot.compute_fk
    measured_rotation = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )

    def rotated_fk(qpos, **kwargs):
        pose = original_fk(qpos, **kwargs)
        pose[:, :3, :3] = measured_rotation
        return pose

    service.robot.compute_fk = rotated_fk
    # Native rows 3 and 1 belong to arenas at X=40 and X=20. The first
    # tangent also tilts upward, so the XY projection must be normalized.
    for row, direction in ((3, (0.0, 0.6, 0.8)), (1, (-0.6, 0.8, 0.0))):
        x, y, z = direction
        quaternion = torch.tensor([-y, x, 0.0, 1 + z])
        state[row, 13, 3:7] = quaternion / quaternion.norm()
    pose = service.pose(target, "left_arm")
    expected = torch.tensor([[0.50, 0.02, 0.78], [0.60, 0.03, 0.78]])
    expected[:, :2] -= retract_distance * torch.tensor([[0.0, 1.0], [-0.6, 0.8]])
    torch.testing.assert_close(pose[:, :3, 3], expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(pose[:, :3, :3], measured_rotation.repeat(2, 1, 1))


def test_material_lift_rejects_vertical_tangent_without_horizontal_retraction_axis():
    service, state, _ = _runtime()
    value = _declaration()
    value["targets"] = {
        "lift": {
            "kind": "material_lift",
            "material_distance": 0.162,
            "position": [None, None, 0.78],
            "retract_distance": 0.008,
        }
    }
    state[1, 13, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="horizontal direction"):
        service.pose(_decode_cable_service(value).target("lift"), "left_arm")


def test_material_reference_is_captured_after_close_and_tracks_slip_per_row():
    service, state, qpos = _runtime()
    service.pending_pick = ("right_arm", service.cfg.target("right_grasp_0.162"))
    action = SimpleNamespace(metadata={"bridge_action_kind": "runtime_command"})
    phase = ["close", False, 5]
    runtime = SimpleNamespace(_last_sent_command_phase=lambda: phase)
    service.after_action(action, runtime)
    assert not service.grasps
    # This represents the just-consumed close frame, before the first lift.
    qpos[3, 0] = 0.004
    phase[1] = True
    service.after_action(action, runtime)
    baseline = service.grasps["right_arm"][1].clone()
    torch.testing.assert_close(
        baseline[:, 0], torch.tensor([0.158, 0.162]), atol=4e-6, rtol=0
    )
    state[3, :, 0] += 0.013
    rejected, _ = service.after_action(action, runtime)
    assert rejected.tolist() == [True, False]
    torch.testing.assert_close(service.grasps["right_arm"][1], baseline)
    fresh, _, _ = _runtime()
    assert not fresh.grasps and fresh.max_slip.tolist() == [0, 0]


def test_partial_close_captures_only_acknowledged_rows_and_keeps_slip_history():
    service, _, _ = _runtime()
    part = "right_arm"
    target = service.cfg.target("right_grasp_0.162")
    original = torch.full((2, 3), 0.25)
    service.grasps[part] = (target.material_distance, original.clone())
    service.grasp_active[part] = torch.tensor([False, False])
    service.grasp_slip[part] = torch.tensor([0.004, 0.003])
    service.max_slip[:] = torch.tensor([0.013, 0.003])
    service.pending_pick = (part, target)
    service.pending_pick_rows = torch.tensor([False, True])
    service.after_action(
        SimpleNamespace(metadata={"bridge_action_kind": "runtime_command"}),
        SimpleNamespace(_last_sent_command_phase=lambda: ("close", True, 7)),
    )
    torch.testing.assert_close(service.grasps[part][1][0], original[0])
    torch.testing.assert_close(
        service.grasps[part][1][1], service.point_in_tool(part, 0.162)[1]
    )
    assert service.grasp_active[part].tolist() == [False, True]
    torch.testing.assert_close(service.max_slip, torch.tensor([0.013, 0.003]))


def test_contact_source_partial_view_keeps_metric_rows_in_physical_id_order():
    from embodichain.lab.task_program.integrations.simulation._cable_contact import (
        CableContactSource,
    )

    original, _, measured = _runtime()
    service = _CableRuntime(
        original.cfg, original.simulation, original.robot, torch.tensor([3, 1, 0])
    )
    part = "left_arm"
    service.grasps[part] = (0.162, service.point_in_tool(part, 0.162).clone())
    service.grasp_slip[part] = torch.tensor([0.013, 0.002, 0.003])
    service.grasp_active[part] = torch.tensor([False, True, True])
    service.max_slip[:] = service.grasp_slip[part]
    source = CableContactSource(service)
    view, rows, _ = source._view(
        SimpleNamespace(
            env_ids=service.env_ids,
            robot=SimpleNamespace(qpos=measured[service.env_ids]),
        ),
        torch.tensor([False, True, True]),
    )
    assert rows.tolist() == [1, 2]
    assert view.env_ids.tolist() == [1, 0]
    metrics = view.metrics()
    torch.testing.assert_close(metrics["max_grasp_slip"], torch.tensor([0.002, 0.003]))
    assert metrics["end_position"].shape == (2, 3)
    torch.testing.assert_close(service.max_slip, torch.tensor([0.013, 0.002, 0.003]))


def test_repeated_contact_check_is_measured_again_after_settling():
    from embodichain.lab.task_program.integrations.simulation._cable_contact import (
        CableContactSource,
    )
    from embodichain.lab.sim.atomic_skills.primitives.contact import (
        ContactPhase,
        ContactContinuation,
        ResolvedContactPhase,
    )

    service, _, _ = _runtime()
    source = CableContactSource(service)
    observations = []

    def condition(target, part):
        observations.append(target.target_id)
        return torch.tensor([len(observations) == 1, True]), {}

    service.condition = condition
    source.current = ContactContinuation(
        1,
        1,
        "waypoint",
        ContactPhase("pose", "waypoint"),
        ResolvedContactPhase(
            torch.zeros(2, 2, 4, 4),
            torch.zeros(2, 3),
            torch.ones(2, dtype=torch.bool),
            data=(("part", "left_arm"),),
        ),
        ((1, "raise_0_tracking"), (5, "raise_0_tracking")),
    )
    source.rejected = torch.zeros(2, dtype=torch.bool)
    runtime = SimpleNamespace(last_sent_plan_reference=(0, 0, 1))
    source.observe(None, runtime)
    runtime.last_sent_plan_reference = (0, 0, 5)
    source.observe(None, runtime)
    accepted, _ = source.finish(
        source.current,
        SimpleNamespace(env_ids=service.env_ids, robot=SimpleNamespace(timestamp=1.0)),
        torch.ones(2, dtype=torch.bool),
    )
    assert observations == ["raise_0_tracking", "raise_0_tracking"]
    assert accepted.tolist() == [False, True]


def test_rejected_concurrent_contact_operation_preserves_the_live_observer():
    from embodichain.lab.task_program.integrations.simulation._cable_contact import (
        CableContactSource,
    )
    from embodichain.lab.sim.atomic_skills.primitives.contact import (
        ContactPhase,
        ContactContinuation,
        ResolvedContactPhase,
    )

    service, _, _ = _runtime()
    source = CableContactSource(service)
    phase = ContactPhase("pose", "right_approach_0.162")
    resolved = ResolvedContactPhase(
        torch.zeros(2, 2, 4, 4),
        torch.zeros(2, 3),
        torch.ones(2, dtype=torch.bool),
        data=(
            ("operation_id", "first"),
            ("rows", torch.tensor([0, 1])),
            ("part", "right_arm"),
        ),
    )
    first = ContactContinuation(1, 2, phase.target, phase, resolved)
    context = SimpleNamespace(robot=SimpleNamespace(timestamp=0))
    source.begin(first, context, torch.ones(2, dtype=torch.bool))
    source.observed_checks.add((3, "tracking"))
    other = replace(
        first, resolved=replace(resolved, data=(("operation_id", "second"),))
    )
    with pytest.raises(ValueError, match="concurrent operations"):
        source.begin(other, context, torch.ones(2, dtype=torch.bool))
    assert source.current is first
    assert source.owner == "first"
    assert source.observed_checks == {(3, "tracking")}


def test_port_stage_replaces_measured_tilt_with_selected_downward_orientation():
    service, _, _ = _runtime()
    original_fk = service.robot.compute_fk
    cosine = math.sqrt(0.5)
    # A physical tool tilted by 45 degrees in both pitch and roll.
    measured_rotation = torch.tensor(
        [[cosine, 0.5, 0.5], [0.0, cosine, -cosine], [-cosine, 0.5, 0.5]]
    )

    def tilted_fk(qpos, **kwargs):
        pose = original_fk(qpos, **kwargs)
        pose[:, :3, :3] = measured_rotation
        return pose

    service.robot.compute_fk = tilted_fk
    service.port_yaw["left_arm"] = torch.tensor([math.pi / 2, -math.pi / 2])
    target = replace(
        service.cfg.target("stage_free_end"), position=(0.586, 0.047, 0.815)
    )
    pose = service.pose(target, "left_arm")
    rotation = pose[:, :3, :3]
    # The two reachable wrist branches point down with opposite channel yaw.
    expected_rotation = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ]
    )
    torch.testing.assert_close(rotation, expected_rotation)
    torch.testing.assert_close(
        rotation.transpose(1, 2) @ rotation, torch.eye(3).repeat(2, 1, 1)
    )
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2))
    torch.testing.assert_close(
        pose[:, :3, 3], torch.tensor([[0.586, 0.047, 0.815]]).repeat(2, 1)
    )


def test_seated_rows_preserve_contact_loading_command_while_other_rows_move():
    service, _, measured = _runtime()
    commanded = measured.clone()
    measured[3] = torch.tensor([0.585, 0.034, 0.781])
    measured[1] = torch.tensor([0.585, 0.033, 0.785])
    commanded[3] = torch.tensor([0.586, 0.035, 0.775])
    commanded[1] = torch.tensor([0.587, 0.034, 0.780])
    service.robot.get_qpos = lambda *, target=False: commanded if target else measured
    service.seated = lambda: torch.tensor([True, False])
    target = replace(
        service.cfg.target("lower_channel_3"), position=(None, None, 0.760)
    )

    pose = service.pose(target, "left_arm")

    # The loaded command remains intact for the seated row; the other row
    # retains its commanded XY and receives the next requested height.
    torch.testing.assert_close(
        pose[:, :3, 3],
        torch.tensor([[0.586, 0.035, 0.775], [0.587, 0.034, 0.760]]),
    )
    service.cfg = replace(service.cfg, targets=(target,))
    lowerer = _CableMoveLowerer(service.cfg)
    lowerer.runtime = service
    endpoint = SimpleNamespace(runtime_target=SimpleNamespace(control_part="left_arm"))
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.cable_move", arguments={"target": target.target_id}
        ),
        context=SimpleNamespace(require_control_dt=lambda: 1 / 60),
        bound=bound,
        option_template=MoveEndEffectorOptions(),
    )
    path = result.goal.xpos
    # Every command for the seated row must retain its loaded pose, including
    # the first interpolation sample. Returning to the measured pose unloads
    # the contact even if the final sample restores the original command.
    torch.testing.assert_close(path[0], pose[0].expand_as(path[0]))
    torch.testing.assert_close(path[1, -1], pose[1])
    assert bool((path[1, :, 2, 3].diff() < 0).all())
    assert path[1, 0, 2, 3] < measured[1, 2]
    assert path[1, 0, 2, 3] > commanded[1, 2]


@pytest.mark.parametrize("target_id", ["align_tip_3_0", "align_tip_3_1"])
def test_terminal_tip_correction_has_downward_travel_and_does_not_stop_at_gate(
    target_id,
):
    service, state, measured = _runtime()
    commanded = measured.clone()
    measured[[3, 1]] = torch.tensor([0.586, 0.05, 0.7706])
    commanded[[3, 1]] = torch.tensor([0.586, 0.05, 0.7685])
    service.robot.get_qpos = lambda *, target=False: commanded if target else measured
    # Reproduce a rejected 760.756 mm endpoint and an accepted endpoint only
    # 0.2 mm below the gate. Both still need to move toward the interior target.
    tip_heights = (0.760756, 0.7588)
    for row, height in zip((3, 1), tip_heights):
        # The fixture's last capsule points along +X and is 12 mm long.
        state[row, -1, :3] = torch.tensor([0.580, 0.077, height])
        state[row, -1, :3] += service.origins[0 if row == 3 else 1]
    service.seated = lambda: torch.tensor([False, True])

    pose = service.pose(service.cfg.target(target_id), "left_arm")

    # Keep 3 mm of downward travel below the last nominal height stage.
    # Reaching the release gate must not freeze the second row at the boundary.
    torch.testing.assert_close(
        pose[:, 2, 3] - commanded[[3, 1], 2],
        torch.tensor([-0.003, -0.003]),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(pose[:, :2, 3], commanded[[3, 1], :2], atol=4e-6, rtol=0)


def _pitch_runtime(tool_pitch, ray_pitch):
    service, state, measured = _runtime()
    value = _declaration()
    value["targets"] = {
        "pitch": {
            "kind": "pitch_align",
            "material_distance": 0.162,
            "pitch": -math.pi / 12,
        }
    }
    service.cfg = _decode_cable_service(value)
    service.port_yaw["left_arm"] = torch.tensor([math.pi / 2, -math.pi / 2])
    branch_rotation = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ]
    )
    c, s = math.cos(tool_pitch), math.sin(tool_pitch)
    rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    initial_rotation = rotation @ branch_rotation
    original_fk = service.robot.compute_fk

    def fk(qpos, **kwargs):
        pose = original_fk(qpos, **kwargs)
        pose[:, :3, :3] = initial_rotation
        return pose

    service.robot.compute_fk = fk
    measured[[3, 1]] = torch.tensor([0.586, 0.04, 0.795])
    point = torch.tensor([0.586, 0.05, 0.785])
    direction = torch.tensor(
        [0.0, 0.04 * math.cos(ray_pitch), 0.04 * math.sin(ray_pitch)]
    )
    for selected_row, native_row in enumerate((3, 1)):
        state[native_row, 13, :3] = point + service.origins[selected_row]
        state[native_row, -1, :3] = (
            point
            + direction
            - torch.tensor([0.006, 0.0, 0.0])
            + service.origins[selected_row]
        )
    return service, state


def _lower_pitch(service, target="pitch"):
    lowerer = _CableMoveLowerer(service.cfg)
    lowerer.runtime = service
    endpoint = SimpleNamespace(runtime_target=SimpleNamespace(control_part="left_arm"))
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )
    return lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.cable_move", arguments={"target": target}
        ),
        context=SimpleNamespace(
            env_ids=service.env_ids, require_control_dt=lambda: 1 / 60
        ),
        bound=bound,
        option_template=MoveEndEffectorOptions(),
    )


def test_declared_angular_speed_changes_duration_without_changing_pivot_goal():
    service, _ = _pitch_runtime(0, math.pi / 6)
    ordinary = _lower_pitch(service).goal.xpos
    target = service.cfg.target("pitch")
    assert target.angular_speed == 0.5
    service.cfg = replace(service.cfg, targets=(replace(target, angular_speed=1.0),))
    accelerated = _lower_pitch(service).goal.xpos
    # A ten-degree rotation at 60 Hz takes ceil(20.94) / ceil(10.47) samples.
    assert ordinary.shape[1] == 21
    assert accelerated.shape[1] == 11
    torch.testing.assert_close(ordinary[:, -1], accelerated[:, -1])


@pytest.mark.parametrize("rate", [0.0, -0.5])
def test_cable_angular_speed_requires_a_positive_rate(rate):
    value = _declaration()
    value["targets"] = {"home": {"kind": "home", "angular_speed": rate}}
    with pytest.raises(ValueError, match="positive"):
        _decode_cable_service(value)


def test_cable_hold_cannot_declare_a_motion_angular_speed():
    value = _declaration()
    value["targets"] = {"hold": {"kind": "hold", "samples": 15, "angular_speed": 1.0}}
    with pytest.raises(ValueError, match="Unsupported fields"):
        _decode_cable_service(value)


@pytest.mark.parametrize(
    "tool_degrees,ray_degrees,expected_degrees",
    [(0, 30, -10), (-15, 30, -20), (-25, 30, -25), (0, -30, 0)],
)
def test_pitch_alignment_bounds_both_wrist_branches_and_keeps_material_pivot(
    tool_degrees, ray_degrees, expected_degrees
):
    service, _ = _pitch_runtime(math.radians(tool_degrees), math.radians(ray_degrees))
    start = service.tool("left_arm")
    point = service.point(0.162)[0] - service.origins
    local_point = start[:, :3, :3].transpose(1, 2) @ (
        point - start[:, :3, 3]
    ).unsqueeze(-1)

    result = _lower_pitch(service)

    assert not result.failed_preconditions
    path = result.goal.xpos
    rotations = path[:, :, :3, :3]
    predicted_point = (rotations @ local_point[:, None]).squeeze(-1) + path[:, :, :3, 3]
    torch.testing.assert_close(
        predicted_point, point[:, None].expand_as(predicted_point)
    )
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations, torch.eye(3).expand_as(rotations)
    )
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(rotations.shape[:2])
    )
    forward = rotations[:, -1, :, 0] * torch.tensor([1.0, -1.0])[:, None]
    torch.testing.assert_close(
        torch.atan2(forward[:, 2], forward[:, 1]),
        torch.full((2,), math.radians(expected_degrees)),
        atol=1e-6,
        rtol=0,
    )


def test_pitch_alignment_rejects_only_the_row_with_backward_distal_span():
    service, state = _pitch_runtime(0, math.pi / 6)
    state[1, -1, 1] = 0.01
    initial = service.tool("left_arm")

    result = _lower_pitch(service)

    assert set(result.failed_preconditions) == {1}
    torch.testing.assert_close(
        result.goal.xpos[1], initial[1].expand_as(result.goal.xpos[1])
    )
    assert not torch.allclose(result.goal.xpos[0, -1], initial[0])


def test_pitch_alignment_preserves_seated_rows_loaded_command():
    service, _ = _pitch_runtime(0, math.pi / 6)
    measured = service.robot.get_qpos().clone()
    commanded = measured.clone()
    commanded[[3, 1], 2] -= 0.003
    service.robot.get_qpos = lambda *, target=False: commanded if target else measured
    target = replace(service.cfg.target("pitch"), stop_when_seated=True)
    service.cfg = replace(service.cfg, targets=(target,))
    service.seated = lambda: torch.tensor([True, False])
    loaded = service.tool("left_arm", commanded=True)

    result = _lower_pitch(service)

    assert torch.equal(result.goal.xpos[0], loaded[0].expand_as(result.goal.xpos[0]))
    assert not torch.allclose(result.goal.xpos[1, -1], loaded[1])


def _lay_down_runtime(tool_degrees=-20, minimum_z=0.7625):
    service, state = _pitch_runtime(math.radians(tool_degrees), -math.pi / 12)
    value = _declaration()
    value["targets"] = {"lay": {"kind": "lay_down", "minimum_z": minimum_z}}
    service.cfg = _decode_cable_service(value)
    service.robot.get_qpos()[[3, 1]] = torch.tensor([0.586, 0.04, 0.7655])
    tip = torch.tensor([0.586, 0.077, 0.756])
    for row, native_row in enumerate((3, 1)):
        state[native_row, -1, :3] = (
            tip - torch.tensor([0.006, 0.0, 0.0]) + service.origins[row]
        )
    service.seated = lambda: torch.ones(2, dtype=torch.bool)
    service.grasps["left_arm"] = (
        0.162,
        service.point_in_tool("left_arm", 0.162).clone(),
    )
    return service, state


@pytest.mark.parametrize("tool_degrees,expected_degrees", [(-20, -10), (-5, 0), (0, 0)])
def test_lay_down_returns_both_wrist_branches_toward_flat_without_overshoot(
    tool_degrees, expected_degrees
):
    service, _ = _lay_down_runtime(tool_degrees, minimum_z=0.74)
    initial = service.tool("left_arm")

    result = _lower_pitch(service, "lay")

    assert not result.failed_preconditions
    forward = result.goal.xpos[:, -1, :3, 0] * torch.tensor([1.0, -1.0])[:, None]
    torch.testing.assert_close(
        torch.atan2(forward[:, 2], forward[:, 1]),
        torch.full((2,), math.radians(expected_degrees)),
        atol=1e-6,
        rtol=0,
    )
    if tool_degrees == 0:
        torch.testing.assert_close(
            result.goal.xpos, initial[:, None].expand_as(result.goal.xpos)
        )


def test_lay_down_limits_rotation_at_floor_and_preserves_tip_in_every_sample():
    service, _ = _lay_down_runtime()
    initial = service.tool("left_arm")
    tip = cable_endpoints(service.state(), service.lengths)[1][:, -1] - service.origins
    local_tip = initial[:, :3, :3].transpose(1, 2) @ (
        tip - initial[:, :3, 3]
    ).unsqueeze(-1)

    result = _lower_pitch(service, "lay")

    assert not result.failed_preconditions
    path = result.goal.xpos
    transformed_tip = (path[:, :, :3, :3] @ local_tip[:, None]).squeeze(-1) + path[
        :, :, :3, 3
    ]
    torch.testing.assert_close(transformed_tip, tip[:, None].expand_as(transformed_tip))
    # The floor is reached before the ten-degree rotation allowance; clamping
    # TCP translation after rotation would violate the fixed-tip check above.
    torch.testing.assert_close(path[:, -1, 2, 3], torch.full((2,), 0.7625))
    assert bool((path[:, :, 2, 3] >= 0.7625 - 1e-7).all())
    assert bool((path[:, 1:, 2, 3] <= path[:, :-1, 2, 3] + 1e-7).all())
    forward = path[:, -1, :3, 0] * torch.tensor([1.0, -1.0])[:, None]
    final_pitch = torch.atan2(forward[:, 2], forward[:, 1])
    assert bool((final_pitch > math.radians(-20)).all())
    assert bool((final_pitch < math.radians(-10)).all())


@pytest.mark.parametrize("below_floor", [0.0, 0.000005, 0.00009])
def test_lay_down_at_floor_holds_loaded_command_without_unloading(below_floor):
    service, _ = _lay_down_runtime()
    measured = service.robot.get_qpos().clone()
    measured[[3, 1], 2] = 0.7625 - below_floor
    commanded = measured.clone()
    commanded[[3, 1], 2] = 0.7625
    commanded[[3, 1], 1] += 0.001
    service.robot.get_qpos = lambda *, target=False: commanded if target else measured
    loaded = service.tool("left_arm", commanded=True)

    result = _lower_pitch(service, "lay")

    assert not result.failed_preconditions
    assert torch.equal(result.goal.xpos, loaded[:, None].expand_as(result.goal.xpos))


@pytest.mark.parametrize(
    "invalid", ["unseated", "backward", "nonfinite", "pivot", "below_floor"]
)
def test_lay_down_rejects_only_invalid_environment_and_keeps_its_pose(invalid):
    service, state = _lay_down_runtime()
    if invalid == "unseated":
        service.seated = lambda: torch.tensor([True, False])
    elif invalid == "backward":
        service.port_yaw["left_arm"][1] = math.pi / 2
    elif invalid == "nonfinite":
        state[1, -1, 0] = float("nan")
    elif invalid == "pivot":
        state[1, -1, 1] = 0.03
    else:
        service.robot.get_qpos()[1, 2] = 0.76239
    initial = service.tool("left_arm")

    result = _lower_pitch(service, "lay")

    assert service.env_ids.tolist() == [3, 1]
    assert set(result.failed_preconditions) == {1}
    assert torch.equal(result.goal.xpos[1], initial[1].expand_as(result.goal.xpos[1]))
    assert not torch.allclose(result.goal.xpos[0, -1], initial[0])


def test_lay_down_preserves_active_grasp_and_original_slip_reference():
    service, state = _lay_down_runtime()
    baseline = service.point_in_tool("left_arm", 0.162).clone()
    reference = (0.162, baseline)
    service.grasps["left_arm"] = reference
    service.hand_to_motion["left_eef"] = "left_arm"
    service.grasp_slip["left_arm"] = torch.tensor([0.001, 0.002])
    service.max_slip[:] = service.grasp_slip["left_arm"]
    pending = ("right_arm", service.cfg.target("lay"))
    service.pending_pick = pending

    _lower_pitch(service, "lay")

    assert service.grasps["left_arm"] is reference
    assert service.grasps["left_arm"][1] is baseline
    assert service.pending_pick is pending
    torch.testing.assert_close(service.max_slip, torch.tensor([0.001, 0.002]))
    torch.testing.assert_close(service.grasp_slip["left_arm"], service.max_slip)
    state[3, :, 0] += 0.013
    assert service.observe().tolist() == [True, False]


def test_lay_down_rejects_a_motion_part_without_an_active_grasp():
    service, _ = _lay_down_runtime()
    service.grasps.pop("left_arm")
    with pytest.raises(ValueError, match="active grasp"):
        _lower_pitch(service, "lay")


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_z": None},
        {"minimum_z": float("nan")},
        {"maximum_rotation": 0},
        {"maximum_rotation": math.pi / 6},
        {"maximum_rotation": float("inf")},
        {"stop_when_seated": True},
    ],
)
def test_lay_down_requires_explicit_floor_and_bounded_unfrozen_rotation(changes):
    value = _declaration()
    target = {"kind": "lay_down", "minimum_z": 0.7625}
    target.update(changes)
    if target.get("minimum_z") is None:
        target.pop("minimum_z")
    value["targets"] = {"lay": target}
    with pytest.raises((ValueError, TypeError)):
        _decode_cable_service(value)


def _grip_adjust_runtime():
    service, state, _ = _runtime()
    value = _declaration()
    value["targets"]["adjust"] = {
        "kind": "grip_adjust",
        "samples": 60,
        "finger_position": 0.003,
    }
    service.cfg = _decode_cable_service(value)
    return service, state


def _lower_grip_adjust(service):
    lowerer = _CableJointLowerer(service.cfg)
    lowerer.runtime = service
    endpoint = SimpleNamespace(runtime_target=SimpleNamespace(control_part="left_eef"))
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )
    return lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.cable_joint", arguments={"target": "adjust"}
        ),
        context=SimpleNamespace(env_ids=service.env_ids),
        bound=bound,
        option_template=MoveJointsOptions(),
    )


def test_grip_adjust_preserves_grasp_reference_and_slip_guard():
    service, state = _grip_adjust_runtime()
    service.hand_to_motion["left_eef"] = "left_arm"
    baseline = service.point_in_tool("left_arm", 0.162).clone()
    reference = (0.162, baseline)
    service.grasps["left_arm"] = reference
    service.grasp_slip["left_arm"] = torch.tensor([0.001, 0.002])
    service.max_slip[:] = service.grasp_slip["left_arm"]
    original_max_slip = service.max_slip
    pending = ("right_arm", service.cfg.target("right_grasp_0.162"))
    service.pending_pick = pending

    result = _lower_grip_adjust(service)

    assert service.grasps["left_arm"] is reference
    assert service.grasps["left_arm"][1] is baseline
    assert service.pending_pick is pending
    assert service.max_slip is original_max_slip
    torch.testing.assert_close(service.max_slip, torch.tensor([0.001, 0.002]))
    torch.testing.assert_close(
        service.grasp_slip["left_arm"], torch.tensor([0.001, 0.002])
    )
    torch.testing.assert_close(result.goal.target[:, 0], torch.zeros(2, 3))
    torch.testing.assert_close(result.goal.target[:, -1], torch.full((2, 3), 0.003))
    assert (
        "adjust" in _CableJointFactory(service.cfg).argument_contract.required["target"]
    )
    assert (
        "adjust"
        not in _CableMoveFactory(service.cfg).argument_contract.required["target"]
    )
    # Adjustment must not rebase or disable the reference when contact slips.
    state[3, :, 0] += 0.013
    assert service.observe().tolist() == [True, False]


@pytest.mark.parametrize("missing", ["mapping", "grasp"])
def test_grip_adjust_requires_current_hand_grasp(missing):
    service, _ = _grip_adjust_runtime()
    if missing == "mapping":
        service.grasps["left_arm"] = (0.162, service.point_in_tool("left_arm", 0.162))
    else:
        service.hand_to_motion["left_eef"] = "left_arm"
    with pytest.raises(ValueError, match="active grasp"):
        _lower_grip_adjust(service)


def test_grip_adjust_rejects_release_semantics():
    value = _declaration()
    value["targets"] = {"adjust": {"kind": "grip_adjust", "release_part": "left_arm"}}
    with pytest.raises(ValueError, match="Unsupported fields"):
        _decode_cable_service(value)


def _lower_check(service, target):
    lowerer = _CableCheckLowerer(service.cfg)
    lowerer.runtime = service
    endpoint = SimpleNamespace(runtime_target=SimpleNamespace(control_part="left_arm"))
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )
    return lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.cable_check", arguments={"target": target}
        ),
        context=SimpleNamespace(env_ids=service.env_ids),
        bound=bound,
        option_template=MoveJointsOptions(),
    )


def test_cable_tracking_rejects_wrong_orientation_at_same_position():
    service, _, _ = _runtime()
    target = service.tool("left_arm")
    target[:, :2, :2] *= -1
    service.last_targets["left_arm"] = target
    failures = _lower_check(service, "stage_free_end_tracking").failed_preconditions
    assert set(failures) == {3, 1}
    assert "orientation_error_rad" in failures[3]


@pytest.mark.parametrize("tracking_error", ["position", "orientation"])
def test_release_alignment_above_channel_does_not_bypass_tcp_tracking(tracking_error):
    service, _, _ = _runtime()
    service.last_targets["left_arm"] = service.tool("left_arm")
    # Both rows are aligned for release, but only the first is physically
    # inside the channel. Both tools miss the 6 mm or 0.25 rad tracking limit.
    if tracking_error == "position":
        service.last_targets["left_arm"][:, 0, 3] += 0.007
    else:
        service.last_targets["left_arm"][:, :2, :2] = torch.tensor(
            [[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]]
        )
    service.seated = lambda: torch.tensor([True, True])
    service.metrics = lambda: {"strict_port": torch.tensor([True, False])}
    failures = _lower_check(service, "lower_channel_3_tracking").failed_preconditions
    assert set(failures) == {1}
    # The hovered row can proceed once it meets the original tracking limit;
    # the inserted row may retain its contact-loaded command.
    if tracking_error == "position":
        service.last_targets["left_arm"][1, 0, 3] = 0.005
    else:
        service.last_targets["left_arm"][1, :2, :2] = torch.tensor(
            [[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]]
        )
    assert not _lower_check(service, "lower_channel_3_tracking").failed_preconditions


@pytest.mark.parametrize("prior_slip", [False, True])
def test_cable_check_allows_surviving_row_to_execute_next_segment(prior_slip):
    from test_simulation_environment import _DualRobot, _motion_generator
    from embodichain.lab.task_program.integrations import SimulationTaskProgramFactory

    # Run the deployed YAM check and two short holds through the actual semantic
    # executor, command sink and Gym bridge, with selected physical rows 3, 1.
    deployment = _deployment()
    full = deployment.load_program()
    first = full.program.items[0]
    _, hold, check = first.steps.items
    segments = tuple(
        replace(first, name=name, steps=node)
        for name, node in (("before", hold), ("check", check), ("after", hold))
    )
    program = replace(full, program=replace(full.program, items=segments))
    service, _, _ = _runtime()
    robot = _DualRobot()
    robot._joint_ids = {
        "left_arm": (0,),
        "right_arm": (1,),
        "left_eef": (2, 3),
        "right_eef": (4, 5),
    }
    robot.control_parts = {
        part: tuple(f"joint_{i}" for i in ids) for part, ids in robot._joint_ids.items()
    }
    robot.dof = 6
    robot.qpos = torch.zeros(4, robot.dof)
    robot.body_data = service.robot.body_data
    simulation = service.simulation
    simulation.get_robot = lambda uid: robot if uid == robot.uid else None
    generator = _motion_generator(robot)
    generator._env_ids = (3, 1)
    generator.planner.cfg.planner_type = "trapezoidal"
    generator.joint_steps_within_velocity_limits.return_value = torch.ones(
        2, dtype=torch.bool
    )
    factory = SimulationTaskProgramFactory(
        simulation,
        robot,
        deployment.integration.registration,
        step_dt=1 / 60,
        motion_generator_factory=lambda: generator,
    )
    adapter = factory.for_env_ids((3, 1)).create_adapter()
    bridge = adapter.create_bridge(adapter.compile(program))
    cable = bridge.post_policy_port._cable_runtime
    cable.last_targets["right_arm"] = cable.tool("right_arm")
    if prior_slip:
        cable.max_slip[0] = 0.013
    else:
        cable.last_targets["right_arm"][0, 0, 3] += 0.1

    accepted = []
    rejected_ids = []
    for index, segment in enumerate(bridge.iter_segments()):
        commands = list(segment.actions)
        assert any(
            a.metadata["bridge_action_kind"] == "runtime_command" for a in commands
        )
        rejected_ids.extend(
            i for a in commands for i in a.metadata.get("rejected_env_ids", ())
        )
        accepted.append(segment.validator().tolist())
        call = segment.metadata["runtime"]["calls"][0]
        if index == 1:
            diagnostics = call["plan_attempts"][-1]["planner_diagnostics"]
            if prior_slip:
                assert call["masks"]["entered"] == [False, True]
                assert not diagnostics["messages"]
            else:
                assert call["masks"]["failed"] == [True, False]
                assert diagnostics["failure"]["code"] == "semantic_precondition_failed"
                assert diagnostics["messages"][0].startswith("env 3: Cable condition")
        if index == 2:
            assert call["masks"]["entered"] == [False, True]
            assert call["masks"]["completed"] == [False, True]
    assert accepted == [[not prior_slip, True], [False, True], [False, True]]
    # The earlier slip went through the bridge's physical guard. A new check
    # failure is reported directly by the semantic executor during planning.
    assert rejected_ids == ([3] if prior_slip else [])


def test_cable_observation_rejects_invalid_quaternion_per_row():
    service, state, _ = _runtime()
    state[1, :, 3:7] = 0
    assert service.observe().tolist() == [False, True]


def test_cable_session_scope_is_checked_before_cached_measurements():
    service, _, _ = _runtime()
    service.simulation.simulation_time = 1.0
    service.metrics()

    def invalid_scope():
        raise RuntimeError("episode changed")

    service._scope_guard = invalid_scope
    with pytest.raises(RuntimeError, match="episode changed"):
        service.metrics()


def test_cable_factories_return_fresh_ready_lowerers():
    service, _, _ = _runtime()
    factory = _CablePickFactory(service.cfg)
    engine = SimpleNamespace(
        robot=service.robot,
        planning_services=SimpleNamespace(env_ids=(3, 1)),
        device=torch.device("cpu"),
    )
    first = factory.create(
        simulation=service.simulation,
        robot=service.robot,
        scene_registry=None,
        engine=engine,
    )
    second = factory.create(
        simulation=service.simulation,
        robot=service.robot,
        scene_registry=None,
        engine=engine,
    )
    assert first is not second and first.runtime is not second.runtime
    assert first.runtime.env_ids.tolist() == [3, 1]
