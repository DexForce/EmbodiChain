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
from pathlib import Path
from types import SimpleNamespace
import math

import pytest
import torch
import yaml

from embodichain.lab.gym.utils.composition import resolve_gym_components
from embodichain.lab.task_program.integrations import (
    load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations.simulation.cable import (
    _decode_cable_service,
    _CableRuntime,
    _CableCheckLowerer,
    _CablePickFactory,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.sim.atomic_skills import MoveJointsOptions

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


def test_cable_deployment_preflights_all_calls_and_measured_final_acceptance():
    path = _TASK / "task.yam.yaml"
    raw = yaml.safe_load(path.read_text())
    physical = resolve_gym_components(raw, base_dir=path.parent)
    deployment = load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        evaluation=physical.evaluation,
        base_dir=path.parent,
    )
    physical.validate_scene_bindings(deployment.scene_binding)
    registration = deployment.integration.registration
    compiled = registration.catalog.preflight(deployment.load_program())
    assert compiled.segment_count == 49
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


def test_cable_tracking_rejects_wrong_orientation_at_same_position():
    service, _, _ = _runtime()
    target = service.tool("left_arm")
    target[:, :2, :2] *= -1
    service.last_targets["left_arm"] = target
    lowerer = _CableCheckLowerer(service.cfg)
    lowerer.runtime = service
    endpoint = SimpleNamespace(runtime_target=SimpleNamespace(control_part="left_arm"))
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={"primary": SimpleNamespace(endpoints={"motion": endpoint})}
        )
    )
    call = RegisteredSemanticCall(
        call_id="simulation.cable_check",
        arguments={"target": "stage_free_end_tracking"},
    )
    with pytest.raises(ValueError, match="orientation_error_rad"):
        lowerer.lower(
            call,
            context=SimpleNamespace(env_ids=service.env_ids),
            bound=bound,
            option_template=MoveJointsOptions(),
        )


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
