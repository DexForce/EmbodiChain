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

"""Target identity, simultaneous finger contact and unavailable evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain_benchmarks.motion_generation.contracts import (
    ExecutionObservation,
    ReplayObservationTrace,
)
from embodichain_benchmarks.motion_generation.scenarios._grasp_contacts import (
    GraspContacts,
    contact_configuration,
    minimum_target_contact_distances,
)
from embodichain_benchmarks.motion_generation.scenarios.atomic_evaluation import (
    gripper_holding,
    trace_validity,
)


def test_contact_reduction_filters_target_actor_order_padding_and_environment():
    data = {
        "user_ids": torch.tensor(
            [
                [[10, 20], [20, 11], [10, 20], [11, 99], [11, 20]],
                [[12, 21], [21, 13], [10, 20], [12, 99], [13, 21]],
                [[14, 22], [15, 23], [10, 20], [14, 22], [15, 22]],
            ]
        ),
        "distance": torch.tensor(
            [
                [0.004, -0.001, 0.002, -0.5, -0.5],
                [0.001, 0.005, -0.5, -0.5, -0.5],
                [0.003, -0.5, -0.5, -0.5, -0.5],
            ]
        ),
        "is_valid": torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, True, True, False],
                [True, True, True, False, False],
            ]
        ),
    }
    fingers = torch.tensor([[10, 11], [12, 13], [14, 15]])
    targets = torch.tensor([[20], [21], [22]])
    result = minimum_target_contact_distances(
        data, fingers, targets, torch.tensor([False, False, False])
    )
    torch.testing.assert_close(
        result, torch.tensor([[0.002, -0.001], [0.001, 0.005], [0.003, torch.inf]])
    )
    # Query truncation and invalid matched distances are absent evidence, not misses.
    data["distance"][0, 0] = torch.nan
    result = minimum_target_contact_distances(
        data, fingers, targets, torch.tensor([False, True, False])
    )
    assert torch.isnan(result[0, 0])
    assert torch.isnan(result[1]).all()
    assert torch.isposinf(result[2, 1])


def test_holding_requires_simultaneous_target_contacts_below_five_mm():
    distances = torch.full((2, 6, 2), 0.001)
    distances[0, 5] = torch.tensor([0.001, torch.inf])
    distances[-1] = torch.tensor(
        [
            [-0.001, 0.0049],
            [0.001, 0.005],
            [0.001, 0.006],
            [0.001, torch.inf],
            [torch.nan, torch.nan],
            [torch.inf, 0.001],
        ]
    )
    trace = ReplayObservationTrace(
        # Identical intermediate joint positions cannot certify any grasp.
        gripper_qpos={"hand": torch.full((2, 6, 1), 0.5)},
        gripper_contact_distances={"hand": distances},
        gripper_contact_targets={"hand": "target"},
    )
    observation = ExecutionObservation(
        execution_success=torch.ones(6, dtype=torch.bool),
        final_tcp_pose=torch.eye(4).repeat(6, 1, 1),
        joint_tracking_rmse_rad=torch.zeros(6),
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
    )
    assert gripper_holding(trace, observation, "hand", object_id="target").tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert trace_validity(
        trace, observation=observation, contact_parts=("hand",)
    ).tolist() == [True, True, True, True, False, True]
    assert gripper_holding(trace, observation, "hand", object_id="other") is None
    strict = ExecutionObservation(
        execution_success=observation.execution_success,
        final_tcp_pose=observation.final_tcp_pose,
        joint_tracking_rmse_rad=observation.joint_tracking_rmse_rad,
        execution_time_ms=1.0,
        task_completion_time_s=0.1,
        gripper_contact_thresholds_m={"hand": 0.002},
    )
    assert not gripper_holding(trace, strict, "hand", object_id="target").any()
    # The immutable trace owns its samples.
    distances.zero_()
    assert not gripper_holding(trace, observation, "hand", object_id="target")[1]


def test_contact_binding_uses_each_grippers_target_object(monkeypatch):
    from embodichain.lab.sim import sensors

    state = {"wrong_target": False}

    class Sensor:
        contact_capabilities = SimpleNamespace(geometry=True)
        dropped_contacts_mask = torch.tensor([False])

        def __init__(self, cfg, **kwargs):
            assert cfg.filter_need_both_actor

        def get_actor_ids(self, uid, links=None):
            return {
                ("robot", ("l1", "l2")): torch.tensor([[10, 11]]),
                ("robot", ("r1", "r2")): torch.tensor([[12, 13]]),
                ("placing", None): torch.tensor([[100]]),
                ("support", None): torch.tensor([[200]]),
            }[(uid, None if links is None else tuple(links))]

        def update(self):
            pass

        def get_data(self):
            target = 100 if state["wrong_target"] else 200
            return {
                "user_ids": torch.tensor(
                    [[[10, 100], [100, 11], [12, target], [target, 13]]]
                ),
                "distance": torch.zeros(1, 4),
                "is_valid": torch.ones(1, 4, dtype=torch.bool),
            }

    monkeypatch.setattr(sensors, "ContactSensor", Sensor)
    simulation = SimpleNamespace(physics=SimpleNamespace(supports_contact_sensor=True))
    robot = SimpleNamespace(
        uid="robot", link_names=["l1", "l2", "r1", "r2"], device=torch.device("cpu")
    )
    config = {
        "grippers": [
            {"control_part": "left", "finger_links": ["l1", "l2"]},
            {"control_part": "right", "finger_links": ["r1", "r2"]},
        ]
    }
    case = SimpleNamespace(
        batch_size=1,
        object_id="placing",
        skill_id="coordinated_placement",
        case_parameters={
            "control_parts": {"support": {"grasp": "right"}},
            "support_object_id": "support",
        },
    )
    objects = {
        name: SimpleNamespace(entity=SimpleNamespace(uid=name))
        for name in ("placing", "support")
    }
    contacts = GraspContacts(
        simulation, robot, config, case, objects, ("left", "right")
    )
    first = contacts.sample()
    assert (first["right"] == 0).all()
    assert contacts.targets == {"left": "placing", "right": "support"}
    assert contacts.thresholds == {"left": 0.005, "right": 0.005}
    state["wrong_target"] = True
    assert torch.isposinf(contacts.sample()["right"]).all()
    assert (first["right"] == 0).all()
    simulation.physics.supports_contact_sensor = False
    missing = GraspContacts(simulation, robot, config, case, objects, ("left", "right"))
    assert torch.isnan(missing.sample()["left"]).all()
    assert missing.metadata()["unavailable"] == {
        "left": "backend_unsupported",
        "right": "backend_unsupported",
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"finger_links": ["a", "a"]},
        {"finger_links": ["a"]},
        {"contact_distance_threshold_m": 0.0},
        {"contact_distance_threshold_m": float("nan")},
    ],
)
def test_contact_configuration_rejects_unusable_fingers_and_thresholds(settings):
    with pytest.raises(ValueError):
        contact_configuration({"gripper": {"control_part": "eef", **settings}})


@pytest.mark.requires_sim
@pytest.mark.gpu
def test_newton_finger_contact_scoring_is_target_and_environment_specific():
    """Exercise real articulated finger identities, not only rigid-body contacts."""
    import warp as wp
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain_benchmarks.motion_generation.config import RobotSpecCfg
    from embodichain_benchmarks.motion_generation.robots.franka import (
        FrankaPgiProvider,
    )

    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Newton contact scoring requires CUDA.")
    simulation = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=1,
                use_cuda_graph=False,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
        )
    )
    try:
        robot = simulation.add_robot(
            cfg=FrankaPgiProvider(
                RobotSpecCfg(id="franka_pgi", provider="franka_pgi")
            ).build_cfg()
        )
        target = simulation.add_rigid_object(
            RigidObjectCfg(
                uid="contact_target",
                shape=CubeCfg(size=[0.04, 0.04, 0.04]),
                body_type="kinematic",
                init_pos=[2.0, 0.0, 2.0],
            )
        )
        simulation.prepare()
        pose = robot.compute_fk(robot.get_qpos(name="arm"), name="arm", to_matrix=True)
        pose[1, 0, 3] += 0.3
        target.set_local_pose(pose)
        robot.set_qpos(
            torch.full((2, 1), 0.04, device=robot.device), name="eef", target=True
        )
        contacts = GraspContacts(
            simulation,
            robot,
            {
                "gripper": {
                    "control_part": "eef",
                    "finger_links": [
                        "gripper_finger1_link_1",
                        "gripper_finger2_link_1",
                    ],
                }
            },
            SimpleNamespace(
                batch_size=2, object_id="cube", skill_id="pick_up", case_parameters={}
            ),
            {"cube": SimpleNamespace(entity=target)},
            ("eef",),
        )
        simulation.update(step=200)
        sample = contacts.sample()["eef"]
        assert contacts.metadata()["unavailable"] == {}
        assert (sample[0] < 0.005).all(), sample
        assert torch.isposinf(sample[1]).all(), sample
    finally:
        simulation.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()
