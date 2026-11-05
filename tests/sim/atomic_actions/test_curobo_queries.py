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


"""Atomic joint-prefix policy on the shared observed cuRobo query service."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload


@pytest.mark.parametrize("backend_failure", [False, True])
def test_atomic_joint_prefix_keeps_torch_boundary_and_exact_backend_options(
    backend_failure,
):
    from contextlib import contextmanager
    import random
    import numpy as np
    from embodichain.lab.sim.atomic_actions.curobo_queries import (
        CuroboJointPrefixPlanner,
    )
    from embodichain.lab.sim.atomic_actions.state import (
        PlanningContext,
        RobotObservation,
        SceneSnapshot,
        TaskState,
    )
    from embodichain.lab.sim.motion.planners.curobo.observed import (
        CuroboWorldObservation,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanResult

    robot = SimpleNamespace(
        joint_names=["joint", "passive"],
        control_parts={"arm": ["joint"]},
        root_link_name="root",
    )
    qpos = torch.tensor([[0.1, 0.7]])
    context = PlanningContext(
        RobotObservation(
            0.4, qpos, torch.zeros_like(qpos), root_pose=torch.eye(4)[None]
        ),
        TaskState(1, "cpu"),
        SceneSnapshot(0.4, 7),
        torch.tensor([0]),
        control_dt=0.04,
    )
    world = CuroboWorldObservation(
        timestamp=0.4, scene_version=7, native_revision=15, poses={}
    )
    calls, observations = [], []

    def generate(targets, *, options):
        torch.rand(3)
        random.random()
        np.random.rand()
        calls.append(
            (
                options.strategy,
                options.sample_count,
                options.velocity_limit,
                options.acceleration_limit,
                options.plan_opts.max_velocity,
                options.plan_opts.max_acceleration,
            )
        )
        torch.testing.assert_close(targets[0].qpos, torch.tensor([[0.4]]))
        if backend_failure:
            raise RuntimeError("native planning failed")
        return PlanResult(
            success=torch.tensor([True]),
            positions=torch.tensor([[[0.1], [0.4]]]),
            dt=torch.tensor([[0.0, 0.04]]),
        )

    @contextmanager
    def opened(**kwargs):
        assert kwargs["world"] is world
        assert kwargs["root_link_name"] == "root"
        torch.testing.assert_close(kwargs["full_qpos"], qpos)
        calls.append("open")
        try:
            yield SimpleNamespace(generate=generate)
        finally:
            calls.append("close")

    queries = SimpleNamespace(
        robot=robot, planning_device=torch.device("cpu"), open=opened
    )
    provider = SimpleNamespace(
        capture=lambda value: observations.append(value) or world
    )
    adapter = CuroboJointPrefixPlanner(queries, provider)
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        py_reference, np_reference = random.Random(100), np.random.RandomState(100)
        py_reference.random()
        np_reference.rand()
        before = torch.get_rng_state().clone()
        kwargs = dict(
            context=context,
            control_part="arm",
            start_qpos=qpos[:, :1],
            target_qpos=torch.tensor([[0.4]]),
            full_qpos=qpos,
            control_dt=0.04,
        )
        if backend_failure:
            with pytest.raises(RuntimeError, match="native planning failed"):
                adapter.plan(**kwargs)
        else:
            result = adapter.plan(**kwargs)
            torch.testing.assert_close(result.positions, torch.tensor([[[0.1], [0.4]]]))
        assert torch.equal(before, torch.get_rng_state())
        assert random.random() == py_reference.random()
        assert np.random.rand() == np_reference.rand()
        assert calls == ["open", ("motion_gen", None, None, None, None, None), "close"]
        wrong = replace(context, robot=replace(context.robot, root_pose=None))
        with pytest.raises(ValueError, match="captured robot root"):
            adapter.plan(**{**kwargs, "context": wrong})
        assert len(observations) == 1
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])


def test_prefix_and_payload_capabilities_share_one_adapter_without_resolving_device():
    from embodichain.lab.sim.atomic_actions.curobo_queries import (
        CuroboJointPrefixPlanner,
    )
    from embodichain.lab.sim.atomic_actions.runtime import ActionPlanningServices

    robot = object()

    class Queries:
        @property
        def robot(self):
            return robot

        @property
        def planning_device(self):
            pytest.fail("capability inspection resolved CUDA")

    adapter = CuroboJointPrefixPlanner(
        Queries(), SimpleNamespace(capture=lambda context: None)
    )
    services = ActionPlanningServices(
        SimpleNamespace(robot=robot, device=torch.device("cpu")),
        joint_prefix_planner=adapter,
    )
    assert services.joint_prefix_planner is adapter
    assert services.curobo_queries is adapter
    prefix_only = SimpleNamespace(robot=robot, plan=lambda **kwargs: None)
    other = ActionPlanningServices(
        SimpleNamespace(robot=robot, device=torch.device("cpu")),
        joint_prefix_planner=prefix_only,
    )
    assert other.joint_prefix_planner is prefix_only
    assert other.curobo_queries is None


def test_payload_query_excludes_its_observed_pose_before_world_capture():
    from contextlib import contextmanager
    from embodichain.lab.sim.atomic_actions.curobo_queries import (
        CuroboJointPrefixPlanner,
    )
    from embodichain.lab.sim.atomic_actions.state import (
        PlanningContext,
        RobotObservation,
        SceneSnapshot,
        TaskState,
    )
    from embodichain.lab.sim.motion.planners.curobo.observed import (
        CuroboWorldObservation,
    )

    robot = SimpleNamespace(
        root_link_name="root", joint_names=["j"], control_parts={"arm": ["j"]}
    )
    context = PlanningContext(
        RobotObservation(
            0.0, torch.zeros(1, 1), torch.zeros(1, 1), root_pose=torch.eye(4)[None]
        ),
        TaskState(1, "cpu"),
        SceneSnapshot(0.0, 0),
        torch.tensor([0]),
        control_dt=0.04,
    )
    calls = []

    def capture(value, *, excluded_entity_ids=()):
        assert excluded_entity_ids == ("carried",)
        calls.append("world_without_payload")
        return CuroboWorldObservation(
            timestamp=0.0,
            scene_version=0,
            native_revision=0,
            poses={"table": torch.eye(4)[None]},
        )

    @contextmanager
    def opened(**kwargs):
        assert tuple(kwargs["world"].poses) == ("table",)
        assert kwargs["payload"].object_id == "carried"
        calls.append("open")
        yield SimpleNamespace(robot=robot)

    service = CuroboJointPrefixPlanner(
        SimpleNamespace(robot=robot, open=opened), SimpleNamespace(capture=capture)
    )
    payload = CuroboPayload(
        object_id="carried",
        control_part="arm",
        spheres=torch.tensor([[0.0, 0.0, 0.0, 0.1]]),
        object_pose=torch.eye(4)[None],
    )
    query = service.open(
        context=context,
        control_part="arm",
        full_qpos=torch.zeros(1, 1),
        payload=payload,
    )
    assert calls == ["world_without_payload"]
    with query:
        pass
    assert calls == ["world_without_payload", "open"]
