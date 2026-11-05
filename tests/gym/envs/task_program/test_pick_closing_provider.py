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

"""Native Pick assembly preserves exact scope, commands and owned cleanup."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch
import numpy as np
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.task_program.integrations.simulation import pick_closing as owner
from embodichain.lab.task_program.integrations.simulation.targets import (
    AbsolutePoseTarget,
)


def _setup(monkeypatch, *, revision=8, opening=False, downstream=False):
    stamp = PhysicalObservationStamp("owner", "default.contacts", 7, 1.0)
    current = [stamp]
    robot = NS(uid="robot", get_joint_ids=lambda part: (1, 2))
    sim = NS(
        physics_backend="default",
        num_envs=1,
        get_robot=lambda uid: robot,
        sim_config=NS(physics_cfg=NS(gravity=(0.0, 0.0, -9.81))),
    )
    binding = owner.PickClosingBinding(
        "program/segment:0",
        "payload",
        "support" if opening or downstream else None,
        AbsolutePoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)) if opening else None,
        "program/place:0" if downstream else None,
    )
    provider = owner.SimulationPickClosingProvider(
        robot,
        sim,
        rigid_uids={"payload": "native_payload", "support": "native_support"},
        bindings=(binding,),
        read_source_stamp=lambda: current[0],
    )
    hand = NS(
        require_target=lambda cls: JointPositionTarget("custom_hand", (1, 2)),
        joint_positions=lambda command, **kw: (
            torch.tensor([[0.04, 0.04]]) if command == "open" else torch.zeros(1, 2)
        ),
    )
    request = NS(
        skill_id="pick_up",
        invocation_id=binding.invocation_id,
        goal=NS(semantics=NS(entity_id="payload"), object_pose=None),
        skill_options=NS(native_closing_revision=revision),
        binding=NS(endpoint=lambda *a: hand),
    )
    pose = torch.eye(4)[None]
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.zeros(1, 3), timestamp=0.5),
        scene=NS(
            timestamp=0.5,
            version=3,
            entities={
                name: NS(pose=pose, confidence=1.0) for name in ("payload", "support")
            },
        ),
        world_source_stamp=stamp,
    )
    calls = []

    class Query:
        def __init__(self, kw):
            self.closed = False
            self.kw = kw

        def close(self):
            self.closed = True

    def capture(actual_robot, actual_sim, **kw):
        assert actual_robot is robot and actual_sim is sim
        calls.append(kw)
        return Query(kw)

    monkeypatch.setattr(owner, "build_native_closing_query", capture)
    return provider, request, context, current, calls, sim


@pytest.mark.parametrize(
    "revision,opening", [(1, False), (3, True), (8, False), (9, False)]
)
def test_native_prepare_uses_exact_commands_geometry_roles_and_source_scope(
    monkeypatch, revision, opening
):
    provider, request, context, current, calls, sim = _setup(
        monkeypatch, revision=revision, opening=opening
    )
    assert calls == []
    first = provider.prepare(request, context, hand_geometry_cfg=object())
    second = provider.prepare(request, context, hand_geometry_cfg=object())
    assert first.kw["scope"] != second.kw["scope"]
    assert first.kw["source_uid"] == "native_payload"
    assert first.kw["observation_qpos"] is context.robot.qpos
    assert torch.equal(first.kw["object_pose"], context.scene.entities["payload"].pose)
    assert torch.equal(first.kw["open_qpos"], torch.tensor([[0.04, 0.04]]))
    assert not first.kw["grasp_qpos"].any()
    if opening:
        target = first.kw["opening"]
        assert (
            target.scope == first.kw["scope"] and target.support_uid == "native_support"
        )
        np.testing.assert_allclose(target.object_target_pose[:3, 3], [0.1, 0.2, 0.3])
    if revision == 9:
        np.testing.assert_array_equal(first.kw["gravity_in_object"], [0, 0, -9.81])
    else:
        assert first.kw["gravity_in_object"] is None
    scene = object()
    monkeypatch.setattr(
        owner, "build_native_rigid_rest_scene", lambda actual, query: scene
    )
    assert provider.prepare_rigid_scene(first) is scene
    current[0] = replace(current[0], source_id="different")
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare_rigid_scene(second)
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare(request, context, hand_geometry_cfg=object())
    assert len(calls) == 2


def test_query_closes_when_capture_or_rigid_scene_changes_world(monkeypatch):
    provider, request, context, current, calls, _ = _setup(monkeypatch)
    original = owner.build_native_closing_query
    captured = []

    def drift(*args, **kw):
        query = original(*args, **kw)
        captured.append(query)
        current[0] = replace(current[0], revision=8)
        return query

    monkeypatch.setattr(owner, "build_native_closing_query", drift)
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare(request, context, hand_geometry_cfg=object())
    assert captured[0].closed
    monkeypatch.setattr(owner, "build_native_closing_query", original)
    current[0] = context.world_source_stamp
    query = provider.prepare(request, context, hand_geometry_cfg=object())

    def scene_drift(*args):
        current[0] = replace(current[0], simulation_time=2.0)
        return object()

    monkeypatch.setattr(owner, "build_native_rigid_rest_scene", scene_drift)
    with pytest.raises(ValueError, match="source changed"):
        provider.prepare_rigid_scene(query)
    assert query.closed


@pytest.mark.parametrize("gravity", [None, (0.0, 0.0, 0.0), (float("nan"), 0.0, -9.81)])
def test_pad_quality_requires_configured_gravity_before_native_capture(
    monkeypatch, gravity
):
    provider, request, context, current, calls, sim = _setup(monkeypatch, revision=9)
    sim.sim_config.physics_cfg.gravity = gravity
    with pytest.raises(ValueError, match="finite nonzero"):
        provider.prepare(request, context, hand_geometry_cfg=object())
    assert calls == []


def test_explicit_object_pose_keeps_atomic_pose_normalization(monkeypatch):
    provider, request, context, current, calls, _ = _setup(monkeypatch)
    request.goal.object_pose = torch.eye(4, dtype=torch.float64)[None]
    request.goal.object_pose[:, 0, 3] = 0.123456789
    query = provider.prepare(request, context, hand_geometry_cfg=object())
    assert query.kw["object_pose"].dtype == torch.float32
    torch.testing.assert_close(
        query.kw["object_pose"], request.goal.object_pose.float(), rtol=0, atol=0
    )


def test_opening_uses_exact_grounded_final_target_without_planning(monkeypatch):
    provider, request, context, _, calls, _ = _setup(
        monkeypatch, revision=8, downstream=True
    )
    target = torch.eye(4)
    target[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target[:3, 3] = torch.tensor([0.21, 0.32, 0.43])
    request.skill_options.adaptive_grasp = "ordinary"
    request.skill_options.downstream_object_target_poses = (torch.eye(4), target)
    query = provider.prepare(request, context, hand_geometry_cfg=object())
    assert len(calls) == 1
    np.testing.assert_array_equal(
        query.kw["opening"].object_target_pose, target.double().numpy()
    )
    target[:3, 3] += 1
    assert query.kw["opening"].object_target_pose[0, 3] == pytest.approx(0.21)
    request.skill_options.downstream_object_target_poses = ()
    with pytest.raises(ValueError, match="grounded final Place"):
        provider.prepare(request, context, hand_geometry_cfg=object())
    assert len(calls) == 1


def test_opening_modes_are_exclusive_and_absence_preserves_identity():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value

    plain = owner.PickClosingBinding("pick", "payload")
    assert "opening_place_invocation_id" not in _canonical_value(plain)["fields"]
    with pytest.raises(ValueError, match="exactly one"):
        owner.PickClosingBinding(
            "pick",
            "payload",
            "support",
            AbsolutePoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            "place",
        )
    with pytest.raises(ValueError, match="differ"):
        owner.PickClosingBinding(
            "pick", "payload", "support", opening_place_invocation_id="pick"
        )
