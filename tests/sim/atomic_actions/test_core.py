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

"""Tests for atomic_actions.core (typed targets, WorldState, ActionResult, ObjectSemantics)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    ActionResult,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)


class TestTypedTargets:
    def test_pose_target_holds_tensor(self):
        x = torch.eye(4)
        assert PoseTarget(xpos=x).xpos is x

    def test_pose_target_is_frozen(self):
        t = PoseTarget(xpos=torch.eye(4))
        with pytest.raises(Exception):
            t.xpos = torch.zeros(4, 4)  # type: ignore[misc]

    def test_grasp_target_holds_semantics(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="mug")
        assert GraspTarget(semantics=sem).semantics is sem

    def test_held_object_target_holds_pose(self):
        x = torch.eye(4)
        assert HeldObjectTarget(object_target_pose=x).object_target_pose is x


class TestObjectSemantics:
    def test_does_not_mutate_affordance_geometry(self):
        # The redesign removes the __post_init__ aliasing footgun.
        aff = Affordance()
        geometry = {"bounding_box": [0.1, 0.1, 0.1]}
        ObjectSemantics(affordance=aff, geometry=geometry, label="mug")
        # affordance should not have a geometry attribute, or if it does it should
        # NOT be the same object as the semantics' geometry dict.
        assert getattr(aff, "geometry", None) is not geometry

    def test_sets_object_label_on_affordance(self):
        aff = Affordance()
        ObjectSemantics(affordance=aff, geometry={}, label="mug")
        assert aff.object_label == "mug"

    def test_default_optional_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        assert sem.label == "none"
        assert sem.properties == {}
        assert sem.entity is None


class TestHeldObjectState:
    def test_required_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        s = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        assert s.semantics is sem
        assert s.object_to_eef.shape == (1, 4, 4)
        assert s.grasp_xpos.shape == (1, 4, 4)


class TestWorldState:
    def test_constructs_with_last_qpos_only(self):
        qpos = torch.zeros(2, 6)
        ws = WorldState(last_qpos=qpos)
        assert ws.last_qpos is qpos
        assert ws.held_object is None

    def test_carries_held_object(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        ws = WorldState(last_qpos=torch.zeros(1, 6), held_object=held)
        assert ws.held_object is held


class TestActionResult:
    def test_shape_contract(self):
        traj = torch.zeros(2, 10, 8)
        ws = WorldState(last_qpos=torch.zeros(2, 8))
        res = ActionResult(success=True, trajectory=traj, next_state=ws)
        assert res.success is True
        assert res.trajectory.shape == (2, 10, 8)
        assert res.next_state is ws


class TestActionCfg:
    def test_defaults(self):
        cfg = ActionCfg()
        assert cfg.name == "default"
        assert cfg.control_part == "arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None
