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

"""Tests for the rigid-constraint event functors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    create_rigid_constraint,
    remove_rigid_constraint,
)
from embodichain.lab.sim.objects.rigid_object import RigidObject


class MockRigidObjectForFunctor:
    """Stand-in for a RigidObject passing the isinstance check.

    The functor checks ``isinstance(asset, RigidObject)``. To avoid building a
    real RigidObject (needs a dexsim World), we monkeypatch the check.
    """


def _make_env(obj_a_is_rigid=True, obj_b_is_rigid=True):
    """Build a mock env with a spied sim.create/remove_rigid_constraint.

    obj_a / obj_b are instances of MockRigidObjectForFunctor so that, when the
    test patches ``events.RigidObject`` to that class, ``isinstance`` passes.
    When ``obj_a_is_rigid`` is False, obj_a becomes a plain object (isinstance
    fails) to exercise the rejection path.
    """
    env = MagicMock()
    env.device = torch.device("cpu")
    env.num_envs = 4

    obj_a = MockRigidObjectForFunctor() if obj_a_is_rigid else object()
    obj_b = MockRigidObjectForFunctor() if obj_b_is_rigid else object()
    env.sim.get_asset.side_effect = lambda uid: {"cube": obj_a, "block": obj_b}[uid]

    env.sim.create_rigid_constraint = MagicMock(return_value=MagicMock())
    env.sim.remove_rigid_constraint = MagicMock(return_value=True)
    return env, obj_a, obj_b


def test_create_functor_delegates_to_sim(monkeypatch):
    """create functor resolves both objects and forwards to sim.create_rigid_constraint."""
    env, obj_a, obj_b = _make_env()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject",
        MockRigidObjectForFunctor,
    )

    env_ids = torch.tensor([0, 2])
    create_rigid_constraint(
        env,
        env_ids,
        obj_a_cfg=SceneEntityCfg(uid="cube"),
        obj_b_cfg=SceneEntityCfg(uid="block"),
        name="weld",
    )

    env.sim.create_rigid_constraint.assert_called_once()
    call_kwargs = env.sim.create_rigid_constraint.call_args
    assert call_kwargs.kwargs["env_ids"] is env_ids
    cfg = call_kwargs.kwargs["cfg"]
    assert cfg.name == "weld"
    assert cfg.rigid_object_a_uid == "cube"
    assert cfg.rigid_object_b_uid == "block"
    assert cfg.local_frame_a is None
    assert cfg.local_frame_b is None


def test_create_functor_forwards_frames(monkeypatch):
    """create functor forwards local frames into the cfg."""
    env, _, _ = _make_env()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject",
        MockRigidObjectForFunctor,
    )
    frame = np.eye(4, dtype=np.float32)
    create_rigid_constraint(
        env,
        None,
        obj_a_cfg=SceneEntityCfg(uid="cube"),
        obj_b_cfg=SceneEntityCfg(uid="block"),
        name="weld",
        local_frame_a=frame,
        local_frame_b=frame,
    )
    cfg = env.sim.create_rigid_constraint.call_args.kwargs["cfg"]
    np.testing.assert_allclose(cfg.local_frame_a, frame)


def test_create_functor_rejects_non_rigid_object(monkeypatch):
    """A non-RigidObject asset raises (log_error raises RuntimeError)."""
    # obj_a is a plain object -> isinstance fails after the patch.
    env, obj_a, obj_b = _make_env(obj_a_is_rigid=False, obj_b_is_rigid=True)
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject",
        MockRigidObjectForFunctor,
    )
    with pytest.raises(RuntimeError):
        create_rigid_constraint(
            env,
            None,
            obj_a_cfg=SceneEntityCfg(uid="cube"),
            obj_b_cfg=SceneEntityCfg(uid="block"),
            name="weld",
        )
    env.sim.create_rigid_constraint.assert_not_called()


def test_remove_functor_delegates():
    """remove functor forwards name + env_ids to sim.remove_rigid_constraint."""
    env, _, _ = _make_env()
    env_ids = torch.tensor([1, 3])
    remove_rigid_constraint(env, env_ids, name="weld")
    env.sim.remove_rigid_constraint.assert_called_once_with("weld", env_ids=env_ids)


def test_remove_functor_none_env_ids():
    """remove functor forwards env_ids=None correctly."""
    env, _, _ = _make_env()
    remove_rigid_constraint(env, None, name="weld")
    env.sim.remove_rigid_constraint.assert_called_once_with("weld", env_ids=None)


from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.utils import configclass


@configclass
class _AttachEventsCfg:
    attach: EventCfg = EventCfg(
        func=create_rigid_constraint,
        mode="attach",
        params={
            "obj_a_cfg": SceneEntityCfg(uid="cube"),
            "obj_b_cfg": SceneEntityCfg(uid="block"),
            "name": "weld",
        },
    )


def test_custom_mode_apply_invokes_functor_with_env_ids(monkeypatch):
    """EventManager.apply(mode="attach", env_ids) calls the functor once with those env_ids."""
    # Build a minimal env stand-in that EventManager needs: num_envs, device, sim.
    env = MagicMock()
    env.num_envs = 4
    env.device = torch.device("cpu")
    env.sim = MagicMock()
    # SceneEntityCfg.resolve() checks scene.asset_uids for each uid; list them.
    env.sim.asset_uids = ["cube", "block"]
    # get_asset returns MockRigidObjectForFunctor instances so isinstance passes.
    obj_a = MockRigidObjectForFunctor()
    obj_b = MockRigidObjectForFunctor()
    env.sim.get_asset.side_effect = lambda uid: {"cube": obj_a, "block": obj_b}[uid]
    env.sim.create_rigid_constraint = MagicMock()

    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject",
        MockRigidObjectForFunctor,
    )

    manager = EventManager(cfg=_AttachEventsCfg(), env=env)

    env_ids = torch.tensor([0, 1])
    manager.apply(mode="attach", env_ids=env_ids)

    env.sim.create_rigid_constraint.assert_called_once()
    assert env.sim.create_rigid_constraint.call_args.kwargs["env_ids"] is env_ids
