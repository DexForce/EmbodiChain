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

"""Tests for the RigidConstraint sim-layer wrapper and its config."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from dataclasses import MISSING

from embodichain.lab.sim.cfg import RigidConstraintCfg
from embodichain.lab.sim.objects.constraint import RigidConstraint


def test_rigid_constraint_cfg_defaults():
    """RigidConstraintCfg requires name + both object uids; frames default None."""
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    assert cfg.name == "weld"
    assert cfg.rigid_object_a_uid == "cube"
    assert cfg.rigid_object_b_uid == "block"
    assert cfg.local_frame_a is None
    assert cfg.local_frame_b is None
    assert cfg.constraint_type == "fixed"


def test_rigid_constraint_cfg_required_fields_are_missing():
    """Required fields default to the MISSING sentinel."""
    assert RigidConstraintCfg.__dataclass_fields__["name"].default is MISSING
    assert (
        RigidConstraintCfg.__dataclass_fields__["rigid_object_a_uid"].default is MISSING
    )
    assert (
        RigidConstraintCfg.__dataclass_fields__["rigid_object_b_uid"].default is MISSING
    )


def test_rigid_constraint_cfg_accepts_frames():
    """Local frames accept 4x4 numpy arrays."""
    frame = np.eye(4, dtype=np.float32)
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
        local_frame_a=frame,
        local_frame_b=frame,
    )
    np.testing.assert_allclose(cfg.local_frame_a, frame)


def _make_handle(name="weld_0", rel_z=0.0, valid=True):
    """Build a mock dexsim constraint handle."""
    h = MagicMock()
    h.get_name.return_value = name
    h.is_valid.return_value = valid
    rel = np.eye(4, dtype=np.float32)
    rel[2, 3] = rel_z
    h.get_relative_transform.return_value = rel
    h.get_local_pose.return_value = np.eye(4, dtype=np.float32)
    return h


def _make_rigid_object(uid="cube", num_envs=4):
    """Build a mock RigidObject with a per-arena entity list."""
    obj = MagicMock()
    obj.uid = uid
    obj.num_instances = num_envs
    obj._entities = [MagicMock() for _ in range(num_envs)]
    return obj


def test_rigid_constraint_num_envs_and_init():
    """RigidConstraint exposes num_envs and stores handles + object refs."""
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    handles = [_make_handle("weld_0"), None, _make_handle("weld_2"), None]
    obj_a = _make_rigid_object("cube", 4)
    obj_b = _make_rigid_object("block", 4)

    constraint = RigidConstraint(
        cfg=cfg,
        constraint_handles=handles,
        rigid_object_a=obj_a,
        rigid_object_b=obj_b,
        device=torch.device("cpu"),
    )
    assert constraint.num_envs == 4
    assert constraint.rigid_object_a is obj_a
    assert constraint.rigid_object_b is obj_b
    assert len(constraint.constraint_handles) == 4


def test_rigid_constraint_get_name_single_and_multi_env():
    """Single env keeps the base name; multi env appends the arena index."""
    cfg_single = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    c_single = RigidConstraint(
        cfg_single,
        [_make_handle("weld")],
        MagicMock(),
        MagicMock(),
        torch.device("cpu"),
    )
    assert c_single.get_name(0) == "weld"

    cfg_multi = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    handles = [_make_handle("weld_0"), _make_handle("weld_1")]
    c_multi = RigidConstraint(
        cfg_multi, handles, MagicMock(), MagicMock(), torch.device("cpu")
    )
    assert c_multi.get_name(0) == "weld_0"
    assert c_multi.get_name(1) == "weld_1"


def test_rigid_constraint_get_relative_transform_skips_none():
    """get_relative_transform skips None handles and only returns for active envs."""
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b"
    )
    handles = [
        _make_handle("weld_0", rel_z=0.1),
        None,
        _make_handle("weld_2", rel_z=0.2),
        None,
    ]
    constraint = RigidConstraint(
        cfg, handles, MagicMock(), MagicMock(), torch.device("cpu")
    )

    # default: all env_ids, skips None
    transforms = constraint.get_relative_transform()
    assert len(transforms) == 2
    assert transforms[0][2, 3] == pytest.approx(0.1)
    assert transforms[1][2, 3] == pytest.approx(0.2)

    # explicit subset including a None handle is skipped
    transforms_subset = constraint.get_relative_transform(env_ids=[1, 2])
    assert len(transforms_subset) == 1
    assert transforms_subset[0][2, 3] == pytest.approx(0.2)


def test_rigid_constraint_is_valid():
    """is_valid reports per-env validity, skipping None handles."""
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b"
    )
    handles = [_make_handle(valid=True), None, _make_handle(valid=False), None]
    constraint = RigidConstraint(
        cfg, handles, MagicMock(), MagicMock(), torch.device("cpu")
    )
    assert constraint.is_valid() == [True, False]


def test_rigid_constraint_destroy_calls_arena_remove_per_env():
    """destroy calls arena.remove_constraint for each active handle in env_ids."""
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b"
    )
    handles = [
        _make_handle("weld_0"),
        None,
        _make_handle("weld_2"),
        _make_handle("weld_3"),
    ]
    constraint = RigidConstraint(
        cfg, handles, MagicMock(), MagicMock(), torch.device("cpu")
    )

    arenas = [MagicMock() for _ in range(4)]
    arena_resolver = lambda i: arenas[i]

    constraint.destroy(env_ids=[0, 2], arena_resolver=arena_resolver)
    arenas[0].remove_constraint.assert_called_once_with("weld_0")
    arenas[2].remove_constraint.assert_called_once_with("weld_2")
    arenas[1].remove_constraint.assert_not_called()
    arenas[3].remove_constraint.assert_not_called()
    # cleared handles become None
    assert constraint.constraint_handles[0] is None
    assert constraint.constraint_handles[2] is None
    assert constraint.constraint_handles[3] is not None  # not in env_ids


def test_rigid_constraint_destroy_all_returns_all_cleared():
    """destroy with env_ids=None clears every active handle."""
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b"
    )
    handles = [_make_handle("weld_0"), None, _make_handle("weld_2"), None]
    constraint = RigidConstraint(
        cfg, handles, MagicMock(), MagicMock(), torch.device("cpu")
    )
    arenas = [MagicMock() for _ in range(4)]
    constraint.destroy(env_ids=None, arena_resolver=lambda i: arenas[i])
    assert all(h is None for h in constraint.constraint_handles)
