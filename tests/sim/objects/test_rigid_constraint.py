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


from embodichain.lab.sim.sim_manager import SimulationManager


class MockArena:
    """Mock dexsim arena that records created constraints."""

    def __init__(self, fail_indices=None):
        self.created = []  # list of (name, actor0, actor1, frame_a, frame_b)
        self.removed = []  # list of names
        self.fail_indices = set(fail_indices or [])

    def create_fixed_constraint(self, name, actor0, actor1, local_frame0, local_frame1):
        self.created.append((name, actor0, actor1, local_frame0, local_frame1))
        if len(self.created) - 1 in self.fail_indices:
            return None
        h = MagicMock()
        h.get_name.return_value = name
        h.is_valid.return_value = True
        h.get_relative_transform.return_value = np.eye(4, dtype=np.float32)
        return h

    def remove_constraint(self, name):
        self.removed.append(name)


class _RigidConstraintTestSim:
    """A SimulationManager stand-in exposing only the constraint registry path.

    We avoid constructing a real dexsim World (which needs a GPU/window). Instead
    we drive create_rigid_constraint by giving it a fake `self` with the
    attributes the method touches: _rigid_objects, _arenas/_env, num_envs, device.
    """

    def __init__(self, num_envs=4, arenas=None):
        self._rigid_objects = {}
        self._constraints = {}
        self.device = torch.device("cpu")
        if num_envs == 1:
            self._arenas = []
            self._env = arenas[0] if arenas else MockArena()
        else:
            self._arenas = arenas or [MockArena() for _ in range(num_envs)]
            self._env = None

    @property
    def num_envs(self):
        return len(self._arenas) if self._arenas else 1

    def get_env(self, arena_index=-1):
        if arena_index >= 0 and self._arenas:
            return self._arenas[arena_index]
        return self._env

    # bind the real method under test
    create_rigid_constraint = SimulationManager.create_rigid_constraint
    remove_rigid_constraint = SimulationManager.remove_rigid_constraint
    get_rigid_constraint = SimulationManager.get_rigid_constraint
    get_rigid_constraint_uid_list = SimulationManager.get_rigid_constraint_uid_list
    _broadcast_frame = staticmethod(SimulationManager._broadcast_frame)


def _register_object(sim, uid, num_envs, z=0.0):
    """Register a mock RigidObject.

    ``get_local_pose`` returns a real ``(num_envs, 4, 4)`` tensor so the
    constraint's default ``local_frame_b`` computation (which reads both
    objects' current poses) works under the mock. ``z`` sets the per-env
    translation so two objects can be placed at different heights.
    """
    obj = MagicMock()
    obj.uid = uid
    obj.num_instances = num_envs
    obj._entities = [MagicMock(name=f"{uid}_{i}") for i in range(num_envs)]
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
    pose[:, 2, 3] = z
    obj.get_local_pose.return_value = pose
    sim._rigid_objects[uid] = obj
    return obj


def test_create_rigid_constraint_resolves_both_objects_all_envs():
    """create builds one handle per arena and stores a RigidConstraint."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg)

    assert cfg.name in sim._constraints
    assert constraint.num_envs == 4
    assert all(h is not None for h in constraint.constraint_handles)
    # each arena got exactly one create call with the right actors
    for i, arena in enumerate(sim._arenas):
        assert arena.created[0][0] == f"weld_{i}"
        assert arena.created[0][1] is sim._rigid_objects["cube"]._entities[i]
        assert arena.created[0][2] is sim._rigid_objects["block"]._entities[i]


def test_create_rigid_constraint_single_env_uses_global_env():
    """Single-env create routes through the global env and keeps the base name."""
    arena = MockArena()
    sim = _RigidConstraintTestSim(num_envs=1, arenas=[arena])
    _register_object(sim, "cube", 1)
    _register_object(sim, "block", 1)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg)
    assert constraint.constraint_handles[0] is not None
    assert arena.created[0][0] == "weld"  # base name, no suffix


def test_create_rigid_constraint_subset_env_ids():
    """env_ids subset populates only those arenas; others stay None."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg, env_ids=[0, 2])
    assert constraint.constraint_handles[0] is not None
    assert constraint.constraint_handles[1] is None
    assert constraint.constraint_handles[2] is not None
    assert constraint.constraint_handles[3] is None
    # only arenas 0 and 2 got a create call
    assert len(sim._arenas[0].created) == 1
    assert len(sim._arenas[1].created) == 0
    assert len(sim._arenas[2].created) == 1
    assert len(sim._arenas[3].created) == 0


def test_create_rigid_constraint_missing_object_raises():
    """A missing object uid raises (log_error raises RuntimeError by default)."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    # block not registered
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_create_rigid_constraint_duplicate_name_raises():
    """A duplicate base name raises."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_create_rigid_constraint_failed_handle_raises():
    """If dexsim returns None for a handle, log_error raises."""
    sim = _RigidConstraintTestSim(
        num_envs=2, arenas=[MockArena(fail_indices=[0]), MockArena()]
    )
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_broadcast_frame_none_to_identity():
    """None frame broadcasts to identity per env."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frames = sim._broadcast_frame(None, num_envs=3, env_ids=[0, 1, 2], name="weld")
    assert len(frames) == 3
    for f in frames:
        np.testing.assert_allclose(f, np.eye(4))


def test_broadcast_frame_4x4_repeats():
    """A single 4x4 matrix repeats across all envs."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frame = np.eye(4, dtype=np.float32) * 2
    frames = sim._broadcast_frame(frame, num_envs=3, env_ids=[0, 1, 2], name="weld")
    assert len(frames) == 3
    for f in frames:
        np.testing.assert_allclose(f, frame)


def test_broadcast_frame_N4x4_indexes():
    """An (N,4,4) array indexes per env and requires N == num_envs."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frames_in = np.stack([np.eye(4) * i for i in range(3)], axis=0).astype(np.float32)
    frames = sim._broadcast_frame(frames_in, num_envs=3, env_ids=[0, 1, 2], name="weld")
    for i, f in enumerate(frames):
        np.testing.assert_allclose(f, frames_in[i])

    # wrong N raises
    bad = np.stack([np.eye(4)] * 2, axis=0).astype(np.float32)
    with pytest.raises(RuntimeError):
        sim._broadcast_frame(bad, num_envs=3, env_ids=[0, 1, 2], name="weld")


def test_remove_rigid_constraint_all_envs():
    """remove with env_ids=None clears every arena and drops the registry entry."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)

    removed = sim.remove_rigid_constraint("weld")
    assert removed is True
    assert "weld" not in sim._constraints
    # each arena got remove_constraint with its per-env name
    for i, arena in enumerate(sim._arenas):
        assert f"weld_{i}" in arena.removed


def test_remove_rigid_constraint_subset_keeps_others():
    """remove with a subset env_ids clears only those arenas; registry kept."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)

    removed = sim.remove_rigid_constraint("weld", env_ids=[0, 2])
    assert removed is True
    # still in registry because envs 1,3 remain active
    assert "weld" in sim._constraints
    assert sim._constraints["weld"].constraint_handles[0] is None
    assert sim._constraints["weld"].constraint_handles[1] is not None
    assert sim._constraints["weld"].constraint_handles[2] is None
    assert sim._constraints["weld"].constraint_handles[3] is not None
    assert "weld_0" in sim._arenas[0].removed
    assert "weld_2" in sim._arenas[2].removed
    assert sim._arenas[1].removed == []


def test_remove_rigid_constraint_unknown_name_warns_false():
    """remove on an unknown name returns False without raising."""
    sim = _RigidConstraintTestSim(num_envs=4)
    removed = sim.remove_rigid_constraint("nope")
    assert removed is False


def test_get_rigid_constraint_and_uid_list():
    """get returns the constraint; uid list lists all registered names."""
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)
    assert sim.get_rigid_constraint("weld") is not None
    assert sim.get_rigid_constraint("nope") is None
    assert sim.get_rigid_constraint_uid_list() == ["weld"]


def test_partial_remove_then_all_drops_registry():
    """Subset remove then removing remaining envs drops the registry entry."""
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)
    sim.remove_rigid_constraint("weld", env_ids=[0])
    assert "weld" in sim._constraints
    sim.remove_rigid_constraint("weld", env_ids=[1])
    assert "weld" not in sim._constraints


def test_create_rigid_constraint_default_frame_b_preserves_relative_pose():
    """With local_frame_b=None, frame_b = inv(pose_B) @ pose_A (preserves offset).

    cube_a at z=1.4, cube_b at z=1.2 -> B is 0.2 below A. The computed
    local_frame_b must translate +0.2 in z (A's pose relative to B) so the
    constraint welds the cubes at their current relative pose instead of
    pulling their origins together.
    """
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube_a", 2, z=1.4)
    _register_object(sim, "cube_b", 2, z=1.2)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube_a", rigid_object_b_uid="cube_b"
    )
    sim.create_rigid_constraint(cfg)

    for i, arena in enumerate(sim._arenas):
        # arena.created[0] = (name, actor0, actor1, frame_a, frame_b)
        frame_a = arena.created[0][3]
        frame_b = arena.created[0][4]
        # frame_a defaults to identity
        np.testing.assert_allclose(frame_a, np.eye(4), atol=1e-6)
        # frame_b = inv(pose_B) @ pose_A = translate(0, 0, +0.2)
        np.testing.assert_allclose(frame_b[:3, 3], [0.0, 0.0, 0.2], atol=1e-5)


def test_create_rigid_constraint_explicit_frame_b_used_verbatim():
    """An explicit local_frame_b is broadcast verbatim (no pose computation)."""
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube_a", 2, z=1.4)
    _register_object(sim, "cube_b", 2, z=1.2)

    explicit = np.eye(4, dtype=np.float32) * 3.0
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube_a",
        rigid_object_b_uid="cube_b",
        local_frame_b=explicit,
    )
    sim.create_rigid_constraint(cfg)
    for arena in sim._arenas:
        frame_b = arena.created[0][4]
        np.testing.assert_allclose(frame_b, explicit)
