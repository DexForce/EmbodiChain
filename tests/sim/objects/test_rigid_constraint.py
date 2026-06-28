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

from dataclasses import MISSING

from embodichain.lab.sim.cfg import RigidConstraintCfg


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
