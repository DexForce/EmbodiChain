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

"""Configuration ownership and frame composition without simulation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.scene.composition import (
    SceneFragment,
    compose,
    declare_fragment,
)
from embodichain.lab.sim.scene.layout import LayoutCfg, LayoutProblem
from embodichain.lab.sim.scene.procedural import SceneRecipe, generate, check_layout
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant

pytestmark = pytest.mark.no_sim


def _cfg(uid, pos=(0.0, 0.0, 0.0)):
    return RigidObjectCfg(uid=uid, shape=CubeCfg(size=[0.1, 0.1, 0.1]), init_pos=pos)


def test_fragment_owns_configs_and_preserves_scale_under_anchor_composition():
    cfg = _cfg("item")
    cfg.body_scale = (2.0, 3.0, 4.0)
    cfg.init_local_pose = np.eye(4)
    cfg.init_local_pose[:3, 3] = [1.0, 0.0, 0.0]
    anchor = np.eye(4)
    anchor[0, 3] = 0.5
    fragment = SceneFragment([cfg], anchors={"mount": anchor})
    cfg.uid = "changed"
    cfg.init_local_pose[0, 3] = 9
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
    pose[:3, 3] = [0.2, 0.3, 0.4]
    composed = compose(
        [fragment], namespace="cell", anchor="mount", arena_from_fragment=pose
    )
    result = composed.entities[0]
    np.testing.assert_allclose(
        result.init_local_pose,
        pose @ np.linalg.inv(anchor) @ fragment.entities[0].init_local_pose,
    )
    assert result.body_scale == (2.0, 3.0, 4.0)
    assert result.uid == "cell__item"
    np.testing.assert_allclose(composed.anchors["cell__mount"], pose)
    result.uid = "corrupt"
    assert composed.entities[0].uid == "cell__item"


def test_fragment_duplicate_uid_fails_before_declaration():
    calls = []
    sim = SimpleNamespace(
        **{
            f"get_{kind}_uid_list": lambda: ["duplicate"]
            for kind in (
                "rigid_object",
                "articulation",
                "robot",
                "rigid_object_group",
                "deformable_object",
            )
        },
        add_rigid_object=calls.append,
    )
    with pytest.raises(ValueError, match="already exist"):
        declare_fragment(sim, SceneFragment([_cfg("new"), _cfg("duplicate")]))
    assert calls == []
    with pytest.raises(ValueError, match="Duplicate"):
        compose([SceneFragment([_cfg("same")]), SceneFragment([_cfg("same")])])


def test_composition_matches_spawn_xyz_order_for_noncommuting_euler_angles():
    from embodichain.lab.sim.spawn.descriptors import _pose_from_cfg

    cfg = _cfg("rotated", (0.1, 0.2, 0.3))
    cfg.init_rot = (20.0, 35.0, 47.0)
    expected = _pose_from_cfg(cfg)
    result = compose([SceneFragment([cfg])]).entities[0]
    np.testing.assert_allclose(result.init_local_pose, expected, atol=1e-7)
    assert result.init_rot == pytest.approx(cfg.init_rot)


def test_local_pose_overrides_conflicting_euler_and_position_fields():
    cfg = _cfg("posed", (9.0, 8.0, 7.0))
    cfg.init_rot = (45.0, 10.0, 80.0)
    cfg.init_local_pose = np.eye(4)
    cfg.init_local_pose[:3, 3] = [0.1, 0.2, 0.3]
    result = compose([SceneFragment([cfg])]).entities[0]
    np.testing.assert_array_equal(result.init_local_pose, cfg.init_local_pose)


def test_namespace_and_anchor_validation_preserves_source_fragment():
    source = SceneFragment([_cfg("item")])
    with pytest.raises(ValueError, match="slash"):
        compose([source], namespace="bad/path")
    with pytest.raises(ValueError, match="Unknown"):
        compose([source], anchor="missing")
    assert source.entities[0].uid == "item"


def test_nested_config_copies_are_detached_on_entry_and_readback():
    cfg = _cfg("item")
    fragment = SceneFragment([cfg])
    cfg.shape.size[0] = 3.0
    readback = fragment.entities[0]
    readback.shape.size[1] = 4.0
    assert list(fragment.entities[0].shape.size) == [0.1, 0.1, 0.1]


def test_declaration_missing_later_kind_capability_does_not_declare_earlier_cfg():
    from embodichain.lab.sim.cfg import ArticulationCfg

    calls = []
    sim = SimpleNamespace(
        **{
            f"get_{kind}_uid_list": lambda: []
            for kind in (
                "rigid_object",
                "articulation",
                "robot",
                "rigid_object_group",
                "deformable_object",
            )
        },
        add_rigid_object=calls.append,
    )
    fragment = SceneFragment(
        [_cfg("item"), ArticulationCfg(uid="drawer", fpath="drawer.urdf")]
    )
    with pytest.raises(ValueError, match="add_articulation"):
        declare_fragment(sim, fragment)
    assert calls == []
