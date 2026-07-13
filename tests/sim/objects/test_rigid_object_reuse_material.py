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

from unittest.mock import MagicMock

import pytest

from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst
from embodichain.lab.sim.objects.rigid_object import RigidObject


class _MockRigidObject(RigidObject):
    def __init__(self, entities, uid):
        # Skip the heavy RigidObject.__init__; set only what the new methods use.
        # num_instances is a read-only property on BatchEntity, driven by _entities.
        self._entities = entities
        self._all_indices = list(range(len(entities)))
        self.uid = uid


def _make_entity(num_segments=1):
    entity = MagicMock(name="MeshObject")
    render_body = MagicMock(name="RenderBody")
    render_body.get_mesh_count.return_value = num_segments
    seg_mats = []
    for i in range(num_segments):
        orig = MagicMock(name=f"orig_inst_{i}")
        tmpl = MagicMock(name=f"template_{i}")
        orig.get_template.return_value = tmpl
        seg_mats.append((orig, tmpl))
    # Return the matching material for each render-body segment index.
    render_body.get_material.side_effect = [orig for orig, _ in seg_mats]
    entity.get_render_body.return_value = render_body
    return entity, render_body, seg_mats


def test_get_existing_visual_material_builds_state_per_segment():
    entity, render_body, seg_mats = _make_entity(num_segments=2)
    obj = _MockRigidObject([entity], "obj")

    states = obj.get_existing_visual_material()

    assert len(states) == 1  # one env
    assert len(states[0]) == 2  # two segments
    for seg, (orig, tmpl) in zip(states[0], seg_mats):
        assert isinstance(seg, ReuseSegmentState)
        assert seg.original_inst is orig
        assert isinstance(seg.working_inst, VisualMaterialInst)
        tmpl.create_inst.assert_called_once()  # working instance created from template


def test_get_existing_visual_material_shared_returns_single_env():
    entity, render_body, seg_mats = _make_entity(num_segments=1)
    obj = _MockRigidObject([entity, entity, entity], "obj")

    states = obj.get_existing_visual_material(shared=True)
    assert len(states) == 1  # shared -> first env only


def test_get_existing_visual_material_raises_when_no_material():
    entity, render_body, _ = _make_entity(num_segments=1)
    render_body.get_material.side_effect = None
    render_body.get_material.return_value = None
    obj = _MockRigidObject([entity], "obj")

    with pytest.raises(ValueError, match="no material"):
        obj.get_existing_visual_material()


def test_apply_render_material_inst_swaps_on_render_body():
    entity, render_body, _ = _make_entity(num_segments=1)
    obj = _MockRigidObject([entity], "obj")
    inst = MagicMock(name="MaterialInst")

    obj.apply_render_material_inst(0, inst, mesh_id=3)

    render_body.set_material.assert_called_once_with(3, inst)
