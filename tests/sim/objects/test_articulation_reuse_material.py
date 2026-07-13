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
from embodichain.lab.sim.objects.articulation import Articulation


class _MockArticulation(Articulation):
    def __init__(self, entities, uid, link_names):
        # Skip the heavy Articulation.__init__; set only what the new methods use.
        # num_instances is a read-only property on BatchEntity, driven by _entities.
        self._entities = entities
        self._all_indices = list(range(len(entities)))
        self.uid = uid
        self.link_names = link_names


def _make_entity(links, segs_per_link=1):
    entity = MagicMock(name="ArtEntity")
    rbs = {}
    for link in links:
        rb = MagicMock(name=f"rb_{link}")
        rb.get_mesh_count.return_value = segs_per_link
        orig = MagicMock(name=f"orig_{link}")
        tmpl = MagicMock(name=f"tmpl_{link}")
        orig.get_template.return_value = tmpl
        rb.get_material.return_value = orig
        rbs[link] = rb
    entity.get_render_body.side_effect = lambda name: rbs.get(name)
    return entity, rbs


def test_get_existing_visual_material_per_link():
    links = ["base", "gripper"]
    entity, rbs = _make_entity(links, segs_per_link=1)
    obj = _MockArticulation([entity], "art", links)

    states = obj.get_existing_visual_material(link_names=links)

    assert len(states) == 1
    assert set(states[0].keys()) == set(links)
    for link in links:
        assert len(states[0][link]) == 1
        seg = states[0][link][0]
        assert isinstance(seg, ReuseSegmentState)
        assert isinstance(seg.working_inst, VisualMaterialInst)


def test_get_existing_visual_material_raises_when_no_material():
    links = ["base"]
    entity, rbs = _make_entity(links)
    rbs["base"].get_material.return_value = None
    obj = _MockArticulation([entity], "art", links)

    with pytest.raises(ValueError, match="no material"):
        obj.get_existing_visual_material(link_names=links)


def test_apply_render_material_inst_swaps_on_link_render_body():
    links = ["base"]
    entity, rbs = _make_entity(links)
    obj = _MockArticulation([entity], "art", links)
    inst = MagicMock(name="MaterialInst")

    obj.apply_render_material_inst(0, inst, link_name="base", mesh_id=2)

    rbs["base"].set_material.assert_called_once_with(2, inst)
