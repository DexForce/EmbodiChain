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

from embodichain.lab.sim.material import VisualMaterialInst


def _make_inst():
    mat = MagicMock(name="Material")
    inst = VisualMaterialInst("uid_test", mat)
    return inst, mat


def test_texture_obj_sets_map_without_upload():
    inst, mat = MagicMock(), MagicMock()
    obj = VisualMaterialInst.__new__(VisualMaterialInst)
    obj.uid = "u"
    obj._mat = mat
    obj.base_color_texture = None

    texture = MagicMock(name="Texture")
    obj.set_base_color_texture(texture_obj=texture)

    dexsim_inst = mat.get_inst.return_value
    dexsim_inst.set_base_color_map.assert_called_once_with(texture)
    assert obj.base_color_texture is texture


def test_texture_obj_takes_priority_over_data():
    obj = VisualMaterialInst.__new__(VisualMaterialInst)
    obj.uid = "u"
    mat = MagicMock()
    obj._mat = mat
    obj.base_color_texture = None

    obj.set_base_color_texture(texture_data=MagicMock(), texture_obj=MagicMock())

    # texture_obj branch used, create_color_texture never called
    mat.get_inst.assert_called_with("u")
