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


def test_texture_object_is_bound_directly_and_takes_priority_over_data():
    material = MagicMock(name="Material")
    instance = VisualMaterialInst("instance", material)
    texture = MagicMock(name="Texture")
    texture_data = MagicMock(name="texture_data")

    instance.set_base_color_texture(
        texture_data=texture_data,
        texture_obj=texture,
    )

    material.get_inst.return_value.set_base_color_map.assert_called_once_with(texture)
    texture_data.cpu.assert_not_called()
    assert instance.base_color_texture is texture
