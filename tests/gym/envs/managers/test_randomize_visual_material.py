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

from embodichain.lab.sim.material import VisualMaterialInst


class FakeMaterialInstance:
    def __init__(self):
        self.base_color_map = None

    def set_base_color_map(self, texture):
        self.base_color_map = texture


class FakeMaterial:
    def __init__(self):
        self.instances = {}

    def get_inst(self, uid):
        return self.instances.setdefault(uid, FakeMaterialInstance())


def test_set_base_color_texture_uses_texture_ref():
    material = FakeMaterial()
    instance = VisualMaterialInst("instance", material)
    texture_ref = object()

    instance.set_base_color_texture(texture_ref=texture_ref)

    assert material.get_inst("instance").base_color_map is texture_ref
    assert instance.base_color_texture is texture_ref
