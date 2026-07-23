# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import torch

from embodichain.lab.sim.sensors.camera import (
    CameraCfg,
    _MIXED_RESOLUTION_GROUP,
    _SHARED_GROUPS,
    _resize_camera_output,
    plan_camera_groups,
)


class _FakeEnv:
    def get_all_arenas(self):
        return [object(), object()]


class _FakeWorld:
    def __init__(self) -> None:
        self.calls = []

    def get_env(self):
        return _FakeEnv()

    def create_camera_group(self, resolution, layer_count, offscreen):
        frame_buffer = object()
        self.calls.append((resolution, layer_count, offscreen, frame_buffer))
        return frame_buffer


def test_plan_mixed_resolution_group_uses_largest_extent(monkeypatch) -> None:
    world = _FakeWorld()
    import embodichain.lab.sim.sensors.camera as camera_module

    monkeypatch.setattr(camera_module.dexsim, "default_world", lambda: world)
    configs = [
        CameraCfg(uid="head", width=960, height=540),
        CameraCfg(uid="right_wrist", width=640, height=480),
        CameraCfg(uid="left_wrist", width=640, height=480),
    ]

    plan_camera_groups(configs, enabled=True, merge_different_resolutions=True)

    assert len(world.calls) == 1
    assert world.calls[0][0] == [960, 540]
    assert world.calls[0][1] == 6
    slot = _SHARED_GROUPS[_MIXED_RESOLUTION_GROUP]
    assert slot["resolution"] == (960, 540)
    assert slot["capacity"] == 6
    plan_camera_groups([], enabled=False)


def test_resize_camera_output_preserves_shape_and_dtype() -> None:
    color = torch.full((2, 540, 960, 4), 137, dtype=torch.uint8)
    mask = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)

    resized_color = _resize_camera_output(color, 480, 640, "bilinear")
    resized_mask = _resize_camera_output(mask, 4, 4, "nearest")

    assert resized_color.shape == (2, 480, 640, 4)
    assert resized_color.dtype == torch.uint8
    assert torch.equal(resized_color, torch.full_like(resized_color, 137))
    assert resized_mask.shape == (1, 4, 4)
    assert resized_mask.dtype == torch.int32
    assert set(resized_mask.unique().tolist()) == {1, 2, 3, 4}
