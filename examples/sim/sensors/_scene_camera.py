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

"""Capture native RGB for the sensor examples from the same simulation step."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import LightCfg, RenderCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import Camera, CameraCfg

SCENE_RENDER = RenderCfg(renderer="hybrid", spp=4)
Size3 = tuple[float, float, float]


def scene_camera(
    sim: SimulationManager,
    eye: Size3,
    target: Size3,
    colors: Mapping[str, Size3] | None = None,
) -> Camera:
    """Prepare a headless camera with consistent lighting and labeled bodies."""
    sim.prepare()
    for uid, rgb in (colors or {}).items():
        obj = sim.get_rigid_object(uid) or sim.get_visual_object(uid)
        obj.set_visual_material(
            sim.create_visual_material(
                VisualMaterialCfg(uid=f"{uid}_preview_mat", base_color=[*rgb, 1.0])
            )
        )
    sim.add_light(
        LightCfg(
            uid="preview_key",
            light_type="direction",
            direction=(-0.4, 0.7, -1.0),
            intensity=4.0,
        )
    )
    return sim.add_sensor(
        CameraCfg(
            uid="preview_camera",
            width=720,
            height=540,
            intrinsics=(760, 760, 360, 270),
            extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
        )
    )


def capture_rgb(sim: SimulationManager, camera: Camera) -> np.ndarray:
    """Render the current physics state and copy the owned RGB frame."""
    sim.render_camera_group([camera.group_id])
    camera.update(fetch_only=True)
    color = camera.get_data()["color"][0, ..., :3].cpu().numpy()
    return color.copy()


def show_scene(axis: Axes, rgb: np.ndarray) -> None:
    """Display the camera frame without schematic axes."""
    axis.imshow(rgb)
    axis.axis("off")
    axis.set_title("DexSim camera")
