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
"""Render qualified Wuji inference using measured native states and FK only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image

__all__ = ["main"]


def main() -> None:
    """Export clean clips for fixed first-index episodes of every object size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.evaluation.resolve()
    provenance = json.loads((folder / "provenance.json").read_text())
    if not provenance["native_policy_qualified"]:
        raise ValueError(
            "Only independently qualified native Wuji policies may be displayed"
        )
    if not provenance.get("grasp_qualification_passed"):
        raise ValueError(
            "Wuji v2 requires independently qualified opposing-pad contact"
        )
    if provenance["task_identity"].get("task_version") != "wuji-palm-up-v2":
        raise ValueError("Rejected palm-sideways v1 policies cannot be exported")
    for name in [
        "independent_physical_metrics_audit.json",
        "independent_reward_audit.json",
        "raw_cache_audit.json",
        "independent_geometry_audit.json",
    ]:
        if not json.loads((folder / name).read_text())["passed"]:
            raise ValueError(f"Independent audit failed: {name}")
    path = folder / "measured_states.pt"
    packet = torch.load(path, map_location="cpu", weights_only=False)
    states = torch.stack(packet["physical_states"][:400]).numpy()
    initial = packet["initial_states"][0].numpy()
    context = packet["initial_contexts"][0].numpy()
    records = {
        r["row"]: r
        for r in json.loads((folder / "evaluation.json").read_text())["records"]
        if r["seed"] == packet["seeds"][0]
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    items = []
    for index, scale in enumerate([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]):
        row = int(
            np.flatnonzero(np.isclose(context[:, 45], scale, atol=1e-7, rtol=0))[0]
        )
        if records[row]["dropped"]:
            raise ValueError(
                "Fixed first-index display episode dropped; retain this fact rather than select another outcome"
            )
        root = ET.parse(args.assets / "scene.xml").getroot()
        asset = root.find("asset")
        ET.SubElement(
            asset,
            "texture",
            name="showcase_object_texture",
            type="2d",
            builtin="checker",
            width="256",
            height="256",
            rgb1=".93 .30 .10",
            rgb2="1 .72 .25",
        )
        ET.SubElement(
            asset,
            "material",
            name="showcase_object_material",
            texture="showcase_object_texture",
            texrepeat="3 2",
            texuniform="false",
            rgba="1 1 1 1",
            specular=".3",
            shininess=".3",
        )
        geom = root.find("worldbody/body[@name='object']/geom")
        geom.set("size", f"{.02*scale} {.016*scale}")
        geom.set("material", "showcase_object_material")
        geom.set("rgba", "1 1 1 1")
        ET.SubElement(
            root.find("worldbody"),
            "light",
            pos=".1 -.2 1.0",
            dir="0 0 -1",
            directional="true",
            diffuse=".5 .5 .5",
            ambient=".15 .15 .15",
            specular=".15 .15 .15",
        )
        ET.SubElement(
            asset,
            "texture",
            name="showcase_background",
            type="skybox",
            builtin="gradient",
            width="512",
            height="3072",
            rgb1=".973 .969 .949",
            rgb2=".973 .969 .949",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        model.vis.global_.offwidth, model.vis.global_.offheight = 480, 320
        model.geom_group[model.geom("object").id] = 2
        scene_options = mujoco.MjvOption()
        # Wuji supplies coincident visual and collision meshes. Draw the
        # source visual group plus object, avoiding z-fighting from duplicates.
        scene_options.geomgroup[0] = 0
        data = mujoco.MjData(model)
        joints = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        object_id = int(model.joint("object_joint").qposadr[0])
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.075, 0.0, 0.525]
        camera.distance, camera.azimuth, camera.elevation = 0.27, 120, -50
        video, poster = output / f"size-{index}.mp4", output / f"initial-{index}.webp"
        with mujoco.Renderer(model, height=320, width=480) as renderer:

            def image(
                state: np.ndarray,
                *,
                data: mujoco.MjData = data,
                model: mujoco.MjModel = model,
                joints: np.ndarray = joints,
                object_id: int = object_id,
                renderer: mujoco.Renderer = renderer,
                camera: mujoco.MjvCamera = camera,
                scene_options: mujoco.MjvOption = scene_options,
            ) -> np.ndarray:
                data.qpos[joints] = state[13:33]
                data.qpos[object_id : object_id + 7] = state[:7]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=scene_options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                return renderer.render().copy()

            Image.fromarray(image(initial[row])).save(poster, quality=90)
            with imageio.get_writer(
                video,
                fps=20,
                codec="libx264",
                quality=None,
                macro_block_size=1,
                ffmpeg_params=[
                    "-crf",
                    "22",
                    "-preset",
                    "fast",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            ) as writer:
                for frame in range(400):
                    writer.append_data(image(states[frame, row]))
            Image.fromarray(image(states[240, row])).save(
                output / f"preview-{index}.png"
            )
        items.append(
            {
                "id": f"{scale:g}",
                "label": f"{scale:g}",
                "row": row,
                "video": video.name,
                "poster": poster.name,
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
        )
        print(f"Rendered Wuji {scale:g}", flush=True)
    manifest = {
        "hand": "Wuji Hand V1 right",
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "measured_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": packet["seeds"][0],
        "selection": "First seed, first row per size; fixed indices independent of outcome",
        "seconds": 20,
        "fps": 20,
        "rendering": "Actual native measured poses; corresponding Wuji20 MJCF FK only, no physics stepping. Source visual group1 and object are drawn; coincident collision meshes are hidden to avoid z-fighting, shadow mapping disabled to avoid shadow acne. Lighting, background, object checker texture are display-only.",
        "sizes": items,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
