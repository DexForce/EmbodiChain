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

"""Render saved native physics states without integrating another trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

__all__ = ["main"]


def main() -> None:
    """Render the first preselected native evaluation episode from measured states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.evaluation.resolve()
    provenance = json.loads((folder / "provenance.json").read_text())
    if not provenance["checkpoint_trained_in_native_environment"]:
        raise ValueError("Expected a native-trained checkpoint's measured evaluation")
    measured_path = folder / "measured_states.pt"
    packet = torch.load(measured_path, map_location="cpu", weights_only=False)
    states = torch.stack(packet["physical_states"][:400])[:, 0].numpy()
    trace = torch.load(
        folder / "physical_rotation_traces.pt", map_location="cpu", weights_only=False
    )[0][:, 0].numpy()
    angles = np.cumsum(trace[:, 0] * trace[:, 1])
    source = (
        args.upstream.resolve()
        / "src/sharpa_rl_unilab/assets/robots/sharpa_wave/scene_scale_1.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(source))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.09, 0, 0.59]
    camera.distance, camera.azimuth, camera.elevation = 0.24, 45, -25
    renderer = mujoco.Renderer(model, height=640, width=960)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title_font = ImageFont.truetype(font_path, 26)
    font = ImageFont.truetype(font_path, 21)
    joint_ids = model.actuator_trnid[:, 0]
    q_ids = model.jnt_qposadr[joint_ids]
    object_id = model.joint("object_joint").qposadr[0]
    stopped = np.flatnonzero(trace[:, 1] == 0)
    last_valid = int(stopped[0] - 1) if len(stopped) else 399
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        with imageio.get_writer(
            args.output, fps=20, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for index in range(400):
                row = states[min(index, last_valid)]
                data.qpos[q_ids] = row[13:35]
                data.qpos[object_id : object_id + 3] = row[:3]
                data.qpos[object_id + 3 : object_id + 7] = row[3:7]
                # Forward kinematics/scene publication only: no mj_step call.
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                canvas = Image.new("RGB", (960, 788), (246, 249, 252))
                canvas.paste(image, (0, 78))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (20, 8),
                    "EmbodiChain 原生仿真 · APG 从头训练",
                    font=title_font,
                    fill=(20, 35, 53),
                )
                draw.text(
                    (20, 45),
                    f"固定尺度 1.0 · seed {packet['seeds'][0]} / 场景 0 · 实时 1x",
                    font=font,
                    fill=(65, 80, 96),
                )
                draw.text(
                    (20, 723),
                    f"t={(index+1)*0.05:05.2f}s  累计转角 {angles[min(index,last_valid)]:+.2f} rad",
                    font=font,
                    fill=(20, 35, 53),
                )
                status = "完整 20s 物理轨迹" if not len(stopped) else "已掉落，定格末帧"
                draw.text(
                    (20, 754),
                    status + " · 由实测状态渲染",
                    font=font,
                    fill=(15, 115, 90) if not len(stopped) else (190, 40, 40),
                )
                writer.append_data(np.asarray(canvas))
                if index == 240:
                    canvas.save(args.output.with_suffix(".preview.png"))
    finally:
        renderer.close()
    manifest = {
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "measured_state_file": str(measured_path),
        "measured_states_sha256": hashlib.sha256(
            measured_path.read_bytes()
        ).hexdigest(),
        "episode_selection": "first seed, row0, selected by fixed index rather than outcome",
        "seed": packet["seeds"][0],
        "row": 0,
        "seconds": 20,
        "fps": 20,
        "native_physics": True,
        "rendering": "reference MJCF kinematics from measured native qpos/root pose; no trajectory integration",
        "scope": provenance["scope"],
        "angle_rad": float(angles[last_valid]),
        "dropped": bool(len(stopped)),
        "camera": {
            "lookat": [-0.09, 0, 0.59],
            "distance": 0.24,
            "azimuth": 45,
            "elevation": -25,
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
