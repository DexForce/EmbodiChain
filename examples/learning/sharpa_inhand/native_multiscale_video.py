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

"""Render all eight native evaluation scales using measured states only."""

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
    """Show the first fixed-index episode of every scale from an accepted run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.evaluation.resolve()
    provenance = json.loads((folder / "provenance.json").read_text())
    if (
        not provenance["checkpoint_trained_in_native_environment"]
        or not provenance["native_training_success_qualified"]
    ):
        raise ValueError("Expected an independently accepted native-trained checkpoint")
    path = folder / "measured_states.pt"
    packet = torch.load(path, map_location="cpu", weights_only=False)
    states = torch.stack(packet["physical_states"][:400]).numpy()
    trace = torch.load(
        folder / "physical_rotation_traces.pt", map_location="cpu", weights_only=False
    )[0].numpy()
    contexts = packet["initial_contexts"][0].numpy()
    scales = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    rows = []
    for scale in scales:
        matches = np.flatnonzero(np.isclose(contexts[:, 49], scale))
        if not len(matches):
            raise ValueError(f"Missing physical scale {scale}")
        rows.append(int(matches[0]))
    states = states[:, rows]
    trace = trace[:, rows]
    evaluation = json.loads((folder / "evaluation.json").read_text())
    first_seed = packet["seeds"][0]
    episodes = {
        record["row"]: record
        for record in evaluation["records"]
        if record["seed"] == first_seed
    }
    episode_dropped = [bool(episodes[row]["dropped"]) for row in rows]
    angles = np.cumsum(trace[..., 0] * trace[..., 1], axis=0)
    last_valid = []
    for column in range(8):
        stopped = np.flatnonzero(trace[:, column, 1] == 0)
        last_valid.append(int(stopped[0] - 1) if len(stopped) else 399)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.09, 0, 0.59]
    camera.distance, camera.azimuth, camera.elevation = 0.24, 45, -25
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_path, 30)
    label = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 20)
    models = []
    renderers = []
    source_root = (
        args.upstream.resolve() / "src/sharpa_rl_unilab/assets/robots/sharpa_wave"
    )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        for scale in scales:
            model = mujoco.MjModel.from_xml_path(
                str(source_root / f"scene_scale_{scale:g}.xml")
            )
            model.vis.global_.offwidth = 480
            model.vis.global_.offheight = 320
            models.append(
                (
                    model,
                    mujoco.MjData(model),
                    model.jnt_qposadr[model.actuator_trnid[:, 0]],
                    int(model.joint("object_joint").qposadr[0]),
                )
            )
            renderers.append(mujoco.Renderer(model, height=320, width=480))
        with imageio.get_writer(
            args.output, fps=20, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for index in range(400):
                canvas = Image.new("RGB", (1920, 920), (246, 249, 252))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (20, 6),
                    "EmbodiChain 原生仿真 · APG 从头训练 · 八尺度随机化",
                    font=title,
                    fill=(20, 35, 53),
                )
                draw.text(
                    (20, 47),
                    f"t={(index+1)*.05:05.2f}s  /  20s · 独立物理评估 · 实时 1x",
                    font=small,
                    fill=(65, 80, 96),
                )
                for column, (
                    scale,
                    (model, data, joint_ids, object_id),
                    renderer,
                ) in enumerate(zip(scales, models, renderers, strict=True)):
                    row = states[min(index, last_valid[column]), column]
                    data.qpos[joint_ids] = row[13:35]
                    data.qpos[object_id : object_id + 3] = row[:3]
                    data.qpos[object_id + 3 : object_id + 7] = row[3:7]
                    # Forward kinematics for rendering; never integrate physics.
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    left, top = (column % 4) * 480, 80 + (column // 4) * 380
                    canvas.paste(Image.fromarray(renderer.render()), (left, top))
                    angle = angles[min(index, last_valid[column]), column]
                    dropped = episode_dropped[column] and index >= last_valid[column]
                    draw.text(
                        (left + 10, top + 324),
                        f"尺度 {scale:g}  |  {angle:+.2f} rad"
                        + ("  已掉落" if dropped else ""),
                        font=label,
                        fill=(190, 40, 40) if dropped else (15, 115, 90),
                    )
                draw.text(
                    (20, 849),
                    f"seed {packet['seeds'][0]} · 每种尺寸取固定首个场景 · 质量 / 重心 / 摩擦 / PD / 重力 / 外力随机化",
                    font=small,
                    fill=(65, 80, 96),
                )
                draw.text(
                    (20, 883),
                    "由实测原生状态渲染；画面使用对应尺寸的参考模型，不进行轨迹积分",
                    font=small,
                    fill=(65, 80, 96),
                )
                writer.append_data(np.asarray(canvas))
                if index == 240:
                    canvas.save(args.output.with_suffix(".preview.png"))
    finally:
        for renderer in renderers:
            renderer.close()
    manifest = {
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "measured_state_file": str(path),
        "measured_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection": "first seed, first row matching each scale; fixed indices independent of outcome",
        "seed": packet["seeds"][0],
        "rows": rows,
        "scales": scales,
        "final_angles_rad": [float(angles[last_valid[i], i]) for i in range(8)],
        "dropped": episode_dropped,
        "seconds": 20,
        "fps": 20,
        "native_physics": True,
        "rendering": "matching reference scale MJCF with measured native qpos/root pose; forward kinematics only, no physics stepping",
        "scope": provenance["scope"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
