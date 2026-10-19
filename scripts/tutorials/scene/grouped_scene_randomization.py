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

"""Generate, reload, verify and preview grouped layouts from an existing export.

Run from the repository root with ``python -m
scripts.tutorials.scene.grouped_scene_randomization --source <workspace>
--render --preview``. See the adjacent README for requirements and outputs.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from embodichain.gen_sim.scene_engine.randomization.cfg import (
    load_randomization_config,
)
from embodichain.gen_sim.scene_engine.randomization.generation import randomize_scene
from embodichain.gen_sim.scene_engine.randomization.loading import (
    LoadedLayouts,
    load_layouts,
    preview_layouts,
)
from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts


def _verify(
    loaded: LoadedLayouts, baseline: dict[str, Any], targets: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Independently measure mesh bounds and preserve every unselected edge."""
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
        load_glb_mesh,
    )

    export = loaded.output_root / "scene_export"
    config = json.loads((export / "scene_config.json").read_text())
    entries = config["background"] + config["rigid_object"]
    vertices = {}
    for entry in entries:
        mesh = load_glb_mesh(export / entry["shape"]["fpath"])
        points = np.asarray(mesh.vertices) * entry.get("body_scale", [1, 1, 1])
        vertices[entry["uid"]] = points[:, [0, 2, 1]] * [1, -1, 1]
    parents = {
        m["entity_id"]: m["parent_id"] for g in baseline["groups"] for m in g["members"]
    }
    table = baseline["table"]
    center, half = np.array(table["center_xy"]), np.array(table["size_xy"]) / 2
    max_error, min_gap, min_edge = 0.0, math.inf, math.inf
    changes_by_uid = {uid: [] for uid in targets}
    for variant in loaded.variants:
        poses = {
            **loaded.baseline_poses,
            **{c.entity_uid: np.array(c.arena_pose) for c in variant.pose_changes},
        }
        boxes = []
        for group in baseline["groups"]:
            members = [group["root_id"], *(m["entity_id"] for m in group["members"])]
            points = np.concatenate(
                [
                    vertices[uid] @ poses[uid][:3, :3].T + poses[uid][:3, 3]
                    for uid in members
                ]
            )[:, :2]
            minimum, maximum = points.min(0), points.max(0)
            boxes.append(((minimum + maximum) / 2, (maximum - minimum) / 2))
            min_edge = min(
                min_edge,
                float(np.min(minimum - (center - half))),
                float(np.min(center + half - maximum)),
            )
        for uid, parent in parents.items():
            relative = np.linalg.inv(poses[parent]) @ poses[uid]
            original = (
                np.linalg.inv(loaded.baseline_poses[parent])
                @ loaded.baseline_poses[uid]
            )
            if uid not in targets:
                max_error = max(max_error, float(np.max(np.abs(relative - original))))
            else:
                changes_by_uid[uid].append(
                    float(np.linalg.norm(relative[:2, 3] - original[:2, 3]))
                )
        for (a, ah), (b, bh) in combinations(boxes, 2):
            min_gap = min(min_gap, float(np.max(np.abs(a - b) - ah - bh)))
    if max_error > 1e-9 or min_edge < -1e-7:
        raise RuntimeError(
            "Saved layouts failed fixed parent-relative pose or table-bound checks."
        )
    return {
        "layout_count": len(loaded.variants),
        "group_count": len(baseline["groups"]),
        "entity_count": len(loaded.baseline_poses),
        "max_fixed_parent_pose_error": max_error,
        "min_table_edge_clearance_m": min_edge,
        "min_group_aabb_axis_clearance_m": min_gap,
        "member_xy_displacement_range_m": {
            uid: [min(values), max(values)] for uid, values in changes_by_uid.items()
        },
        "physics_steps": 0,
    }


def _render(loaded: LoadedLayouts, baseline: dict[str, Any], output: Path) -> None:
    """Render every saved layout with one native scene and one fixed camera."""
    from PIL import Image, ImageDraw, ImageFont

    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
        load_scene_export_into_sim,
    )
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.scene import SceneChangeSet, SceneView, apply_changes
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.visualization import VisualizationCfg

    width, height = 960, 640
    table = baseline["table"]
    span = max(table["size_xy"])
    target = np.array([*table["center_xy"], table["surface_z"] + 0.1 * span])
    eye = target + span * np.array([0.62, -0.96, 1.52])
    focal = height / (2 * math.tan(math.radians(45) / 2))
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            width=width,
            height=height,
            visualization=VisualizationCfg(backend="none"),
        )
    )
    images = []
    try:
        load_scene_export_into_sim(
            sim=sim, output_root=loaded.output_root, force_static_rigids=True
        )
        camera = sim.add_sensor(
            sensor_cfg=CameraCfg(
                uid="layout_preview",
                width=width,
                height=height,
                intrinsics=(focal, focal, width / 2, height / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(eye), target=tuple(target), up=(0.0, 0.0, 1.0)
                ),
            )
        )
        sim.prepare()
        view = SceneView(sim)
        all_poses = [loaded.baseline_poses] + [
            {change.entity_uid: change.arena_pose for change in variant.pose_changes}
            for variant in loaded.variants
        ]
        for index, poses in enumerate(all_poses):
            apply_changes(view, SceneChangeSet(poses, tuple(poses)))
            # Refresh the native camera after the pose publication; never step physics.
            with sim.render_frame(capture_camera_images=False):
                for _ in range(4):
                    camera.update()
            rgb = camera.get_data()["color"][0, ..., :3].cpu().numpy()
            images.append(Image.fromarray(rgb).copy())
            print(f"Rendered {index}/{len(loaded.variants)}", flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    for index, image in enumerate(images):
        image.save(output / f"layout_{index:03d}.png")
        frame = Image.new("RGB", (width * 2, height + 64), "#111827")
        frame.paste(images[0], (0, 64))
        frame.paste(image, (width, 64))
        draw = ImageDraw.Draw(frame)
        draw.text((24, 20), "ORIGINAL SCENE", font=font, fill="#e5e7eb")
        label = "ORIGINAL" if index == 0 else f"VARIANT {index:02d} / {len(images) - 1}"
        draw.text((width + 24, 20), label, font=font, fill="#67e8f9")
        frame.save(frames / f"frame_{index:03d}.png")

    selected = np.linspace(0, len(images) - 1, min(6, len(images)), dtype=int)
    sheet = Image.new("RGB", (1440, math.ceil(len(selected) / 3) * 360), "#111827")
    draw = ImageDraw.Draw(sheet)
    for cell, index in enumerate(selected):
        x, y = cell % 3 * 480, cell // 3 * 360
        sheet.paste(images[index].resize((480, 320)), (x, y + 40))
        label = "Original" if index == 0 else f"Variant {index:02d}"
        draw.text((x + 16, y + 8), label, font=font, fill="#e5e7eb")
    sheet.save(output / "contact_sheet.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "2",
            "-i",
            str(frames / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-r",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "comparison.mp4"),
        ],
        check=True,
    )
    # Keep the inline animation compact while retaining the full-resolution MP4.
    animation = []
    for path in sorted(frames.glob("frame_*.png")):
        with Image.open(path) as frame:
            animation.append(frame.resize((1152, 422)))
    animation[0].save(
        output / "comparison.gif",
        save_all=True,
        append_images=animation[1:],
        duration=500,
        loop=0,
    )


def main() -> None:
    """Copy an export, generate a saved run, and optionally render or play it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/tabletop_four_bases_demo")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("grouped_randomization.yaml"),
    )
    parser.add_argument(
        "--render", action="store_true", help="Save native PNG/MP4/GIF previews."
    )
    parser.add_argument(
        "--preview", action="store_true", help="Play layouts in Viser until Ctrl-C."
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--layout-format", choices=("json", "packed"), default="json")
    args = parser.parse_args()
    source, workspace = (
        args.source.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
    )
    if source == workspace or source in workspace.parents:
        parser.error("Choose an output workspace outside the source asset directory.")
    if not (source / "scene_export/scene_config.json").is_file():
        parser.error("--source must contain scene_export/scene_config.json.")
    if workspace.exists():
        if any(workspace.iterdir()):
            parser.error(
                "--output-root must be empty; use a new directory for each demo."
            )
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source / "scene_export",
        workspace / "scene_export",
        ignore=shutil.ignore_patterns(
            "scene_randomization_base.yaml", "scene_randomization_geometry"
        ),
    )
    cfg = load_randomization_config(args.config, output_root=workspace)
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=workspace, layout_format=args.layout_format)
    if not result.candidates:
        raise RuntimeError(f"No layouts generated: {result.status}; inspect {run}.")
    loaded = load_layouts(run / "layouts")
    baseline = result.baseline.data
    report = _verify(
        loaded, baseline, tuple(cfg.intra_group.targets) if cfg.intra_group else ()
    )
    report.update(seed=cfg.seed, status=result.status, attempts=result.attempts)
    if report["min_group_aabb_axis_clearance_m"] < cfg.collision_margin_m - 1e-9:
        raise RuntimeError("Saved layouts violate the configured group clearance.")
    (run / "demo_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved run: {run}", flush=True)
    if args.render:
        _render(loaded, baseline, run / "preview")
        print(f"Previews: {run / 'preview'}", flush=True)
    if args.preview:
        from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

        preview_layouts(
            run / "layouts",
            interval=1.5,
            headless=True,
            visualization=VisualizationCfg(
                backend="viser",
                viser_server=ViserServerCfg(host="127.0.0.1", port=args.port),
            ),
        )


if __name__ == "__main__":
    main()
