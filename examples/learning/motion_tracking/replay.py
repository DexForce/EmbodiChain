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

"""Render saved motion-tracking evaluation states with FastRT.

The input is the self-contained NPZ written by evaluate_online.py. Recorded
successor states and their corresponding reference states are replayed without
advancing physics. Select several clips to produce a labelled video playlist.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time


def build_parser() -> argparse.ArgumentParser:
    """Build the playback CLI without initializing a renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asset",
        default="LAFAN1G1/robot/g1_29dof_rev_1_0.urdf",
        help="Prepared robot asset ID or absolute URDF path.",
    )
    parser.add_argument(
        "--clips", help="Comma-separated clips in playback order; default: first clip."
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="Duration per clip."
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--denoising", choices=("off", "optix", "dlss"), default="dlss")
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument("--ghost-offset", type=float, nargs=2, default=(0.3, 1.65))
    return parser


def _load(args: argparse.Namespace) -> dict:
    import numpy as np

    with np.load(args.trajectory, allow_pickle=False) as data:
        required = {
            "state",
            "reference",
            "active",
            "names",
            "dt",
            "joint_names",
            "tracked_link_names",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Trajectory lacks {sorted(missing)}; record it with evaluate_online.py --trajectories."
            )
        names = data["names"].tolist()
        joints = data["joint_names"].tolist()
        tracked = data["tracked_link_names"].tolist()
        dt = float(data["dt"])
        state, reference, active = data["state"], data["reference"], data["active"]
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Trajectory dt must be finite and positive.")
        if state.ndim != 3 or state.shape[1:] != (len(names), 18 + 2 * len(joints)):
            raise ValueError("Trajectory states must match clip and joint counts.")
        if reference.shape != state.shape or active.shape != state.shape[:2]:
            raise ValueError("Reference and active arrays must match recorded states.")
        selected = args.clips.split(",") if args.clips else names[:1]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select at least one clip, without duplicate names.")
        clips = []
        first = round(args.start_seconds / dt)
        count = max(1, round(args.seconds / dt))
        for name in selected:
            if name not in names:
                raise ValueError(f"Unknown clip {name!r}; available clips: {names}")
            index = names.index(name)
            valid = np.asarray(active[:, index], dtype=bool)
            length = int(valid.sum())
            if not np.array_equal(valid, np.arange(len(valid)) < length):
                raise ValueError(
                    f"Clip {name!r} must contain one contiguous active prefix."
                )
            if first >= length:
                raise ValueError(f"Clip {name!r} ends before the requested start time.")
            end = min(first + count, length)
            actual = np.asarray(state[first:end, index], dtype=np.float32).copy()
            target = np.asarray(reference[first:end, index], dtype=np.float32).copy()
            if not np.isfinite(actual).all() or not np.isfinite(target).all():
                raise ValueError(f"Clip {name!r} contains a non-finite recorded state.")
            clips.append(
                {
                    "name": name,
                    "state": actual,
                    "reference": target,
                    "world_actual": np.asarray(state[:length, index, :2]).copy(),
                    "world_reference": np.asarray(reference[:length, index, :2]).copy(),
                    "start": first,
                }
            )
    return {"clips": clips, "joint_names": joints, "tracked_links": tracked, "dt": dt}


def _poses(tree, states):
    import numpy as np
    import torch

    x = torch.as_tensor(states, dtype=tree.dtype, device=tree.device)
    dof = len(tree.joint_names)
    links = tree.evaluate(
        x[..., :3],
        x[..., 3:12].unflatten(-1, (3, 3)),
        x[..., 18 : 18 + dof],
        x[..., 12:15],
        x[..., 15:18],
        x[..., 18 + dof :],
    )
    result = np.broadcast_to(np.eye(4), (*links.position.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = links.rotation.cpu().numpy()
    result[..., :3, 3] = links.position.cpu().numpy()
    return result


def _caption(image, *, clip, local_frame, segment, schedule, dt):
    from PIL import ImageDraw
    from _replay_hud import load_font

    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale = min(width / 1920, height / 1080)

    def xy(x, y):
        return round(x * scale), round(y * scale)

    font, small = load_font(max(1, round(23 * scale))), load_font(
        max(1, round(19 * scale))
    )
    header, footer = round(54 * scale), round(86 * scale)
    draw.rectangle((0, 0, width, header), fill="#18232e")
    source_time = (clip["start"] + local_frame + 1) * dt
    draw.text(
        xy(20, 13),
        f"G1 | {clip['name']} | source {source_time:.2f} s",
        font=font,
        fill="#f2f6f8",
    )
    draw.text(
        (width - xy(20, 0)[0], xy(0, 15)[1]),
        "EmbodiChain + FastRT",
        font=small,
        fill="#b9c9d3",
        anchor="ra",
    )
    top = height - footer
    draw.rectangle((0, top, width, height), fill="#18232e")
    draw.text(
        (xy(20, 0)[0], top + xy(0, 10)[1]),
        f"Clip {segment + 1}/{len(schedule)}: {clip['name']}",
        font=font,
        fill="#f2f6f8",
    )
    remaining = (schedule[segment]["frames"] - local_frame - 1) * dt
    next_label = "Next clip" if segment + 1 < len(schedule) else "End"
    draw.text(
        (width - xy(20, 0)[0], top + xy(0, 13)[1]),
        f"{next_label} in {remaining:.1f} s",
        font=small,
        fill="#b9c9d3",
        anchor="ra",
    )
    draw.text(
        (xy(20, 0)[0], top + xy(0, 42)[1]),
        "Tracking | independent evaluation clips",
        font=small,
        fill="#93d5b0",
    )
    draw.text(
        (width - xy(20, 0)[0], top + xy(0, 42)[1]),
        "Red: policy points   Green: reference points / robot",
        font=small,
        fill="#b9c9d3",
        anchor="ra",
    )
    total = sum(item["frames"] for item in schedule)
    offset, margin = 0, xy(20, 0)[0]
    for index, item in enumerate(schedule):
        left = margin + (width - 2 * margin) * offset / total
        right = margin + (width - 2 * margin) * (offset + item["frames"]) / total
        draw.rectangle(
            (left, height - xy(0, 12)[1], right - 2, height - xy(0, 8)[1]),
            fill="#426075",
        )
        completed = (
            item["frames"]
            if index < segment
            else local_frame + 1 if index == segment else 0
        )
        if completed:
            end = left + (right - left) * completed / item["frames"]
            draw.rectangle(
                (left, height - xy(0, 12)[1], max(left, end - 1), height - xy(0, 8)[1]),
                fill="#62c697",
            )
        offset += item["frames"]


def main() -> None:
    """Render recorded clips and save the video with its playback metadata."""
    args = build_parser().parse_args()
    import math

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("--seconds must be finite and positive.")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        raise ValueError("--start-seconds must be finite and nonnegative.")
    if min(args.resolution) < 2 or args.spp < 1 or args.warmup_frames < 0:
        raise ValueError("Resolution, spp and warmup must have valid positive sizes.")
    if any(value % 2 for value in args.resolution):
        raise ValueError("Video resolution dimensions must be even.")
    if not all(math.isfinite(value) for value in args.ghost_offset):
        raise ValueError("Ghost offsets must be finite.")
    recording = _load(args)
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from _replay_hud import TrajectoryHUD
    from _replay_visuals import FloorTrajectory
    from dexsim import spawn
    from embodichain.compute.kinematics import TreeKinematics
    from embodichain.compute.kinematics.features import heading_rotation
    from embodichain.data import get_data_path
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationRootPropertiesCfg,
        DenoisingCfg,
        DLSSCfg,
        LightCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RobotCfg,
        VisualObjectCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.shapes import MeshCfg
    from embodichain.utils.math import quat_from_matrix
    from embodichain.lab.visualization.markers import (
        MarkerPrototypeCfg,
        RobotMarkers,
        RobotMarkersAppearanceCfg,
        RobotMarkersCfg,
        VisualizationMarkers,
        VisualizationMarkersCfg,
    )

    torch.set_num_threads(4)
    asset = Path(get_data_path(args.asset))
    tree = TreeKinematics(asset.read_text(), recording["joint_names"])
    tracked = [tree.link_names.index(name) for name in recording["tracked_links"]]
    if not tracked:
        raise ValueError("Trajectory must identify at least one tracking link.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    width, height = args.resolution
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            headless=True,
            device=str(device),
            gpu_id=device.index,
            arena_space=0.0,
            width=width,
            height=height,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(
                renderer="fast-rt",
                spp=args.spp,
                denoising=DenoisingCfg(window=args.denoising, offscreen=args.denoising),
                dlss=DLSSCfg(dlss_quality=5),
            ),
            physics_cfg=NewtonPhysicsCfg(
                device=str(device),
                sync_to_renderer=False,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
        )
    )
    camera = None
    robot = None
    robots = []
    balls = []
    floor_trajectory = None
    hud_metadata = []
    started = time.perf_counter()
    try:
        grid_texture = args.output.with_suffix(".floor.png")
        pixels = np.arange(512, dtype=np.float32) + 0.5
        minor = np.minimum(pixels % 128, 128 - pixels % 128) < 2
        major = np.minimum(pixels % 256, 256 - pixels % 256) < 3
        tile = np.empty((512, 512, 3), dtype=np.uint8)
        tile[:] = (175, 182, 188)
        tile[minor[:, None] | minor[None, :]] = (136, 148, 158)
        tile[major[:, None] | major[None, :]] = (124, 137, 148)
        Image.fromarray(tile).save(grid_texture)
        floor_mesh = args.output.with_suffix(".floor.obj")
        floor_mesh.write_text(
            "v -100 -100 0\nv 100 -100 0\nv 100 100 0\nv -100 100 0\n"
            "vt 0 0\nvt 100 0\nvt 100 100\nvt 0 100\nvn 0 0 1\n"
            "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
        )
        sim.add_visual_object(
            VisualObjectCfg(
                uid="replay_floor",
                init_pos=(0, 0, -0.005),
                shape=MeshCfg(
                    fpath=str(floor_mesh.resolve()),
                    visual_material=VisualMaterialCfg(
                        base_color=[1, 1, 1, 1],
                        roughness=1.0,
                        base_color_texture=str(grid_texture.resolve()),
                    ),
                ),
            )
        )
        sim.add_light(
            LightCfg(
                uid="replay_key",
                light_type="direction",
                direction=(-0.4, -0.5, -1),
                intensity=3.0,
            )
        )
        robot = sim.add_robot(
            RobotCfg(
                uid="recorded_policy",
                fpath=str(asset),
                root_props=ArticulationRootPropertiesCfg(fixed_base=False),
            )
        )
        sim.prepare()
        joint_order = [
            recording["joint_names"].index(name) for name in robot.joint_names
        ]
        recorded_qpos = torch.as_tensor(
            np.concatenate(
                [
                    clip["state"][:, 18 : 18 + len(joint_order)]
                    for clip in recording["clips"]
                ]
            )[:, joint_order],
            dtype=torch.float32,
            device=device,
        )
        # Preserve recorded excursions beyond the source's soft joint limits.
        limits = robot.get_qpos_limits().clone()
        limits[..., 0] = torch.minimum(limits[..., 0], recorded_qpos.amin(0) - 1e-5)
        limits[..., 1] = torch.maximum(limits[..., 1], recorded_qpos.amax(0) + 1e-5)
        robot.set_qpos_limits(limits)
        sim.set_ground_plane_visibility(False)
        robots = [
            RobotMarkers(
                RobotMarkersCfg(
                    name="reference_ghost",
                    asset_path=str(asset),
                    appearance=RobotMarkersAppearanceCfg(
                        mode="override",
                        color=(0.08, 0.83, 0.24),
                        opacity=1.0,
                        unlit=False,
                    ),
                ),
                sim,
            )
        ]
        visual_links = [tree.link_names.index(name) for name in robots[0].link_names]
        colors = ((1.0, 0.03, 0.03, 1.0), (0.02, 1.0, 0.08, 1.0))
        balls = [
            VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"tracking_{index}",
                    markers={
                        "point": MarkerPrototypeCfg(
                            shape="sphere", scale=(0.044, 0.044, 0.044), color=color
                        )
                    },
                ),
                sim,
            )
            for index, color in enumerate(colors)
        ]
        camera = sim.spawn_result.add_camera(
            spawn.CameraDesc(name="motion_replay", resolution=(width, height))
        )
        camera.open_camera()
        camera.set_visual_overlay_enabled(True)
        dt = recording["dt"]
        schedule = [
            {
                "clip": item["name"],
                "frames": len(item["state"]),
                "source_start_s": (item["start"] + 1) * dt,
            }
            for item in recording["clips"]
        ]
        frames = 0
        with imageio.get_writer(
            str(args.output),
            fps=1 / dt,
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=[
                "-crf",
                "16",
                "-preset",
                "medium",
                "-threads",
                "4",
                "-movflags",
                "+faststart",
            ],
        ) as writer:
            for segment, clip in enumerate(recording["clips"]):
                actual, target = clip["state"], clip["reference"]
                poses = (_poses(tree, actual), _poses(tree, target))
                root_rotation = torch.from_numpy(actual[:, 3:12].reshape(-1, 3, 3))
                root_pose = torch.cat(
                    (torch.from_numpy(actual[:, :3]), quat_from_matrix(root_rotation)),
                    dim=-1,
                ).to(device)
                qpos = torch.as_tensor(
                    actual[:, 18 : 18 + len(joint_order)][:, joint_order], device=device
                )
                target_rotation = torch.from_numpy(target[:, 3:12].reshape(-1, 3, 3))
                alignment = (
                    heading_rotation(root_rotation)
                    @ heading_rotation(target_rotation).transpose(-1, -2)
                ).numpy()
                root_a, root_r = actual[:, :3].copy(), target[:, :3].copy()
                root_a[:, 2] = root_r[:, 2] = 0
                translation = root_a - np.einsum("fij,fj->fi", alignment, root_r)
                aligned = poses[1].copy()
                aligned[..., :3, :3] = alignment[:, None] @ aligned[..., :3, :3]
                aligned[..., :3, 3] = (
                    np.einsum("fij,flj->fli", alignment, aligned[..., :3, 3])
                    + translation[:, None]
                )
                ghost = aligned.copy()
                ghost[..., :2, 3] += args.ghost_offset
                hud = TrajectoryHUD(
                    clip["world_actual"], clip["world_reference"], dt=dt
                )
                hud_metadata.append({"clip": clip["name"], **hud.metadata()})
                floor_trajectory = FloorTrajectory(
                    sim, clip["world_reference"], clip["world_actual"], dt=dt
                )
                body_error = np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                poses[0][..., :3, 3][:, tracked]
                                - aligned[..., :3, 3][:, tracked]
                            )
                            ** 2,
                            axis=-1,
                        ),
                        axis=-1,
                    )
                )
                for frame in range(len(actual)):
                    floor_trajectory.update(clip["start"] + frame)
                    robot.set_state(
                        root_pose=root_pose[frame : frame + 1],
                        qpos=qpos[frame : frame + 1],
                    )
                    robots[0].visualize_link_poses(
                        ghost[frame : frame + 1, visual_links]
                    )
                    balls[0].visualize(translations=poses[0][frame, tracked, :3, 3])
                    balls[1].visualize(translations=aligned[frame, tracked, :3, 3])
                    center = np.array(
                        [
                            actual[frame, 0] + args.ghost_offset[0] / 2,
                            actual[frame, 1] + args.ghost_offset[1] / 2,
                            0.55,
                        ]
                    )
                    camera.look_at(
                        (center + [3.1, -3.1, 1.7]).tolist(), center.tolist(), [0, 0, 1]
                    )
                    sim.sync_render_state()
                    if frame == 0:
                        for _ in range(args.warmup_frames):
                            camera.render()
                    camera.render()
                    rgb = np.array(camera.get_rgb_map(), copy=True)[..., :3]
                    if np.issubdtype(rgb.dtype, np.floating):
                        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    image = Image.fromarray(rgb)
                    _caption(
                        image,
                        clip=clip,
                        local_frame=frame,
                        segment=segment,
                        schedule=schedule,
                        dt=dt,
                    )
                    hud.draw(image, clip["start"] + frame, float(body_error[frame]))
                    writer.append_data(np.asarray(image))
                    if frames == 0:
                        image.save(args.output.with_suffix(".png"))
                    frames += 1
                floor_trajectory.close()
                floor_trajectory = None
        versions = {}
        for name in ("embodichain", "torch", "numpy", "imageio", "warp-lang"):
            versions[name] = importlib.metadata.version(name)
        result = {
            "video": str(args.output),
            "trajectory": str(args.trajectory),
            "asset": str(asset),
            "frames": frames,
            "fps": 1 / dt,
            "resolution": [width, height],
            "renderer": "fast-rt",
            "denoising": args.denoising,
            "spp": args.spp,
            "physics_steps": 0,
            "floor_grid_spacing_m": 0.5,
            "mode": "saved-state playback",
            "actual_renderer": "scene Robot with source materials",
            "reference_renderer": "opaque green RobotMarkers with scene lighting",
            "visual_style": "compact-world-xy-floor-trails-v1",
            "world_xy_hud": hud_metadata,
            "floor_trajectory": {
                "history_seconds": 3,
                "future_seconds": 2,
                "coordinates": "unaligned world root XY",
            },
            "schedule": schedule,
            "playlist_boundary": "Each selected clip starts from its recorded evaluation state.",
            "reference_alignment": "Root heading and horizontal position; display-only ghost offset.",
            "ghost_offset_xy_m": args.ghost_offset,
            "tracked_links": recording["tracked_links"],
            "tracking_colors": {"actual": "red", "reference": "green"},
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "software": versions,
            "seconds": time.perf_counter() - started,
        }
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if floor_trajectory is not None:
            floor_trajectory.close()
        camera = None
        balls.clear()
        robots.clear()
        robot = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
