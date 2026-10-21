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
"""Play named robot link trajectories with ghosts, paths and visual task objects.

Input NPZ fields: poses (T,K,L,7), link_names (L,), dt (scalar). Poses use
world xyz+xyzw. Optional fields are tips (T,K,3), objects (T,K,7), targets
(T,K,3), command (T,3) and velocity (T,3). No physics steps are executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from dexsim import spawn
from embodichain.learning.rl.policy_evaluation.camera import PolicyViewerCameraCfg
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg, VisualObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)
from embodichain.lab.visualization.markers import (
    RobotVisualAppearanceCfg,
    MarkerPrototypeCfg,
    RobotVisual,
    RobotVisualCfg,
    VisualizationMarkers,
    VisualizationMarkersCfg,
)

__all__ = ["main"]
COLORS = [(0.15, 0.8, 0.4), (0.2, 0.55, 1.0), (1.0, 0.55, 0.15)]


def _camera_path(
    poses: np.ndarray, offsets: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the whole clip once, then track XY at fixed height and distance."""
    anchors = np.zeros((len(poses), 3), dtype=np.float64)
    if mode == "follow_xy":
        anchors[:, :2] = poses[:, 0, 0, :2]
    else:
        anchors[:, :2] = poses[0, 0, 0, :2]
    relative = poses[..., :3] + offsets[None, :, None] - anchors[:, None, None]
    lower = relative.min(axis=(0, 1, 2))
    upper = relative.max(axis=(0, 1, 2))
    center = (lower + upper) / 2
    spread = max(0.8, float(np.max(upper - lower)))
    cfg = PolicyViewerCameraCfg(
        eye_offset=tuple(np.array([1.25, -1.7, 0.9]) * spread),
        target_height=float(center[2]),
    )
    targets = anchors + center
    targets[:, 2] = cfg.target_height
    # Apply the preset heading once, as in the policy viewer's reset.
    yaw = Rotation.from_quat(poses[0, 0, 0, 3:]).as_euler("xyz")[2]
    offset = Rotation.from_euler("z", yaw).apply(cfg.eye_offset)
    return targets + offset, targets


def main() -> None:
    """Render one recorded task trajectory with explicit visualization defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=["overlap", "offset"], default="overlap")
    parser.add_argument("--spacing", type=float, default=1.1)
    parser.add_argument("--opacity", type=float, default=0.4)
    parser.add_argument("--original-first", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--camera-mode",
        choices=["follow_xy", "fixed"],
        default="follow_xy",
        help="Follow root XY at fixed world height, or use one fixed view.",
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0 or args.spacing < 0:
        parser.error("seconds/fps must be positive and spacing nonnegative")
    data = np.load(args.trajectory, allow_pickle=False)
    poses = np.asarray(data["poses"], dtype=np.float32)
    names = tuple(str(v) for v in data["link_names"])
    if poses.ndim != 4 or poses.shape[-2:] != (len(names), 7):
        parser.error("poses must be (T,K,L,7), matching link_names")
    if not np.isfinite(poses).all() or not len(poses) or not poses.shape[1]:
        parser.error("poses must be finite and nonempty")
    if np.any(np.linalg.norm(poses[..., 3:], axis=-1) < 1e-8):
        parser.error("pose quaternions must be nonzero")
    dt = float(data["dt"])
    if not np.isfinite(dt) or dt <= 0:
        parser.error("dt must be positive and finite")
    count = poses.shape[1]
    offsets = np.zeros((count, 3))
    if args.layout == "offset":
        offsets[:, 1] = (np.arange(count) - (count - 1) / 2) * args.spacing
    camera_eyes, camera_targets = _camera_path(poses, offsets, args.camera_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    sim = SimulationManager(
        SimulationManagerCfg(
            visualization=visualization_cfg_from_args(args),
            headless=not args.live,
            device="cpu",
            num_envs=1,
            width=960,
            height=720,
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer="hybrid",
                spp=1,
                denoising=DenoisingCfg(window="dlss", offscreen="dlss"),
            ),
        )
    )
    items = []
    objects = []
    encoder = None
    try:
        for k in range(count):
            if "objects" in data:
                obj = sim.add_visual_object(
                    VisualObjectCfg(
                        uid=f"task_object_{k}", shape=CubeCfg(size=(0.04, 0.04, 0.06))
                    )
                )
                objects.append(obj)
        sim.prepare()
        for k, obj in enumerate(objects):
            obj.set_visual_material(
                sim.create_visual_material(
                    VisualMaterialCfg(
                        uid=f"object_material_{k}",
                        base_color=[*COLORS[k % len(COLORS)], 1.0],
                    )
                )
            )
        scene = sim._spawn_scene.builder.result
        for name, position, intensity in [
            ("key", [2, -3, 4], 45),
            ("fill", [-2, 2, 3], 15),
        ]:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = position
            scene.add_light(
                spawn.LightDesc(
                    name=name, light_type="POINT", pose=matrix, intensity=intensity
                )
            )
        for k in range(count):
            original = args.original_first and k == 0
            item = RobotVisual(
                RobotVisualCfg(
                    name=f"trajectory_{k}",
                    asset_path=args.asset,
                    appearance=RobotVisualAppearanceCfg(
                        mode="asset" if original else "override",
                        color=COLORS[(k - int(args.original_first)) % len(COLORS)],
                        opacity=1.0 if original else args.opacity,
                    ),
                ),
                sim,
            )
            items.append((item, [names.index(name) for name in item.link_names]))
        marker_cfg = {
            str(k): MarkerPrototypeCfg(
                shape="sphere",
                scale=(0.015, 0.015, 0.015),
                color=(*COLORS[k % len(COLORS)], 0.85),
            )
            for k in range(count)
        }
        paths = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="trajectory_paths", markers=marker_cfg),
            sim,
        )
        if "tips" in data:
            tips = np.asarray(data["tips"])
            sample = tips[:: max(1, len(tips) // 100)] + offsets[None]
            paths.visualize(
                sample.reshape(-1, 3),
                marker_indices=np.tile(np.arange(count), len(sample)),
            )
        targets = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="task_targets",
                markers={
                    "target": MarkerPrototypeCfg(shape="frame", scale=(0.1, 0.1, 0.1))
                },
            ),
            sim,
        )
        arrows = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="velocity",
                markers={
                    "command": MarkerPrototypeCfg(
                        shape="arrow", color=(0.1, 0.9, 0.3, 1)
                    ),
                    "measured": MarkerPrototypeCfg(
                        shape="arrow", color=(0.2, 0.55, 1, 1)
                    ),
                },
            ),
            sim,
        )
        camera = scene.add_camera(
            spawn.CameraDesc(name="trajectory_capture", resolution=(960, 720))
        )
        camera.open_camera()
        camera.set_visual_overlay_enabled(True)
        if args.live:
            sim.open_window()
        else:
            encoder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "960x720",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "4",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(args.output / "video.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        started = time.monotonic()
        frames = round(args.seconds * args.fps)
        rendered_frames = 0
        for frame in range(frames):
            if args.live and time.monotonic() - started >= args.seconds:
                break
            index = min(round(frame / args.fps / dt), len(poses) - 1)
            if args.live:
                index = int((time.monotonic() - started) / dt) % len(poses)
            for k, (item, selected) in enumerate(items):
                row = poses[index, k, selected].copy()
                row[:, :3] += offsets[k]
                item.visualize_link_poses(row[None])
                if objects:
                    pose = np.asarray(data["objects"][index, k]).copy()
                    pose[:3] += offsets[k]
                    objects[k].set_local_pose(torch.as_tensor(pose[None]))
            if "targets" in data:
                targets.visualize(np.asarray(data["targets"][index]) + offsets)
            if "command" in data:
                root = poses[index, 0, 0]
                vectors = np.stack(
                    [data["command"][index], data["velocity"][index]]
                ).copy()
                vectors[:, 2] = 0
                vectors = Rotation.from_quat(root[3:]).apply(vectors)
                angle = np.arctan2(vectors[:, 1], vectors[:, 0])
                origins = np.tile(root[:3], (2, 1))
                origins[:, 2] += [0.2, 0.3]
                scales = np.tile([1.0, 0.35, 0.35], (2, 1))
                scales[:, 0] = np.maximum(0.02, np.linalg.norm(vectors, axis=-1))
                arrows.visualize(
                    origins,
                    Rotation.from_euler("z", angle[:, None]).as_quat(),
                    scales,
                    [0, 1],
                )
            eye, center = camera_eyes[index], camera_targets[index]
            camera.look_at(eye.tolist(), center.tolist(), [0, 0, 1])
            if args.live:
                window = sim.get_world().get_windows()
                if frame == 0:
                    window.set_look_at(eye.tolist(), center.tolist(), [0, 0, 1])
                else:
                    window.translate_camera_orbit(center - previous_center)
                previous_center = center.copy()
            sim.get_world()._flush_render_updates()
            capture_frame = not args.live or frame == 0
            with sim.render_frame(
                force_visualization=True, capture_camera_images=False
            ):
                if capture_frame:
                    camera.render()
            if capture_frame:
                rgb = np.ascontiguousarray(
                    camera.get_rgb_map()[..., :3], dtype=np.uint8
                )
                if frame in (0, frames // 2, frames - 1):
                    Image.fromarray(rgb).save(args.output / f"frame-{frame:04}.png")
                if frame == frames // 2:
                    Image.fromarray(rgb).save(args.output / "poster.png")
                if encoder is not None:
                    encoder.stdin.write(rgb.tobytes())
            if args.live and frame == 0:
                # Start the viewing period after the first complete frame.
                started = time.monotonic()
            rendered_frames += 1
            if frame % args.fps == 0:
                print(f"frame {frame}/{frames}", flush=True)
            if args.live:
                time.sleep(
                    max(0.0, started + (frame + 1) / args.fps - time.monotonic())
                )
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video encoder failed")
        np.savez(
            args.output / "camera-poses.npz",
            eye=camera_eyes,
            target=camera_targets,
            dt=dt,
            mode=args.camera_mode,
        )
        (args.output / "capture.json").write_text(
            json.dumps(
                {
                    "camera_mode": args.camera_mode,
                    "camera_eye_z_range": float(np.ptp(camera_eyes[:, 2])),
                    "camera_target_z_range": float(np.ptp(camera_targets[:, 2])),
                    "camera_offset_range": np.ptp(
                        camera_eyes - camera_targets, axis=0
                    ).tolist(),
                    "asset": args.asset,
                    "trajectory": str(args.trajectory),
                    "layout": args.layout,
                    "spacing": args.spacing,
                    "opacity": args.opacity,
                    "original_first": args.original_first,
                    "frames": rendered_frames,
                    "fps": args.fps,
                    "seconds": (
                        time.monotonic() - started
                        if args.live
                        else rendered_frames / args.fps
                    ),
                    "live": args.live,
                    "physics_steps": sim._visualization_sim_step,
                    "links": len(names),
                    "trajectories": count,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
        )
    finally:
        if args.live and sim.is_window_opened:
            sim.close_window()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        for item, _ in items:
            item.remove()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
