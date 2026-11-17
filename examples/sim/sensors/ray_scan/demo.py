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

"""Record geometric ray-sensor samples from a headless simulation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import VisualizationCfg

from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

TERRAIN_SIZE = (2.8, 2.4)
TERRAIN_SPACING = 0.08
TERRAIN_RANGE = 1.48
TERRAIN_MOUNT_Z = 1.29
TERRAIN_TILT = math.radians(28)
# Each box specifies its scene UID, XY center and XYZ extents, all in metres.
TERRAIN_BOXES = (
    ("low_box", (-0.6, 0.48), (0.67, 0.54, 0.35)),
    ("high_box", (0.18, -0.50), (0.56, 0.50, 0.63)),
    ("short_box", (0.78, 0.48), (0.48, 0.58, 0.44)),
)
BOX_COLORS = ("#81b7b9", "#477f95", "#82a5c5")
CAMERA_CENTER = (0.0, 0.0, 0.68)
DEPTH_SIZE = (112, 76)
COLLISION_SIZE = (0.42, 0.82, 0.91)
VISUAL_SIZE = (0.62, 0.70, 1.06)

Sample = dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    """Configure the headless sequence and output format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("terrain", "dynamic"), default="terrain")
    parser.add_argument(
        "--geometry",
        choices=("collision", "visual"),
        default="collision",
        help="Target geometry for dynamic ray sensors.",
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video", action="store_true", help="Also export H.264 MP4.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def add_box(
    sim: SimulationManager,
    uid: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    body_type: str = "static",
) -> RigidObject:
    """Add a geometric scene box."""
    return sim.add_rigid_object(
        RigidObjectCfg(
            uid=uid, body_type=body_type, init_pos=center, shape=CubeCfg(size=size)
        )
    )


def read_sample(sensor: BaseSensor) -> Sample:
    """Copy data and metadata for the first arena's last completed sample."""
    metadata = sensor.get_sample_metadata()
    if metadata is None or not metadata["valid"][0].item():
        raise RuntimeError(f"No completed sample from {sensor.uid}.")
    return {
        name: value[0].detach().cpu().numpy().copy()
        for fields in (sensor.get_data(), metadata)
        for name, value in fields.items()
    }


def record_terrain(sim: SimulationManager, frames: int) -> list[Sample]:
    """Move a downward-looking terrain scanner over static geometry."""
    for uid, (x, y), size in TERRAIN_BOXES:
        add_box(sim, uid, (x, y, size[2] / 2), size)
    mount = add_box(
        sim, "scanner", (0.0, 0.0, TERRAIN_MOUNT_Z), (0.03, 0.03, 0.03), "kinematic"
    )
    sensor = sim.add_sensor(
        SensorCfg.from_dict(
            {
                "uid": "terrain",
                "sensor_type": "TerrainScan",
                "target_uids": ["default_plane", *[uid for uid, _, _ in TERRAIN_BOXES]],
                "mount": {"parent": "scanner"},
                "size": TERRAIN_SIZE,
                "resolution": TERRAIN_SPACING,
                "max_distance": TERRAIN_RANGE,
            }
        )
    )
    samples = []
    camera = scene_camera(
        sim,
        (2.2, -3.55, 2.9),
        (0.15, 0.0, 0.35),
        {
            "scanner": (0.95, 0.49, 0.17),
            **{uid: (0.19, 0.47, 0.61) for uid, _, _ in TERRAIN_BOXES},
        },
    )
    for y in np.linspace(-0.35, 0.35, frames):
        mount.set_local_pose(
            torch.tensor(
                [
                    [
                        0.32,
                        y,
                        TERRAIN_MOUNT_Z,
                        0.0,
                        math.sin(TERRAIN_TILT / 2),
                        0.0,
                        math.cos(TERRAIN_TILT / 2),
                    ]
                ],
                dtype=torch.float32,
                device=sim.device,
            )
        )
        sim.update(step=1, render_final_step=False)
        sample = read_sample(sensor)
        sample["scanner_position"] = mount.get_local_pose()[0, :3].cpu().numpy().copy()
        sample["camera_rgb"] = capture_rgb(sim, camera)
        samples.append(sample)
    return samples


def record_dynamic(sim: SimulationManager, frames: int, geometry: str) -> list[Sample]:
    """Sample co-located LiDAR and depth sensors as a box moves."""
    target = sim.add_rigid_object(
        RigidObjectCfg(
            uid="target",
            body_type="kinematic",
            init_pos=(1.7, -0.55, 0.69),
            shape=CubeCfg(size=COLLISION_SIZE),
            visual_shape=CubeCfg(size=VISUAL_SIZE),
        )
    )
    lidar = sim.add_sensor(
        SensorCfg.from_dict(
            {
                "uid": "scan",
                "sensor_type": "RayLidar",
                "mount": {"pos": CAMERA_CENTER},
                "target_uids": ["target"],
                "geometry_source": geometry,
                "vertical_angles": list(range(-20, 21, 2)),
                "horizontal_samples": 300,
                "horizontal_fov": 90.0,
                "max_distance": 4.0,
            }
        )
    )
    depth = sim.add_sensor(
        SensorCfg.from_dict(
            {
                "uid": "depth",
                "sensor_type": "RayDepthCamera",
                # Camera +Z looks along arena +X; camera +Y points down.
                "mount": {"pos": CAMERA_CENTER, "quat": [-0.5, 0.5, -0.5, 0.5]},
                "target_uids": ["target"],
                "geometry_source": geometry,
                "width": DEPTH_SIZE[0],
                "height": DEPTH_SIZE[1],
                "fx": 82.0,
                "fy": 82.0,
                "max_distance": 4.0,
            }
        )
    )
    samples = []
    camera = scene_camera(
        sim, (-0.75, -3.25, 1.65), (1.89, 0.0, 0.65), {"target": (0.14, 0.49, 0.58)}
    )
    for phase in np.linspace(0, 1, frames):
        position = [1.72 + 0.85 * math.sin(math.pi * phase), -0.55 + 1.10 * phase, 0.69]
        target.set_local_pose(
            torch.tensor(
                [[*position, 0, 0, 0, 1]], dtype=torch.float32, device=sim.device
            )
        )
        sim.update(step=1, render_final_step=False)
        lidar_sample, depth_sample = read_sample(lidar), read_sample(depth)
        if (
            lidar_sample["sample_id"] != depth_sample["sample_id"]
            or lidar_sample["time"] != depth_sample["time"]
        ):
            raise RuntimeError("LiDAR and depth image were not acquired together.")
        sample = {f"lidar_{key}": value for key, value in lidar_sample.items()}
        sample.update({f"depth_{key}": value for key, value in depth_sample.items()})
        sample["target_center"] = target.get_local_pose()[0, :3].cpu().numpy().copy()
        sample["target_size"] = np.asarray(
            COLLISION_SIZE if geometry == "collision" else VISUAL_SIZE
        )
        sample["camera_rgb"] = capture_rgb(sim, camera)
        samples.append(sample)
    return samples


def style_scene(
    axis: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    """Apply metre axes and plain panes to a rendered geometry view."""
    axis.set(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Scene and returns",
    )
    axis.set_box_aspect(tuple(high - low for low, high in (xlim, ylim, zlim)))
    axis.set_proj_type("ortho")
    axis.view_init(elev=28, azim=-125)
    axis.set_zlabel("Z (m)", labelpad=-8)
    axis.tick_params(axis="z", pad=1)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.fill = False
    axis.tick_params(labelsize=7)


def draw_range(
    figure: Figure,
    axis: Axes,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    extent: tuple[float, float, float, float] | None,
    limits: tuple[float, float],
    title: str,
    labels: tuple[str, str],
    color_label: str,
    origin: str,
) -> None:
    """Display metre values and render missed rays in gray."""
    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("#dfe4e8")
    image = axis.imshow(
        np.ma.masked_where(~valid, values),
        origin=origin,
        extent=extent,
        vmin=limits[0],
        vmax=limits[1],
        cmap=cmap,
        interpolation="nearest",
    )
    axis.set(title=title, xlabel=labels[0], ylabel=labels[1])
    figure.colorbar(image, ax=axis, label=color_label, shrink=0.71)


def draw_terrain(figure: Figure, sample: Sample) -> None:
    """Render the scene, measured points and scan-local range image."""
    figure.clear()
    show_scene(figure.add_subplot(1, 2, 1), sample["camera_rgb"])
    map_axis = figure.add_subplot(1, 2, 2)

    width, height = TERRAIN_SIZE
    nx, ny = (math.ceil(length / TERRAIN_SPACING) + 1 for length in TERRAIN_SIZE)
    draw_range(
        figure,
        map_axis,
        sample["distance"].reshape(nx, ny).T,
        sample["hit"].reshape(nx, ny).T,
        extent=(-width / 2, width / 2, -height / 2, height / 2),
        limits=(0.6, TERRAIN_RANGE),
        title="Terrain range",
        labels=("Probe X (m)", "Probe Y (m)"),
        color_label="Radial range (m)",
        origin="lower",
    )
    figure.supxlabel(
        f"Sample {sample['sample_id']}   ·   t = {sample['time']:.3f} s   ·   Gray: no hit",
        fontsize=9,
    )


def draw_dynamic(figure: Figure, sample: Sample) -> None:
    """Render the LiDAR returns next to axial depth and camera ray range."""
    figure.clear()
    show_scene(figure.add_subplot(1, 4, 1), sample["camera_rgb"])
    scene = figure.add_subplot(1, 4, 2, projection="3d")
    depth_axis = figure.add_subplot(1, 4, 3)
    range_axis = figure.add_subplot(1, 4, 4)
    returns = sample["lidar_position"][sample["lidar_hit"]]
    scene.scatter(*returns.T, s=6, c="#cd602c", depthshade=False, zorder=4)
    scene.scatter(*CAMERA_CENTER, s=30, c="#223e57", marker="^", depthshade=False)
    style_scene(scene, (-0.1, 3.0), (-1.2, 1.2), (0.05, 1.35))
    scene.set_title("LiDAR points")

    depth = sample["depth_depth"]
    hit = sample["depth_hit"].reshape(depth.shape)
    for axis, values, title in (
        (depth_axis, depth, "Axial depth"),
        (range_axis, sample["depth_distance"].reshape(depth.shape), "Radial range"),
    ):
        draw_range(
            figure,
            axis,
            values,
            hit,
            extent=None,
            limits=(1.25, 2.65),
            title=title,
            labels=("Column (px)", "Row (px)"),
            color_label="Metres",
            origin="upper",
        )
    figure.supxlabel(
        f"Sample {sample['lidar_sample_id']}   ·   t = {sample['lidar_time']:.3f} s   ·   Gray: no hit",
        fontsize=9,
    )


def export(samples: list[Sample], args: argparse.Namespace) -> None:
    """Save the raw data, the middle PNG and the optional full sequence."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
    )
    draw = draw_terrain if args.mode == "terrain" else draw_dynamic
    width = 8.8 if args.mode == "terrain" else 12.6
    figure = plt.figure(figsize=(width, 4.0), dpi=120, layout="constrained")
    preview = len(samples) // (4 if args.mode == "dynamic" else 2)
    draw(figure, samples[preview])
    figure.savefig(args.output_dir / "preview.png")
    if args.video:
        writer = animation.FFMpegWriter(
            fps=args.fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"]
        )
        with writer.saving(figure, args.output_dir / "preview.mp4", dpi=120):
            for sample in samples:
                draw(figure, sample)
                writer.grab_frame()
    plt.close(figure)


def main() -> None:
    """Run the Default CPU demo and write files without an application window."""
    parser = build_parser()
    args = parser.parse_args()
    if args.frames < 2 or args.fps < 1:
        parser.error("--frames must be at least 2 and --fps must be positive")
    if args.video and not animation.writers.is_available("ffmpeg"):
        parser.error("--video requires ffmpeg on PATH")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(backend="none"),
            startup_summary="off",
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            physics_cfg=DefaultPhysicsCfg(device="cpu"),
        )
    )
    try:
        if args.mode == "terrain":
            samples = record_terrain(sim, args.frames)
        else:
            samples = record_dynamic(sim, args.frames, args.geometry)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    export(samples, args)


if __name__ == "__main__":
    main()
