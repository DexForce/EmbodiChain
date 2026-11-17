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

Sample = dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    """Configure the headless sequence and output format."""
    parser = argparse.ArgumentParser(description=__doc__)
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


def export(samples: list[Sample], args: argparse.Namespace) -> None:
    """Save the raw data, the middle PNG and the optional full sequence."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
    )
    figure = plt.figure(figsize=(8.8, 4.0), dpi=120, layout="constrained")
    draw_terrain(figure, samples[len(samples) // 2])
    figure.savefig(args.output_dir / "preview.png")
    if args.video:
        writer = animation.FFMpegWriter(
            fps=args.fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"]
        )
        with writer.saving(figure, args.output_dir / "preview.mp4", dpi=120):
            for sample in samples:
                draw_terrain(figure, sample)
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
            startup_summary="off",
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            physics_cfg=DefaultPhysicsCfg(device="cpu"),
        )
    )
    try:
        samples = record_terrain(sim, args.frames)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    export(samples, args)


if __name__ == "__main__":
    main()
