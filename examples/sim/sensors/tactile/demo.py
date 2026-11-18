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

"""Record completed geometric tactile samples during a controlled indentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RigidObjectCfg, VisualObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.sensors import BaseSensor, SensorCfg
from embodichain.lab.sim.shapes import MeshCfg, SphereCfg
from embodichain.lab.visualization import VisualizationCfg

from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

PAD_SIZE = (0.048, 0.036)
PAD_HEIGHT = 0.14
TOOL_RADIUS = 0.035
MAX_DEPTH = 0.012
RESOLUTION = (32, 24)
CURVATURE_RADIUS = 0.075

Sample = dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    """Configure the pad and local, headless outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pad", choices=("planar", "curved"), default="curved")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rgb", action="store_true", help="Show the optical pad model."
    )
    parser.add_argument("--background", nargs=3, type=float, default=(0.27, 0.35, 0.48))
    parser.add_argument(
        "--x-response", nargs=3, type=float, default=(0.60, -0.12, -0.15)
    )
    parser.add_argument(
        "--y-response", nargs=3, type=float, default=(0.07, 0.43, -0.28)
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video", action="store_true", help="Also export H.264 MP4.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def create_manager(args: argparse.Namespace) -> SimulationManager:
    """Create a headless scene with the Default backend."""
    return SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(backend="none"),
            startup_summary="off",
            enable_ground_plane=False,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            physics_cfg=DefaultPhysicsCfg(device=args.device, gravity=(0.0, 0.0, 0.0)),
        )
    )


def create_scene(
    sim: SimulationManager, args: argparse.Namespace
) -> tuple[RigidObject, BaseSensor]:
    """Add a closed spherical tool and register the geometric pad by sensor type."""
    # The virtual pad has no collider. Draw its undeformed surface in DexSim
    # without adding a new target or affecting the geometric measurements.
    x, y = np.meshgrid(
        np.linspace(-PAD_SIZE[0] / 2, PAD_SIZE[0] / 2, RESOLUTION[0]),
        np.linspace(-PAD_SIZE[1] / 2, PAD_SIZE[1] / 2, RESOLUTION[1]),
    )
    height = (
        np.sqrt(CURVATURE_RADIUS**2 - x**2 - y**2) - CURVATURE_RADIUS
        if args.pad == "curved"
        else np.zeros_like(x)
    )
    vertices = np.stack((x, y, height), axis=-1).reshape(-1, 3)
    nx, ny = RESOLUTION
    cell = (np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)).ravel()
    triangles = np.concatenate(
        (
            np.stack((cell, cell + 1, cell + nx), axis=1),
            np.stack((cell + 1, cell + nx + 1, cell + nx), axis=1),
        )
    )
    sim.add_visual_object(
        VisualObjectCfg(
            uid="pad_surface",
            init_pos=(0, 0, PAD_HEIGHT),
            shape=MeshCfg(vertices=vertices, triangles=triangles),
        )
    )
    tool = sim.add_rigid_object(
        RigidObjectCfg(
            uid="tool",
            body_type="kinematic",
            shape=SphereCfg(radius=TOOL_RADIUS),
            init_pos=(0.0, 0.0, PAD_HEIGHT - TOOL_RADIUS - 0.006),
        )
    )
    optical = None
    if args.rgb:
        optical = {
            "background": args.background,
            "x_response": args.x_response,
            "y_response": args.y_response,
        }
    sensor = sim.add_sensor(
        SensorCfg.from_dict(
            {
                "sensor_type": "TactileIndentation",
                "uid": "pad",
                "target_uids": ["tool"],
                "mount": {"pos": [0.0, 0.0, PAD_HEIGHT]},
                "size": PAD_SIZE,
                "resolution": RESOLUTION,
                "curvature_radius": CURVATURE_RADIUS if args.pad == "curved" else None,
                "max_depth": MAX_DEPTH,
                "rgb": optical,
            }
        )
    )
    return tool, sensor


def move_tool(
    tool: RigidObject, sensor: BaseSensor, args: argparse.Namespace, phase: float
) -> None:
    """Prescribe loading, sliding forward and back, and release."""
    phases = (0.0, 0.25, 0.50, 0.75, 1.0)
    x = np.interp(phase, phases, (-0.005, -0.005, 0.007, -0.005, -0.005))
    overlap = np.interp(phase, phases, (-0.006, 0.008, 0.008, 0.008, -0.006))
    tool.set_local_pose(
        torch.tensor(
            [[x, 0.0, PAD_HEIGHT - TOOL_RADIUS + overlap, 0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=tool.device,
        )
    )


def collect_samples(
    sim: SimulationManager,
    tool: RigidObject,
    sensor: BaseSensor,
    args: argparse.Namespace,
) -> list[Sample]:
    """Step once per pose and copy the latest completed sample for each frame."""
    camera = scene_camera(
        sim,
        (0.058, -0.11, 0.178),
        (0.0, 0.0, 0.125),
        {"pad_surface": (0.12, 0.46, 0.58), "tool": (0.96, 0.57, 0.19)},
    )
    samples = []
    for phase in np.linspace(0.0, 1.0, args.frames):
        move_tool(tool, sensor, args, float(phase))
        sim.update(step=1, render_final_step=False)
        metadata = sensor.get_sample_metadata()
        if metadata is None or not metadata["valid"][0].item():
            raise RuntimeError("No completed tactile sample.")
        sample = {
            name: value[0].detach().cpu().numpy().copy()
            for name, value in sensor.get_data().items()
        }
        sample.update(
            {key: metadata[key][0].cpu().numpy().copy() for key in metadata.keys()}
        )
        sample["tool_center"] = tool.get_local_pose()[0, :3].cpu().numpy().copy()
        sample["camera_rgb"] = capture_rgb(sim, camera)
        samples.append(sample)
    return samples


def draw_depth(figure: Figure, axis: Axes, sample: Sample) -> None:
    """Plot penetration in metres with no-contact pixels masked in gray."""
    depth = np.ma.masked_where(~sample["hit"], sample["depth"])
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#e6e8eb")
    image = axis.imshow(
        depth,
        origin="lower",
        extent=(-PAD_SIZE[0] / 2, PAD_SIZE[0] / 2, -PAD_SIZE[1] / 2, PAD_SIZE[1] / 2),
        cmap=cmap,
        vmin=0,
        vmax=MAX_DEPTH,
        interpolation="nearest",
    )
    axis.set(title="Indentation depth", xlabel="Pad X (m)", ylabel="Pad Y (m)")
    figure.colorbar(image, ax=axis, label="Depth (m)", shrink=0.75)


def draw_rgb(axis: Axes, sample: Sample) -> None:
    """Display the normalized optical output from the same pad sample."""
    axis.imshow(
        sample["rgb"],
        origin="lower",
        extent=(-PAD_SIZE[0] / 2, PAD_SIZE[0] / 2, -PAD_SIZE[1] / 2, PAD_SIZE[1] / 2),
        interpolation="nearest",
    )
    axis.set(title="Tactile RGB", xlabel="Pad X (m)", ylabel="Pad Y (m)")


def draw_sample(
    figure: Figure, samples: list[Sample], index: int, args: argparse.Namespace
) -> None:
    """Render both panels using the same completed physics sample."""
    sample = samples[index]
    figure.clear()
    figure.set_layout_engine("constrained")
    axes = figure.subplots(1, 3 if args.rgb else 2)
    show_scene(axes[0], sample["camera_rgb"])
    draw_depth(figure, axes[1], sample)
    if args.rgb:
        draw_rgb(axes[2], sample)
    figure.supxlabel(
        f"Sample {sample['sample_id']}   ·   t = {sample['time']:.3f} s", fontsize=9
    )


def export(samples: list[Sample], args: argparse.Namespace) -> None:
    """Write the sampled tensors, an English PNG, and optional H.264 video."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        **{key: np.stack([sample[key] for sample in samples]) for key in samples[0]},
    )
    figure = plt.figure(figsize=(10.8 if args.rgb else 7.6, 3.8), dpi=120)
    preview = int(np.argmax([sample["depth"].max() for sample in samples]))
    draw_sample(figure, samples, preview, args)
    figure.savefig(args.output_dir / "preview.png")
    if args.video:
        writer = animation.FFMpegWriter(
            fps=args.fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"]
        )
        with writer.saving(figure, args.output_dir / "preview.mp4", dpi=120):
            for index in range(len(samples)):
                draw_sample(figure, samples, index, args)
                writer.grab_frame()
    plt.close(figure)


def main() -> None:
    """Run the headless demonstration and export all requested views."""
    parser = build_parser()
    args = parser.parse_args()
    if args.frames < 9 or args.fps < 1:
        parser.error("--frames must be at least 9 and --fps must be positive")
    if args.video and not animation.writers.is_available("ffmpeg"):
        parser.error("--video requires ffmpeg on PATH")

    sim = create_manager(args)
    try:
        tool, sensor = create_scene(sim, args)
        samples = collect_samples(sim, tool, sensor, args)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    export(samples, args)


if __name__ == "__main__":
    main()
