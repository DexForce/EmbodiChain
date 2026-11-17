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

"""Visualize completed contact samples in a small drop-and-slide scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    MassPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import SensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import VisualizationCfg

from .._preview import write_preview
from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

DT = 0.01
FRAMES = 60
PAD_WIDTH = 0.30
PAD_DEPTH = 0.04
PAD_TOP = 0.07
CUBE_WIDTH = 0.08
NORMAL_COLOR = "#265ebb"
FRICTION_COLOR = "#bd5729"


def draw_resultant(
    force: Axes,
    validity: Axes,
    times: list[float],
    normal: list[float],
    friction: list[float],
    contact: list[bool],
    index: int,
) -> None:
    """Plot magnitudes only where the corresponding components are valid."""
    for values, label, color in (
        (normal, "Normal", NORMAL_COLOR),
        (friction, "Friction", FRICTION_COLOR),
    ):
        if np.isfinite(values).any():
            force.plot(times, values, label=label, color=color)
        else:
            force.plot([], [], label=f"{label} unavailable", color=color, linestyle=":")
    force.axvline(times[index], color="#555e68", alpha=0.65)
    force.set_ylabel("Force magnitude (N)")
    force.legend(frameon=False, fontsize=8)
    for offset, signal, color in (
        (0, contact, "#50746c"),
        (2, np.isfinite(normal), NORMAL_COLOR),
        (4, np.isfinite(friction), FRICTION_COLOR),
    ):
        validity.step(
            times, offset + 0.7 * np.asarray(signal), where="post", color=color
        )
    validity.axvline(times[index], color="#555e68", alpha=0.65)
    validity.set(
        xlabel="Simulation time (s)",
        ylim=(-0.4, 5.1),
        yticks=[0.35, 2.35, 4.35],
        yticklabels=["Contact", "Normal valid", "Friction valid"],
    )
    for axis in (force, validity):
        axis.set_xlim(times[0], times[-1])
        axis.grid(axis="x", alpha=0.2)


def run_contact(output: Path, backend: str, device: str, video: bool) -> None:
    """Drop and slide a small box using the requested physics provider."""
    physics = (
        DefaultPhysicsCfg(device=device)
        if backend == "default"
        else NewtonPhysicsCfg(
            device=device, num_substeps=6, solver_cfg={"solver_type": "mujoco_warp"}
        )
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(backend="none"),
            physics_cfg=physics,
            physics_dt=DT,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            enable_ground_plane=False,
        )
    )
    times, poses, normal, friction, contact = [], [], [], [], []
    try:
        material = RigidBodyMaterialCfg(static_friction=0.55, dynamic_friction=0.45)
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="pad",
                shape=CubeCfg(size=(PAD_WIDTH, PAD_WIDTH, PAD_DEPTH)),
                body_type="static",
                init_pos=(0, 0, PAD_TOP - PAD_DEPTH / 2),
                attrs=RigidBodyPhysicsCfg(material_props=material),
            )
        )
        cube = sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=(CUBE_WIDTH,) * 3),
                init_pos=(-0.05, 0, 0.21),
                attrs=RigidBodyPhysicsCfg(
                    material_props=material,
                    mass_props=MassPropertiesCfg(mass=0.2),
                ),
            )
        )
        sensor = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "ContactResultant",
                    "uid": "resultant",
                    "asset_uid": "cube",
                }
            )
        )
        camera = scene_camera(
            sim,
            (0.25, -0.44, 0.25),
            (0, 0, 0.10),
            {"pad": (0.13, 0.36, 0.48), "cube": (0.87, 0.50, 0.21)},
        )
        cube.set_velocity(lin_vel=torch.tensor([[0.35, 0, 0]], device=sim.device))
        for _ in range(FRAMES):
            sim.update(step=1, render_final_step=False)
            metadata = sensor.get_sample_metadata()
            if not bool(metadata["valid"][0]):
                raise RuntimeError("Expected the completed contact sample.")
            times.append(float(metadata["time"][0]))
            poses.append(capture_rgb(sim, camera))
            data = sensor.get_data()
            normal.append(
                float(data["normal_force"][0].norm())
                if bool(data["normal_valid"][0])
                else np.nan
            )
            friction.append(
                float(data["friction_force"][0].norm())
                if bool(data["friction_valid"][0])
                else np.nan
            )
            contact.append(bool(data["contact"][0]))
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    active = np.flatnonzero(contact)
    if not len(active):
        raise RuntimeError("The cube did not reach the pad.")
    print(
        f"Measured {len(active)} contact samples. Normal valid: {int(np.isfinite(normal).sum())}, friction valid: {int(np.isfinite(friction).sum())}"
    )

    def draw(figure: Figure, index: int) -> None:
        layout = figure.add_gridspec(
            2, 2, width_ratios=(1, 1.4), height_ratios=(1.3, 1)
        )
        show_scene(figure.add_subplot(layout[:, 0]), poses[index])
        force = figure.add_subplot(layout[0, 1])
        draw_resultant(
            force,
            figure.add_subplot(layout[1, 1]),
            times,
            normal,
            friction,
            contact,
            index,
        )

    still = (
        int(np.nanargmax(friction))
        if np.isfinite(friction).any() and np.nanmax(friction) > 0
        else int(active[0])
    )
    write_preview(output, times, draw, still=still, video=video)


def main() -> None:
    """Parse backend and headless export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("newton", "default"), default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.backend == "newton" and not args.device.startswith("cuda"):
        parser.error("The Newton MuJoCo Warp example requires --device cuda:0.")
    run_contact(args.output, args.backend, args.device, not args.no_video)


if __name__ == "__main__":
    main()
