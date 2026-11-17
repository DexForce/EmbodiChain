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

"""Measure a known eccentric gravity load at a named joint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, NewtonPhysicsCfg, RenderCfg
from embodichain.lab.sim.sensors import SensorCfg

from .._preview import write_preview
from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

AXIS_COLORS = ("#b3413a", "#34744d", "#2e59a4")
EXPECTED_FORCE = np.array([0.0, 0.0, 9.81])
EXPECTED_TORQUE = np.array([0.0, -0.981, 0.0])
LOAD_ARM = np.array([0.10, 0.0, 0.0])


def run_wrench(output: Path, device: str, video: bool) -> None:
    """Show the reaction and measurement point from a completed solver step."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            physics_dt=0.01,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            enable_ground_plane=False,
            physics_cfg=NewtonPhysicsCfg(
                device=device,
                num_substeps=3,
                enable_joint_wrenches=True,
                solver_cfg={"solver_type": "mujoco_warp", "rne_mode": "last_substep"},
            ),
        )
    )
    times, valid, forces, torques, frames = [], [], [], [], []
    images = []
    try:
        sim.add_articulation(
            ArticulationCfg(
                uid="chain",
                fpath=str(Path(__file__).with_name("wrench.urdf")),
                build_pk_chain=False,
            )
        )
        sensor = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "JointWrench",
                    "uid": "reaction",
                    "asset_uid": "chain",
                    "joint_name": "load_cell",
                }
            )
        )
        pose = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "FrameTransform",
                    "uid": "wrench_frame",
                    "target_uid": "chain",
                    "target_link": "payload",
                }
            )
        )

        camera = scene_camera(sim, (0.36, -0.58, 0.82), (0.04, 0, 0.54))

        def record() -> None:
            images.append(capture_rgb(sim, camera))
            data = sensor.get_data()
            meta = sensor.get_sample_metadata()
            values_valid = bool(meta["valid"][0] and data["wrench_valid"][0])
            times.append(sim.simulation_time)
            valid.append(values_valid)
            forces.append(
                data["force"][0].cpu().numpy().copy()
                if values_valid
                else np.full(3, np.nan)
            )
            torques.append(
                data["torque"][0].cpu().numpy().copy()
                if values_valid
                else np.full(3, np.nan)
            )
            frames.append(pose.get_data()["transform"][0].cpu().numpy().copy())

        record()
        for step in range(42):
            if step == 21:
                sim.reset_objects_state()
                record()
            sim.update(step=1, render_final_step=False)
            record()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    if not (
        np.allclose(forces[-1], EXPECTED_FORCE, atol=0.03)
        and np.allclose(torques[-1], EXPECTED_TORQUE, atol=0.01)
    ):
        raise RuntimeError(f"Unexpected gravity reaction: {forces[-1]}, {torques[-1]}")

    forces_array = np.asarray(forces)
    torques_array = np.asarray(torques)

    def draw(figure: Figure, index: int) -> None:
        layout = figure.add_gridspec(
            3, 2, height_ratios=(1, 1, 0.5), width_ratios=(1.05, 1)
        )
        show_scene(figure.add_subplot(layout[0, 0]), images[index])
        scene = figure.add_subplot(layout[1:, 0], projection="3d")
        origin = frames[index][:3, 3]
        rotation = frames[index][:3, :3]
        if not valid[index]:
            # The frame sensor also invalidates its buffer on the explicit reset.
            valid_index = next((i for i in range(index, -1, -1) if valid[i]), 1)
            origin = frames[valid_index][:3, 3]
            rotation = frames[valid_index][:3, :3]
        payload_com = origin + rotation @ LOAD_ARM
        for i, color in enumerate(AXIS_COLORS):
            scene.quiver(*origin, *rotation[:, i], length=0.045, color=color, alpha=0.6)
        scene.scatter(*origin, color="#344255", s=15)
        scene.scatter(*payload_com, color="#344255", s=25)
        scene.text(*payload_com, "  Payload COM")
        scene.text(*(origin - [0.065, 0, 0.015]), "load_cell")
        if valid[index]:
            scene.quiver(
                *origin,
                *(rotation @ forces_array[index] * 0.012),
                color="#2d6db9",
                label="Force",
            )
            scene.quiver(
                *origin,
                *(rotation @ torques_array[index] * 0.11),
                color="#a74583",
                label="Torque",
            )
        else:
            scene.plot([], [], [], color="#59636e", label="Wrench unavailable")
        scene.legend(frameon=False, loc="upper left")
        scene.set(
            xlim=(-0.12, 0.22),
            ylim=(-0.15, 0.15),
            zlim=(0.43, 0.66),
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        scene.set_box_aspect((0.34, 0.30, 0.23))
        scene.view_init(elev=22, azim=-59)

        force = figure.add_subplot(layout[0, 1])
        torque = figure.add_subplot(layout[1, 1], sharex=force)
        validity = figure.add_subplot(layout[2, 1], sharex=force)
        for axis, array, label, ref in (
            (force, forces_array, "Force (N)", EXPECTED_FORCE[2]),
            (torque, torques_array, "Torque (N·m)", EXPECTED_TORQUE[1]),
        ):
            for i, color in enumerate(AXIS_COLORS):
                axis.plot(times, array[:, i], label="XYZ"[i], color=color)
            axis.axhline(
                ref,
                color="#4a5460",
                linewidth=1,
                linestyle=":",
                label="Gravity reference",
            )
            axis.set_ylabel(label)
        force.legend(frameon=False, fontsize=8, loc="center right")
        validity.step(times, valid, color="#2e6964", where="post")
        validity.set(
            xlabel="Simulation time (s)",
            ylabel="Wrench",
            yticks=[0, 1],
            yticklabels=["Invalid", "Valid"],
            ylim=(-0.22, 1.22),
        )
        for axis in (force, torque, validity):
            axis.axvline(times[index], color="#9ea7b1", linewidth=0.9)
            axis.grid(alpha=0.15)

    write_preview(output, times, draw, still=20, video=video)
    print(f"Last valid gravity reaction: {forces[-1]} N, {torques[-1]} N·m")


def main() -> None:
    """Parse CUDA and headless export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("JointWrench requires CUDA Newton MuJoCo Warp with RNE.")
    run_wrench(args.output, args.device, not args.no_video)


if __name__ == "__main__":
    main()
