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

"""Record registered state sensors in a headless, self-contained simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    DefaultPhysicsCfg,
    JointDrivePropertiesCfg,
)
from embodichain.lab.sim.sensors import SensorCfg
from embodichain.lab.visualization import VisualizationCfg

from .._preview import write_preview
from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

DT = 0.02
FRAMES = 48
AXIS_COLORS = ("#c0392b", "#26704f", "#2b5bb0")


def run_state(output: Path, device: str, video: bool) -> None:
    """Move a two-arena hinge and visualize arena 0 from batched outputs."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(backend="none"),
            physics_cfg=DefaultPhysicsCfg(device=device, gravity=(0, 0, 0)),
            physics_dt=DT,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
            num_envs=2,
            enable_ground_plane=False,
        )
    )
    times, angles, speeds, frames = [], [], [], []
    images = []
    try:
        hinge = sim.add_articulation(
            ArticulationCfg(
                uid="arm",
                fpath=str(Path(__file__).with_name("hinge.urdf")),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
            )
        )
        encoder = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "JointEncoder",
                    "uid": "encoder",
                    "asset_uid": "arm",
                    "joint_names": ["hinge"],
                }
            )
        )
        transform = sim.add_sensor(
            SensorCfg.from_dict(
                {
                    "sensor_type": "FrameTransform",
                    "uid": "frames",
                    "source_uid": "arm",
                    "source_link": "base",
                    "target_uid": "arm",
                    "target_link": "arm",
                    "target_offset": {"pos": [0.24, 0, 0]},
                }
            )
        )
        camera = scene_camera(sim, (0.41, -0.57, 0.33), (0.10, 0, 0.045))
        speed_command = torch.tensor([[1.25], [-1.25]], device=sim.device)
        for step in range(FRAMES):
            if step in (0, FRAMES // 2):
                hinge.set_qvel(
                    speed_command if step == 0 else -speed_command, target=False
                )
            sim.update(step=1, render_final_step=False)
            images.append(capture_rgb(sim, camera))
            sample = encoder.get_sample_metadata()
            poses = transform.get_sample_metadata()
            if not sample["valid"].all() or not torch.equal(
                sample["physics_step"], poses["physics_step"]
            ):
                raise RuntimeError(
                    "Expected both sensors to sample the completed step."
                )
            times.append(float(sample["time"][0]))
            angles.append(encoder.get_data()["qpos"].cpu().numpy().copy()[:, 0])
            speeds.append(encoder.get_data()["qvel"].cpu().numpy().copy()[:, 0])
            frames.append(transform.get_data()["transform"][0].cpu().numpy().copy())
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    if np.ptp(np.asarray(angles)[:, 0]) < 0.2:
        raise RuntimeError("The simulated hinge did not advance.")

    def draw(figure: Figure, index: int) -> None:
        grid = figure.add_gridspec(2, 2, width_ratios=(1.1, 1))
        show_scene(figure.add_subplot(grid[0, 0]), images[index])
        scene = figure.add_subplot(grid[1, 0], projection="3d")
        angle = figure.add_subplot(grid[0, 1])
        speed = figure.add_subplot(grid[1, 1], sharex=angle)

        frame = frames[index]
        tip = frame[:3, 3]
        for i, color in enumerate(AXIS_COLORS):
            direction = np.eye(3)[i] * 0.08
            scene.plot(
                *np.stack((np.zeros(3), direction)).T, color=color, linestyle="--"
            )
            scene.plot(*np.stack((tip, tip + 0.08 * frame[:3, i])).T, color=color)
        scene.text(0, 0, -0.035, "Base")
        scene.text(*tip, "Tip")
        scene.set(
            xlim=(-0.10, 0.34),
            ylim=(-0.22, 0.22),
            zlim=(-0.08, 0.20),
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )
        scene.set_box_aspect((0.44, 0.44, 0.28))
        scene.view_init(elev=25, azim=-66)

        for axis, values, label in (
            (angle, angles, "Joint angle (rad)"),
            (speed, speeds, "Joint speed (rad/s)"),
        ):
            axis.plot(times, np.asarray(values)[:, 0], label="Arena 0", color="#2b5bb0")
            axis.plot(
                times,
                np.asarray(values)[:, 1],
                label="Arena 1",
                color="#78859b",
                linestyle="--",
            )
            axis.axvline(times[index], color="#59616c", linewidth=0.9, alpha=0.7)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        angle.legend(frameon=False, loc="upper right")
        speed.set_xlabel("Simulation time (s)")

    write_preview(
        output, times, draw, still=int(np.argmax(np.asarray(angles)[:, 0])), video=video
    )
    print(f"Saved {len(times)} completed samples from two arenas to {output}")


def main() -> None:
    """Parse export options and run the selected simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_state(args.output, args.device, not args.no_video)


if __name__ == "__main__":
    main()
