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

"""Compare PVA and IMU readings for rest, free fall, and an offset spin."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    DefaultRigidBodyPropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import SensorCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import VisualizationCfg

from .._preview import write_preview
from .._scene_camera import capture_rgb, scene_camera, show_scene, SCENE_RENDER

DT = 0.015
FRAMES = 60
OFFSET = 0.18
OMEGA = 6.0
SCENARIOS = (
    ("rest", "Rest", "#426c59", (0, 0, 0.05)),
    ("fall", "Free fall", "#2b60ac", (0.6, 0, 3.0)),
    ("spin", "Offset spin", "#c76c33", (1.2, 0, 3.0)),
)


def run_inertial(output: Path, device: str, video: bool) -> None:
    """Read the registered sensors at completed physics boundaries."""
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(backend="none"),
            physics_cfg=DefaultPhysicsCfg(device=device),
            physics_dt=DT,
            render_cfg=SCENE_RENDER,
            enable_entity_gizmo=False,
            width=720,
            height=540,
        )
    )
    times, images = [], []
    reset_time = DT * (FRAMES // 2)
    readings = {uid: [] for uid, _, _, _ in SCENARIOS}
    try:
        bodies = {
            uid: sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    shape=CubeCfg(size=(0.10,) * 3),
                    init_pos=position,
                    attrs=RigidBodyPhysicsCfg(
                        rigid_props=DefaultRigidBodyPropertiesCfg(
                            linear_damping=0.0, angular_damping=0.0
                        )
                    ),
                )
            )
            for uid, _, _, position in SCENARIOS
        }
        sensors = {
            uid: tuple(
                sim.add_sensor(
                    SensorCfg.from_dict(
                        {
                            "sensor_type": kind,
                            "uid": f"{uid}_{kind.lower()}",
                            "asset_uid": uid,
                            "offset": {"pos": [OFFSET if uid == "spin" else 0, 0, 0]},
                        }
                    )
                )
                for kind in ("PVA", "IMU")
            )
            for uid, _, _, _ in SCENARIOS
        }

        camera = scene_camera(
            sim,
            (4.7, -5.0, 2.85),
            (0.58, 0.0, 1.48),
            {
                "rest": (0.23, 0.58, 0.40),
                "fall": (0.17, 0.40, 0.80),
                "spin": (0.91, 0.48, 0.20),
            },
        )

        def record() -> None:
            images.append(capture_rgb(sim, camera))
            times.append(sim.simulation_time)
            for uid, (pva, imu) in sensors.items():
                pva_data, imu_data = pva.get_data(), imu.get_data()
                pva_meta, imu_meta = (
                    pva.get_sample_metadata(),
                    imu.get_sample_metadata(),
                )
                if not torch.equal(pva_meta["physics_step"], imu_meta["physics_step"]):
                    raise RuntimeError("Expected PVA and IMU on the same physics step.")
                valid = bool(imu_meta["valid"][0] and imu_data["acceleration_valid"][0])
                readings[uid].append(
                    (
                        (
                            float(imu_data["angular_velocity"][0, 2])
                            if bool(imu_meta["valid"][0])
                            else np.nan
                        ),
                        (
                            float(pva_data["linear_acceleration"][0, 2])
                            if valid
                            else np.nan
                        ),
                        (
                            float(
                                imu_data["specific_force"][0, 0 if uid == "spin" else 2]
                            )
                            if valid
                            else np.nan
                        ),
                        valid,
                    )
                )

        spin = torch.tensor([[0, 0, OMEGA]], device=sim.device)
        bodies["spin"].set_velocity(ang_vel=spin)
        record()
        for step in range(FRAMES):
            if step == FRAMES // 2:
                sim.reset_objects_state()
                bodies["spin"].set_velocity(ang_vel=spin)
                record()
            sim.update(step=1, render_final_step=False)
            record()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    traces = {uid: np.asarray(samples) for uid, samples in readings.items()}
    if not (
        np.nanmedian(traces["spin"][:, 2]) < -1
        and np.nanmedian(traces["fall"][:, 1]) < -8
    ):
        raise RuntimeError(
            "The simulated spin and free fall did not produce the expected signals."
        )

    def draw(figure: Figure, index: int) -> None:
        layout = figure.add_gridspec(2, 3, width_ratios=(1.2, 1, 1))
        show_scene(figure.add_subplot(layout[:, 0]), images[index])
        angular = figure.add_subplot(layout[0, 1])
        acceleration = figure.add_subplot(layout[0, 2], sharex=angular)
        specific = figure.add_subplot(layout[1, 1], sharex=angular)
        validity = figure.add_subplot(layout[1, 2], sharex=angular)
        axes = np.asarray([angular, acceleration, specific, validity])
        for row, (uid, name, color, _) in enumerate(SCENARIOS):
            data = traces[uid]
            angular.plot(times, data[:, 0], color=color, label=name)
            acceleration.plot(times, data[:, 1], color=color)
            specific.plot(
                times,
                data[:, 2],
                color=color,
                label=f"{name} {'X' if uid == 'spin' else 'Z'}",
            )
            validity.step(times, row + 0.45 * data[:, 3], where="post", color=color)
            invalid = data[:, 3] == 0
            validity.scatter(
                np.asarray(times)[invalid],
                np.full(invalid.sum(), row),
                s=10,
                color=color,
            )
        angular.set_ylabel("Angular velocity Z (rad/s)")
        angular.legend(frameon=False, fontsize=8, loc="center right")
        acceleration.set_ylabel("PVA acceleration Z (m/s²)")
        specific.set_ylabel("IMU specific force (m/s²)")
        specific.legend(frameon=False, fontsize=8, loc="center right")
        validity.set(
            yticks=[0.22, 1.22, 2.22],
            yticklabels=["Rest", "Free fall", "Offset spin"],
            ylim=(-0.25, 2.8),
        )
        validity.set_ylabel("Acceleration validity")
        for axis in axes.ravel():
            axis.axvline(reset_time, color="#84909f", linestyle=":", linewidth=1)
            axis.axvline(times[index], color="#444c57", linewidth=0.8, alpha=0.7)
            axis.grid(alpha=0.18)
        specific.set_xlabel("Simulation time (s)")
        validity.set_xlabel("Simulation time (s)")
        validity.text(
            reset_time, 2.68, "Reset", ha="center", fontsize=8, color="#4e5969"
        )

    write_preview(output, times, draw, still=FRAMES // 2, video=video)
    print(f"Captured {FRAMES} completed steps with startup and reset validity samples.")


def main() -> None:
    """Parse the output and simulation device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_inertial(args.output, args.device, not args.no_video)


if __name__ == "__main__":
    main()
