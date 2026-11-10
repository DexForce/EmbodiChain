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

"""Display adapters that never advance the workspace demo's physics clock."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, LightCfg, RenderCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.visualization import (
    FrameOverlay,
    PointCloudOverlay,
    SceneOverlays,
    TargetOverlay,
    VisualizationCfg,
    ViserServerCfg,
)

from ._robot_workspace_scene import CONTROL_DT, GOAL, NEGATIVE

__all__: list[str] = []


class _Display:
    def __init__(self, view: str, output: Path, port: int = 8080) -> None:
        self.view, self.output, self.port = view, output, port
        self.camera = None
        self.writer = None
        self.steps = 0
        self.last_video_step = -1
        self.video_frames = 0
        self.video_events: list[dict[str, Any]] = []
        self.pool: tuple = ()
        self.grasp = None
        self.workspace_points = np.empty((0, 3))
        self.workspace_colors = np.empty((0, 3), dtype=np.uint8)
        self.native_cloud = None
        self.status: dict[int, str] = {}
        self.keyframes: list[str] = []
        self.stage = "setup"

    def config(self) -> SimulationManagerCfg:
        return SimulationManagerCfg(
            num_envs=1,
            device="cpu",
            headless=self.view != "window",
            physics_dt=0.01,
            physics_cfg=DefaultPhysicsCfg(),
            # Hybrid supports visible pixel-sized points in window/camera RGB.
            render_cfg=RenderCfg(
                renderer="no-render" if self.view in ("viser", "none") else "hybrid"
            ),
            visualization=VisualizationCfg(
                backend="viser" if self.view == "viser" else "none",
                allow_commands=False,
                viser_server=ViserServerCfg(port=self.port),
            ),
        )

    def attach(self, sim: Any) -> None:
        if self.view in ("window", "offscreen"):
            sim.add_light(
                LightCfg(
                    uid="workspace_light",
                    light_type="sun",
                    direction=(0.0, 0.0, -1.0),
                    intensity=5.0,
                )
            )
        if self.view == "offscreen":
            self.camera = sim.add_sensor(self.camera_config())

    def camera_config(self) -> CameraCfg:
        return CameraCfg(
            uid="workspace_camera",
            width=640,
            height=480,
            intrinsics=(600.0, 600.0, 320.0, 240.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(1.8, -2.6, 2.4),
                target=(0.0, 0.0, 0.85),
                up=(0.0, 0.0, 1.0),
            ),
        )

    def start(self, sim: Any) -> None:
        if self.view == "window":
            if not sim.open_window():
                raise RuntimeError("Native window could not be opened")
        if self.view == "offscreen":
            import imageio.v2 as imageio

            (self.output / "frames").mkdir(exist_ok=True)
            self.camera.reset(env_ids=[0])
            self.writer = imageio.get_writer(
                str(self.output / "preview.mp4"), fps=1.0 / CONTROL_DT
            )

    def candidates(
        self, candidates: tuple, grasp: Any, info: dict[str, Any], *, sim: Any
    ) -> None:
        self.pool, self.grasp = candidates, grasp.detach().cpu().numpy()
        with np.load(info["reachability"]["cache"], allow_pickle=False) as cache:
            self.workspace_points = cache["points"].copy()
            self.workspace_colors = cache["colors"].copy()
        if self.view in ("window", "offscreen"):
            self.native_cloud = sim.visualize_point_cloud(
                points=self.workspace_points,
                colors=self.workspace_colors,
                point_size=4.0,
                name="fixed_grasp_workspace",
            )

    def _overlays(self) -> SceneOverlays:
        points, colors, frames = [], [], []
        palette = {
            "pending": (255, 190, 40),
            "passed": (40, 210, 100),
            "failed": (240, 60, 60),
        }
        for index, candidate in enumerate(self.pool):
            pose = np.asarray(candidate.variant.pose_changes[0].arena_pose)
            points.append(pose[:3, 3])
            colors.append(palette[self.status.get(index, "pending")])
            tcp = pose @ self.grasp
            xyzw = Rotation.from_matrix(tcp[:3, :3]).as_quat()
            frames.append(
                FrameOverlay(
                    overlay_id=f"tcp_{index}",
                    position=tcp[:3, 3],
                    wxyz=xyzw[[3, 0, 1, 2]],
                )
            )
        points.append(np.asarray(NEGATIVE))
        colors.append(palette[self.status.get(len(self.pool), "pending")])
        clouds = []
        if len(self.workspace_points):
            clouds.append(
                PointCloudOverlay(
                    overlay_id="workspace_reachability",
                    points=self.workspace_points,
                    colors=self.workspace_colors,
                    point_size=0.012,
                )
            )
        clouds.append(
            PointCloudOverlay(
                overlay_id="workspace_candidates",
                points=np.asarray(points),
                colors=np.asarray(colors, dtype=np.uint8),
                point_size=0.015,
            )
        )
        return SceneOverlays(
            frames=tuple(frames),
            point_clouds=tuple(clouds),
            targets=(
                TargetOverlay(
                    overlay_id="placement_goal",
                    position=np.asarray(GOAL),
                    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                ),
            ),
        )

    def frame(self, sim: Any, stage: str, *, force: bool = False) -> None:
        self.stage = stage
        if force:
            print(f"[workspace-demo] {stage}", flush=True)
            if stage.startswith("candidate_") and "settled" in stage:
                self.status[int(stage.split("_")[1])] = "passed"
            elif stage.startswith("candidate_") and "rejected" in stage:
                self.status[int(stage.split("_")[1])] = "failed"
            elif stage == "negative_rejected":
                self.status[len(self.pool)] = "failed"
        if self.view == "viser" and self.pool:
            sim.set_visualization_overlays(self._overlays())
        if self.view != "none":
            with sim.render_frame(
                force_visualization=force, capture_camera_images=False
            ):
                if self.camera is not None:
                    self.camera.update()
                    rgb = (
                        self.camera.get_data()["color"][0, ..., :3]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(
                            rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255
                        ).astype(np.uint8)
                    if self.steps != self.last_video_step:
                        self.writer.append_data(rgb)
                        self.last_video_step = self.steps
                        self.video_frames += 1
                    if force:
                        import imageio.v2 as imageio

                        path = (
                            self.output
                            / "frames"
                            / f"{len(self.keyframes):03d}_{stage}.png"
                        )
                        imageio.imwrite(path, rgb)
                        self.keyframes.append(str(path.relative_to(self.output)))
                        self.video_events.append(
                            {
                                "after_frame": self.video_frames,
                                "stage": stage,
                                "keyframe": str(path.relative_to(self.output)),
                            }
                        )
        if force and self.view in ("viser", "window"):
            # Hold a frame for inspection without integrating physics.
            time.sleep(1.0)

    def advance(self, sim: Any, count: int) -> None:
        for _ in range(count):
            sim.update(step=1)
            self.steps += 1
            if self.steps % 4 == 0:
                self.frame(sim, "settling")

    def control_step(self, sim: Any) -> None:
        self.steps += 4
        self.frame(sim, "execution")
        if self.view in ("viser", "window"):
            time.sleep(CONTROL_DT)

    def close(self) -> None:
        self.native_cloud = None
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()
