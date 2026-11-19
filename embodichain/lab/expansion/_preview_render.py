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

"""Bounded native mesh replay without constructing a task or advancing physics."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any
import io
import math
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from ._preview_data import PreviewData, PreviewState, pose_matrix

__all__: list[str] = []


def prepare_geometry(data: PreviewData) -> tuple[list[str], np.ndarray]:
    import torch
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(data.asset.read_bytes())
    names = chain.get_link_names()
    joints = chain.get_joint_parameter_names()
    if data.eef_link not in names:
        raise ValueError(f"Unknown end-effector link: {data.eef_link}")
    total = (
        sum(len(c.measured.timestamps) for c in data.all_clips) * len(names) * 16 * 4
    )
    if total > 256 * 1024 * 1024:
        raise ValueError(
            "Preview FK exceeds the 256 MiB budget; select smaller result directories"
        )
    for c in data.all_clips:
        m = c.measured
        if set(joints) != set(m.joint_names):
            raise ValueError(
                "Recorded joints do not match the supplied URDF (including mimic joints)"
            )
        order = [m.joint_names.index(n) for n in joints]
        c.links = np.empty((len(m.timestamps), len(names), 4, 4), np.float32)
        for start in range(0, len(m.timestamps), 256):
            stop = min(len(m.timestamps), start + 256)
            q = torch.tensor(m.qpos[start:stop][:, order], dtype=torch.float32)
            with torch.no_grad():
                fk = chain.forward_kinematics(q)
            roots = np.stack([pose_matrix(p) for p in m.root_poses[start:stop]])
            c.links[start:stop] = roots[:, None] @ np.stack(
                [fk[n].get_matrix().numpy() for n in names], axis=1
            )
        c.tips = c.links[:, names.index(data.eef_link), :3, 3]
        if not np.isfinite(c.links).all():
            raise ValueError("FK produced nonfinite poses")
    points = np.concatenate([c.links[0, :, :3, 3] for c in data.all_clips])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    return names, center


class NativeRenderer:
    def __init__(
        self,
        data: PreviewData,
        sim: Any,
        publisher: Any,
        names: list[str],
        center: np.ndarray,
        models: dict[str, Any] | None = None,
        geometry: dict[str, Any] | None = None,
    ) -> None:
        from dexsim import spawn
        from embodichain.lab.visualization.markers import (
            RobotMarkers,
            RobotMarkersCfg,
            RobotMarkersAppearanceCfg,
            VisualizationMarkers,
            VisualizationMarkersCfg,
            MarkerPrototypeCfg,
        )

        self.data, self.sim, self.publisher, self.center = data, sim, publisher, center
        self.link_names = names
        scene = sim.spawn_result
        self.step = sim.physics_step_index
        self.previous = None
        self.models = models or {}
        self.geometry = geometry or {}
        self.backgrounds = {
            uid
            for uid, spec in (data.object_specs or {}).items()
            if spec.get("body_type") in {"static", "kinematic"}
            and all(
                np.allclose(
                    c.measured.object_poses[uid],
                    data.solid_clip.measured.object_poses[uid][0],
                    rtol=0,
                    atol=1e-7,
                )
                for c in data.all_clips
            )
        }
        self.moving_objects = [uid for uid in data.boxes if uid not in self.backgrounds]
        for uid, model in self.models.items():
            model.set_local_pose(data.solid_clip.measured.object_poses[uid][0:1])
            model.set_visible(True)
        palette = [
            (0.18, 0.58, 1),
            (0.15, 0.9, 0.5),
            (0.9, 0.55, 0.2),
            (0.7, 0.4, 0.95),
        ]
        variants = sorted({data.clips[i].variant for i in data.ghost_indices})
        self.colors = {v: palette[i % len(palette)] for i, v in enumerate(variants)}
        for name, pos, intensity in [
            ("preview_key", [-1, -2, 3], 35),
            ("preview_fill", [1, 1, 2], 18),
        ]:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = np.asarray(pos) + center
            scene.add_light(
                spawn.LightDesc(
                    name=name, light_type="POINT", pose=pose, intensity=intensity
                )
            )
        self.ghosts = {}

        def markers(name: str, shapes: dict) -> Any:
            return VisualizationMarkers(
                VisualizationMarkersCfg(prim_path=name, markers=shapes), publisher
            )

        for i in data.ghost_indices:
            c = data.clips[i]
            color = self.colors[c.variant]
            robot = RobotMarkers(
                RobotMarkersCfg(
                    name=f"preview_robot_{i}",
                    asset_path=str(data.asset),
                    appearance=RobotMarkersAppearanceCfg(color=color, opacity=0.18),
                ),
                publisher,
            )
            link_order = [names.index(n) for n in robot.link_names]
            objects = (
                markers(
                    f"preview_objects_{i}",
                    {
                        uid: self._object_prototype(uid, color=(*color, 0.2))
                        for uid in self.moving_objects
                    },
                )
                if self.moving_objects
                else None
            )
            self.ghosts[i] = (robot, link_order, objects)
            robot.visualize_link_poses(c.links[0, link_order][None])
            robot.set_visibility(False)
            if objects:
                p = np.array(
                    [c.measured.object_poses[uid][0] for uid in self.moving_objects]
                )
                objects.visualize(p[:, :3], p[:, 3:], marker_indices=np.arange(len(p)))
                objects.set_visibility(False)
        self.solid_robot = RobotMarkers(
            RobotMarkersCfg(
                name="solid_model",
                asset_path=str(data.asset),
                appearance=RobotMarkersAppearanceCfg(mode="asset", opacity=1),
            ),
            publisher,
        )
        self.solid_order = [names.index(n) for n in self.solid_robot.link_names]
        self.solid_robot.visualize_link_poses(
            data.solid_clip.links[0, self.solid_order][None]
        )
        self.solid_objects = (
            markers(
                "solid_objects",
                {
                    uid: self._object_prototype(
                        uid, color=(0.9, 0.58, 0.2, 1), scene_lighting=True, unlit=False
                    )
                    for uid in self.moving_objects
                },
            )
            if self.moving_objects and not self.models
            else None
        )
        path_shapes = {
            "solid": MarkerPrototypeCfg(shape="cylinder", color=(1, 0.9, 0.7, 1))
        }
        self.variant_ids = {v: i for i, v in enumerate(variants, start=1)}
        for variant, index in self.variant_ids.items():
            path_shapes[str(index)] = MarkerPrototypeCfg(
                shape="cylinder", color=(*self.colors[variant], 0.8)
            )
        self.paths = markers("measured_paths", path_shapes)
        self.targets = markers(
            "recorded_grasp_targets",
            {"target": MarkerPrototypeCfg(shape="frame", scale=(0.065, 0.065, 0.065))},
        )
        self.positions = markers(
            "initial_positions",
            {
                "point": MarkerPrototypeCfg(
                    shape="sphere", scale=(0.014, 0.014, 0.014), color=(1, 0.7, 0.1, 1)
                )
            },
        )
        self.planned = markers(
            "planned_parameter_positions",
            {
                "confirmed": MarkerPrototypeCfg(
                    shape="sphere",
                    scale=(0.018, 0.018, 0.018),
                    color=(0.3, 0.8, 0.4, 0.4),
                ),
                "unqualified": MarkerPrototypeCfg(
                    shape="sphere",
                    scale=(0.022, 0.022, 0.022),
                    color=(1, 0.25, 0.15, 0.65),
                ),
            },
        )
        if data.coverage:
            points, types = [], []
            for record in data.coverage["records"]:
                for point in record.get("requested_positions", {}).values():
                    points.append(point)
                    types.append(0 if record["status"] == "confirmed" else 1)
            self.planned.visualize(
                np.asarray(points).reshape(-1, 3),
                marker_indices=np.asarray(types, dtype=int),
            )
        self.camera = scene.add_camera(
            spawn.CameraDesc(name="expansion_preview_camera", resolution=(1100, 760))
        )
        self.camera.open_camera()
        self.camera.set_visual_overlay_enabled(True)
        publisher.publish(preload=True)

    def _object_prototype(self, uid: str, **appearance: Any) -> Any:
        from embodichain.lab.visualization.markers import MarkerPrototypeCfg

        if uid in self.geometry:
            vertices, faces = self.geometry[uid]
            return MarkerPrototypeCfg(
                shape="mesh", vertices=vertices, faces=faces, **appearance
            )
        return MarkerPrototypeCfg(shape="box", scale=self.data.boxes[uid], **appearance)

    def draw(self, state: PreviewState) -> bytes:
        v = state.values
        solid = self.data.solid_clip
        visible = state.visible_ghosts()
        if tuple(visible) != self.previous:
            displayed = [solid, *(self.data.clips[i] for i in visible)]
            positions = []
            rotations = []
            scales = []
            types = []
            for c in displayed:
                path_type = 0 if c is solid else self.variant_ids[c.variant]
                points = c.tips[
                    np.unique(np.r_[np.arange(0, len(c.tips), 4), len(c.tips) - 1])
                ]
                for a, b in zip(points[:-1], points[1:]):
                    delta = b - a
                    length = np.linalg.norm(delta)
                    if length < 1e-6:
                        continue
                    z = delta / length
                    axis = np.cross([0, 0, 1], z)
                    norm = np.linalg.norm(axis)
                    rotation = (
                        Rotation.from_rotvec(
                            axis / norm * np.arccos(np.clip(z[2], -1, 1))
                        )
                        if norm > 1e-8
                        else Rotation.from_rotvec([np.pi if z[2] < 0 else 0, 0, 0])
                    )
                    positions.append((a + b) / 2)
                    rotations.append(rotation.as_quat())
                    scales.append([0.002, 0.002, length])
                    types.append(path_type)
            self.paths.visualize(
                np.array(positions).reshape(-1, 3),
                np.array(rotations).reshape(-1, 4),
                np.array(scales).reshape(-1, 3),
                np.asarray(types, dtype=int),
            )
            targets = np.concatenate([c.targets for c in displayed])
            self.targets.visualize(
                targets[:, :3, 3],
                (
                    Rotation.from_matrix(targets[:, :3, :3]).as_quat()
                    if len(targets)
                    else np.empty((0, 4))
                ),
            )
            initial = [
                p[0, :3]
                for c in displayed
                for uid, p in c.measured.object_poses.items()
                if uid not in self.backgrounds
            ]
            self.positions.visualize(np.asarray(initial).reshape(-1, 3))
            self.previous = tuple(visible)
        f = state.frame_at(solid)
        self.solid_robot.visualize_link_poses(solid.links[f, self.solid_order][None])
        for uid in self.moving_objects:
            if uid in self.models:
                self.models[uid].set_local_pose(
                    solid.measured.object_poses[uid][f : f + 1]
                )
        if self.solid_objects:
            p = np.array(
                [solid.measured.object_poses[uid][f] for uid in self.moving_objects]
            )
            self.solid_objects.visualize(
                p[:, :3], p[:, 3:], marker_indices=np.arange(len(p))
            )
        for i, (robot, order, objects) in self.ghosts.items():
            show = i in visible
            robot.set_visibility(show)
            if objects:
                objects.set_visibility(show)
            if show:
                c = self.data.clips[i]
                f = state.frame_at(c)
                robot.visualize_link_poses(c.links[f, order][None])
                if objects:
                    p = np.array(
                        [c.measured.object_poses[uid][f] for uid in self.moving_objects]
                    )
                    objects.visualize(
                        p[:, :3], p[:, 3:], marker_indices=np.arange(len(p))
                    )
        self.planned.set_visibility(v["positions"] and v["space"] == "all")
        self.paths.set_visibility(v["paths"])
        self.targets.set_visibility(v["targets"])
        self.positions.set_visibility(v["positions"])
        self.publisher.publish()
        yaw = v["yaw"]
        e = v["elevation"]
        d = v["distance"]
        eye = self.center + d * np.array(
            [math.cos(e) * math.cos(yaw), math.cos(e) * math.sin(yaw), math.sin(e)]
        )
        self.camera.look_at(eye.tolist(), self.center.tolist(), [0, 0, 1])
        with self.sim.render_frame(capture_camera_images=False):
            self.sim.sync_render_state()
            self.camera.render()
        if self.sim.physics_step_index != self.step:
            raise RuntimeError("Preview unexpectedly advanced physics")
        image = Image.fromarray(
            np.ascontiguousarray(self.camera.get_rgb_map()[..., :3], dtype=np.uint8)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        self.stats = {
            "physics_steps": self.sim.physics_step_index - self.step,
            "ghost_count": len(visible),
            "path_segments": self.paths.count if v["paths"] else 0,
            "grasp_targets": self.targets.count if v["targets"] else 0,
            "scene_objects": len(self.data.boxes),
            "fixed_backgrounds": len(self.backgrounds),
        }
        return buffer.getvalue()


@contextmanager
def native_renderer(data: PreviewData, gpu: int = 0) -> Iterator[NativeRenderer]:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, DenoisingCfg
    from embodichain.lab.visualization.markers import MarkerFramePublisher

    torch.set_num_threads(1)
    names, center = prepare_geometry(data)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            gpu_id=gpu,
            num_envs=1,
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer="hybrid",
                spp=4,
                denoising=DenoisingCfg(window="off", offscreen="optix"),
            ),
        )
    )
    publisher = None
    try:
        sim.enable_physics(False)
        models = {}
        if data.object_specs is not None:
            from embodichain.lab.sim.cfg import VisualObjectCfg

            for index, (uid, spec) in enumerate(data.object_specs.items()):
                visual = {
                    key: value
                    for key, value in spec.items()
                    if key in {"shape", "source", "body_scale"} and value is not None
                }
                visual.update(uid=f"expansion_preview_asset_{index}", visible=False)
                models[uid] = sim.add_visual_object(VisualObjectCfg.from_dict(visual))
        sim.prepare()
        sim.set_ground_plane_visibility(data.ground_visible)
        geometry = {}
        for uid, obj in models.items():
            vertices = obj.get_vertices(env_ids=[0], scale=True)[0].cpu().numpy()
            faces = obj.get_triangles(env_ids=[0])[0].cpu().numpy()
            if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
                raise ValueError(f"Recorded object has no finite visual mesh: {uid}")
            geometry[uid] = vertices, faces
            data.boxes[uid] = tuple(np.ptp(vertices, axis=0))
        publisher = MarkerFramePublisher(sim)
        yield NativeRenderer(data, sim, publisher, names, center, models, geometry)
    finally:
        if publisher is not None:
            publisher.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
