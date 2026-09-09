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

"""Optional simulation conveniences, not a skill or task-program interpreter."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from embodichain.utils import configclass
from embodichain.utils.configclass import update_class_from_dict
from embodichain.utils.math import look_at_to_pose
from embodichain.utils.utility import load_config

from .catalog import write_json

__all__ = ["Lab", "LabCfg"]


@configclass
class LabCfg:
    """One experiment's physical setup; robot overrides are applied before reset."""

    scene: str = ""
    robot_component: str = ""
    robot_overrides: dict = {}
    sim: dict = {}
    seed: int = 0
    control_dt: float = 0.04
    fps: int = 25
    camera_eye: list[float] = [1.5, -1.6, 1.8]
    camera_target: list[float] = [0.0, 0.0, 0.75]


def _merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def _scene_entities(physical: dict) -> Iterator[tuple[str, dict]]:
    # Match EmbodiedEnv: USD initialization must precede dynamic rigid insertion.
    for section in ("background", "articulation", "rigid_object"):
        for item in physical[section]:
            yield section, item


class Lab:
    """Expose the real simulator and robot while recording every update.

    Scripts can use any project API. The update wrapper only observes physics;
    it does not choose actions, retry IK, move objects, or decide task success.
    """

    def __init__(self, cfg: LabCfg, output: Path) -> None:
        from embodichain.gen_sim.task_engine.orchestration.source_scene import (
            prepare_scene,
        )
        from embodichain.gen_sim.task_engine.orchestration.scene_source import (
            fingerprint_scene_source,
        )
        from embodichain.gen_sim.task_engine.orchestration.scene_assets import (
            normalize_scene_assets,
        )
        from embodichain.gen_sim.task_engine.task_program_bundle import (
            _bind_embodiment_to_scene,
            _scene_payload,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            ArticulationCfg,
            LightCfg,
            RigidObjectCfg,
            RobotCfg,
        )

        self.cfg = cfg
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.time = 0.0
        self.physics_steps = 0
        self.frames = 0
        self.events: list[dict] = []
        self.ik_attempts = 0
        self.ik_failures = 0
        self.sim = None
        self.camera = None
        self._writer = None
        self._telemetry = None
        self._last_qpos = None
        self.max_joint_step = 0.0
        self.max_joint_speed = 0.0
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        write_json(
            output / "source_fingerprint.json",
            fingerprint_scene_source(cfg.scene).to_dict(),
        )
        scene = normalize_scene_assets(prepare_scene(cfg.scene), output / "assets")
        self.scene = scene
        physical = _scene_payload(scene, program_id="agent_lab")["simulation"]
        component = load_config(cfg.robot_component)
        _bind_embodiment_to_scene(component, table_top_z=scene.table_top_z)
        robot_data = _merge(component["simulation"], cfg.robot_overrides)
        sim_data = _merge(
            {
                "headless": True,
                "sim_device": "cuda:0",
                "physics_dt": 0.01,
                "width": 960,
                "height": 720,
                "render_cfg": {"renderer": "fast-rt"},
            },
            cfg.sim,
        )
        write_json(
            output / "physical_setup.json",
            {"simulation": sim_data, "robot": robot_data, "scene": physical},
        )
        sim_cfg = SimulationManagerCfg()
        update_class_from_dict(sim_cfg, sim_data)
        sim_cfg.validate()
        self.sim = SimulationManager(sim_cfg)
        for light in physical["light"]["direct"]:
            self.sim.add_light(LightCfg.from_dict(light))
        self.robot = self.sim.add_robot(RobotCfg.from_dict(robot_data))
        self.objects = {}
        self.articulations = {}
        for section, item in _scene_entities(physical):
            if section == "articulation":
                self.articulations[item["uid"]] = self.sim.add_articulation(
                    ArticulationCfg.from_dict(item)
                )
            else:
                self.objects[item["uid"]] = self.sim.add_rigid_object(
                    RigidObjectCfg.from_dict(item)
                )
        self.camera = self.sim.get_env().create_camera(
            "agent_lab_audience", sim_data["width"], sim_data["height"]
        )
        self.camera.open_camera()
        pose = look_at_to_pose(cfg.camera_eye, cfg.camera_target, [0, 0, 1])[0]
        pose = pose.cpu().numpy()
        pose[:3, 1:3] *= -1
        self.camera.set_world_pose(pose.astype(np.float32))
        self._writer = imageio.get_writer(
            str(output / "video.mp4"),
            fps=cfg.fps,
            codec="libx264",
            quality=7,
            macro_block_size=1,
            ffmpeg_params=[
                "-movflags",
                "+frag_keyframe+empty_moov",
                "-g",
                str(cfg.fps),
            ],
        )
        self._telemetry = (output / "telemetry.jsonl").open("w", buffering=1)
        # Keep a visual artifact even if the native GPU physics initializer aborts.
        self.camera.render()
        loaded = np.ascontiguousarray(np.asarray(self.camera.get_rgb_map())[..., :3])
        imageio.imwrite(output / "loaded.png", loaded)
        self._writer.append_data(loaded)
        self.frames += 1
        if self.sim.is_use_gpu_physics:
            self.sim.init_gpu_physics()
        # Instrument this run's instance, including scripts calling sim.update directly.
        self._physics_update = self.sim.update
        self.sim.update = self._recorded_update
        self.initial = self.snapshot()
        write_json(output / "initial_state.json", self.initial)
        write_json(
            output / "articulation_info.json",
            {
                uid: {
                    "joint_names": obj.joint_names,
                    "link_names": obj.link_names,
                    "reported_qpos_limits": obj.get_qpos_limits()
                    .detach()
                    .cpu()
                    .tolist(),
                    "body_scale": list(obj.cfg.body_scale),
                }
                for uid, obj in self.articulations.items()
            },
        )
        self.capture("initial")
        self._next_capture = 1.0 / cfg.fps

    def snapshot(self) -> dict[str, Any]:
        """Read actual positions, not desired commands or symbolic held state."""
        return {
            "time": self.time,
            "physics_steps": self.physics_steps,
            "qpos": self.robot.get_qpos().detach().cpu().tolist(),
            "qpos_target": self.robot.get_qpos(target=True).detach().cpu().tolist(),
            "objects": {
                uid: obj.get_local_pose().detach().cpu().tolist()
                for uid, obj in self.objects.items()
            },
            "articulations": {
                uid: {
                    "pose": obj.get_local_pose().detach().cpu().tolist(),
                    "qpos": obj.get_qpos().detach().cpu().tolist(),
                }
                for uid, obj in self.articulations.items()
            },
        }

    def capture(self, name: str | None = None) -> Path | None:
        """Append a frame to the streaming MP4 and optionally save a keyframe."""
        self.camera.render()
        frame = np.ascontiguousarray(np.asarray(self.camera.get_rgb_map())[..., :3])
        self._writer.append_data(frame)
        self.frames += 1
        self._telemetry.write(json.dumps(self.snapshot(), allow_nan=False) + "\n")
        imageio.imwrite(self.output / "latest.png", frame)
        if name is not None:
            path = self.output / f"{name}.png"
            imageio.imwrite(path, frame)
            return path
        return None

    def _recorded_update(self, physics_dt: float | None = None, step: int = 10) -> None:
        dt = self.sim.sim_config.physics_dt if physics_dt is None else physics_dt
        for _ in range(step):
            self._physics_update(physics_dt=dt, step=1)
            self.time += dt
            self.physics_steps += 1
            current = self.robot.get_qpos().detach().clone()
            if self._last_qpos is not None:
                delta = float((current - self._last_qpos).abs().max())
                self.max_joint_step = max(self.max_joint_step, delta)
                self.max_joint_speed = max(self.max_joint_speed, delta / dt)
            self._last_qpos = current
            if self.time + 1e-9 >= self._next_capture:
                self.capture()
                self._next_capture += 1.0 / self.cfg.fps

    def step(self, seconds: float | None = None) -> None:
        """Advance real physics for a control interval or a specified duration."""
        seconds = self.cfg.control_dt if seconds is None else seconds
        count = max(1, round(seconds / self.sim.sim_config.physics_dt))
        self.sim.update(step=count)

    def event(self, name: str, **measurements: Any) -> None:
        """Save a named milestone with caller-supplied measurements."""
        self.events.append({"name": name, "time": self.time, **measurements})
        write_json(self.output / "events.json", self.events)
        self.capture(f"event_{len(self.events):03d}")

    def move_joints(self, qpos: Any, *, part: str, seconds: float = 2.0) -> None:
        """Send a smooth target trajectory; never overwrite actual robot state."""
        start = self.robot.get_qpos(name=part).clone()
        goal = torch.as_tensor(qpos, device=start.device, dtype=start.dtype).reshape_as(
            start
        )
        count = max(1, round(seconds / self.cfg.control_dt))
        for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
            blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
            self.robot.set_qpos(start + blend * (goal - start), name=part, target=True)
            self.step()

    def move_tcp(self, pose: Any, *, part: str, seconds: float = 2.0) -> None:
        """Optional single-target IK helper; scripts may use any other planner."""
        target = torch.as_tensor(pose, device=self.robot.device, dtype=torch.float32)
        target = target.reshape(1, 4, 4)
        self.ik_attempts += 1
        success, qpos = self.robot.compute_ik(
            target, joint_seed=self.robot.get_qpos(name=part), name=part
        )
        if not bool(success.all()):
            self.ik_failures += 1
            raise RuntimeError(
                f"IK failed for {part}; try another pose, seed or planner."
            )
        self.move_joints(qpos, part=part, seconds=seconds)

    def close(self) -> None:
        """Flush recordings even after a solver exception."""
        from embodichain.lab.sim import SimulationManager

        try:
            if self._writer is not None:
                try:
                    self.capture("final")
                    write_json(self.output / "final_state.json", self.snapshot())
                    write_json(
                        self.output / "metrics.json",
                        {
                            "simulation_seconds": self.time,
                            "physics_steps": self.physics_steps,
                            "video_frames": self.frames,
                            "max_joint_step_rad": self.max_joint_step,
                            "max_joint_speed_rad_s": self.max_joint_speed,
                            "helper_ik_attempts": self.ik_attempts,
                            "helper_ik_failures": self.ik_failures,
                            "task_success": None,
                        },
                    )
                finally:
                    self._writer.close()
                    self._telemetry.close()
        finally:
            if self.sim is not None:
                self.sim.destroy(exit_process=False)
                SimulationManager.flush_cleanup_queue()
