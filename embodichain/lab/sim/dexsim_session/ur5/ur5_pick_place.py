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

"""Run the existing UR5 cube/platform Task Program in a DexSim-owned World."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
import json

import numpy as np
import yaml

from .verify_ur5 import verify

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext
    from dexsim.scene import Scene
    import torch
    from embodichain.lab.gym.envs.demo import DemoSegment
    from embodichain.lab.sim.cfg import RobotCfg, RigidObjectCfg

__all__: list[str] = []

_PHYSICS_DT = 0.01
_STABILITY_SECONDS = 0.6
_SEED = 0


def _build_scene(
    context: SessionContext,
    robot_cfg: RobotCfg,
    rigid_cfgs: list[RigidObjectCfg],
    *,
    video: bool,
) -> Scene:
    from dexsim import spawn
    from embodichain.lab.sim.spawn.descriptors import (
        articulation_desc_from_cfg,
        configure_articulation_desc,
        rigid_desc_from_cfg,
    )
    from embodichain.lab.sim.spawn.source import (
        _apply_dexsim_source_overlay,
        _capture_dexsim_source_physics,
        _retain_dexsim_source_descriptor,
    )

    builder = context.builder
    robot_desc = builder.add_articulation(
        articulation_desc_from_cfg(robot_cfg, per_env=False)
    )

    for cfg in rigid_cfgs:
        desc, materials = rigid_desc_from_cfg(cfg, per_env=False)
        builder.materials.update(materials)
        builder.add_object(desc)

    if video:
        builder.add_camera(
            spawn.CameraDesc(name="demo", width=1280, height=720, fov=48)
        )
        for name, direction, intensity in (
            ("key", (-0.45, 0.38, -0.81), 4.0),
            ("fill", (0.42, -0.58, -0.70), 1.5),
        ):
            builder.add_light(
                spawn.LightDesc(
                    name=name,
                    light_type="DIRECTION",
                    direction=direction,
                    intensity=intensity,
                )
            )

    scene = builder.finalize()
    handle = scene.get_articulation(robot_cfg.uid)
    # Default materializes via the native URDF loader. Apply the same source
    # overlay that SpawnScene applies before constructing Robot views.
    source = robot_desc if robot_desc.links else handle.articulation_desc
    _capture_dexsim_source_physics(handle, source)
    configure_articulation_desc(source, robot_cfg)
    if source._embodichain_preserve_source_physics:
        _retain_dexsim_source_descriptor(handle, source)
    else:
        _apply_dexsim_source_overlay(handle, source)
    return scene


class _Episode:
    """One live configuration and a lazy Task Program cursor on the owning thread."""

    def __init__(
        self,
        context: SessionContext,
        deployment: Path,
        video: bool,
        stack: ExitStack,
        *,
        live_paths: dict[str, str] | None = None,
        destinations: list[str] | None = None,
    ) -> None:
        import torch
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg
        from embodichain.lab.gym.utils.registration import get_env_spec
        from embodichain.lab.sim import SimulationManager
        from embodichain.lab.sim.cfg import (
            DefaultPhysicsCfg,
            RigidObjectCfg,
            RobotPresetCfg,
        )
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.shapes import CubeCfg
        from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers
        from embodichain.utils import set_seed

        if live_paths is None:
            set_seed(_SEED)
        spec = yaml.safe_load(deployment.read_text(encoding="utf-8"))
        cfg = config_to_cfg(spec, source_path=deployment)
        factory = get_env_spec(spec["id"]).task_program_adapter_factory
        physics = cfg.sim_cfg.physics_cfg
        if live_paths is None and (
            not isinstance(physics, DefaultPhysicsCfg)
            or physics.physics_dt != _PHYSICS_DT
            or physics.num_substeps != 1
            or cfg.num_envs != 1
            or context.builder.backend != "dexsim"
            or not context.world.is_physics_manually_update()
        ):
            raise ValueError(
                "UR5 demo requires one manually stepped Default arena at 10 ms"
            )
        robot_cfg = cfg.robot
        if isinstance(robot_cfg, RobotPresetCfg):
            robot_cfg = robot_cfg.resolve(cfg.sim_cfg.physics_cfg)
        prepare_robot_grippers(robot_cfg)
        robot_cfg.fpath = robot_cfg.urdf_cfg.assemble_urdf()
        ground = RigidObjectCfg(
            uid="ground",
            shape=CubeCfg(
                size=[4.0, 4.0, 0.04],
                visual_material=VisualMaterialCfg(
                    base_color=[0.24, 0.30, 0.37, 1.0], roughness=0.8
                ),
            ),
            body_type="static",
            init_pos=[0.0, 0.0, -0.02],
        )
        objects = {body.uid: body for body in [*cfg.background, *cfg.rigid_object]}
        objects["cube"].shape.visual_material = VisualMaterialCfg(
            base_color=[0.95, 0.32, 0.07, 1.0]
        )
        objects["platform"].shape.visual_material = VisualMaterialCfg(
            base_color=[0.12, 0.55, 0.53, 1.0]
        )
        self.live_paths = live_paths
        self.destinations = destinations or []
        if live_paths is None:
            scene = _build_scene(
                context, robot_cfg, [ground, *objects.values()], video=video
            )
            paths = {name: name for name in (robot_cfg.uid, *objects)}
        else:
            from copy import deepcopy

            scene = context.scene
            if scene is None or not context.world.is_physics_manually_update():
                raise ValueError("Expected an existing manual finalized World")
            diagnostics = scene.get_solver_diagnostics() or {}
            if (
                scene.backend != "newton"
                or diagnostics.get("solver_type") != "dexuni"
                or diagnostics.get("dt") != _PHYSICS_DT
                or diagnostics.get("num_substeps") != physics.num_substeps
                or not (diagnostics.get("features") or {}).get("mujoco_solve_enabled")
            ):
                raise ValueError("Scene does not match the UR5 DexUni control profile")
            paths = dict(live_paths)
            if not self.destinations or paths["platform"] not in self.destinations:
                raise ValueError("A bound platform must be present in destinations")
            for index, path in enumerate(self.destinations):
                if path == paths["platform"]:
                    continue
                extra = deepcopy(objects["platform"])
                extra.uid = f"support_{index}"
                objects[extra.uid] = extra
                paths[extra.uid] = path
        if live_paths is not None:
            for uid, object_cfg in objects.items():
                body = scene.get_object(paths[uid])
                colliders = body.object_desc.collisions
                if (
                    len(colliders) != 1
                    or colliders[0].geometry_type.value != "cube"
                    or not np.allclose(
                        colliders[0].size, object_cfg.shape.size, rtol=0, atol=1e-5
                    )
                    or not np.allclose(
                        colliders[0].local_transform, np.eye(4), atol=1e-5
                    )
                ):
                    raise ValueError(
                        f"{paths[uid]} must have the deployment's centered box collider"
                    )
        manager = SimulationManager.attach(
            context,
            arena=scene.arenas.names[0],
            robots=[robot_cfg],
            rigid_objects=list(objects.values()),
            paths=paths,
        )
        stack.callback(manager.detach)
        robot = manager.get_robot(robot_cfg.uid)
        if live_paths is None:
            home = torch.as_tensor(robot_cfg.init_qpos, dtype=torch.float32).reshape(
                1, -1
            )
            robot.set_qpos(home, target=False)
            robot.set_qpos(home)
            context.world.set_delta_time(_PHYSICS_DT)
            for _ in range(60):
                context.world.update(_PHYSICS_DT)
        self._platform_views = {
            path: manager.get_rigid_object(uid)
            for uid, path in paths.items()
            if uid == "platform" or uid.startswith("support_")
        }
        self._selected_platform = paths["platform"]
        task_manager = manager
        if live_paths is not None:
            episode = self

            class PlatformRole:
                uid = "platform"

                def get_local_pose(self, to_matrix=False):
                    return episode._platform_views[
                        episode._selected_platform
                    ].get_local_pose(to_matrix=to_matrix)

                def __getattr__(self, name):
                    return getattr(
                        episode._platform_views[episode._selected_platform], name
                    )

            platform_role = PlatformRole()

            class TaskScene:
                def get_rigid_object(self, uid):
                    return (
                        platform_role
                        if uid == "platform"
                        else manager.get_rigid_object(uid)
                    )

                def __getattr__(self, name):
                    return getattr(manager, name)

            task_manager = TaskScene()

        substeps = cfg.sim_steps_per_control
        period = substeps * _PHYSICS_DT

        def new_adapter():
            adapter = factory.create_adapter(
                SimpleNamespace(sim=task_manager, robot=robot, step_dt=period, cfg=cfg)
            )
            stack.callback(adapter.close)
            return adapter

        self._new_adapter = new_adapter
        self._program = cfg.task_program
        adapter = new_adapter()
        cube, platform = (
            scene.get_object(paths[name]) for name in ("cube", "platform")
        )
        arm_solver = robot.cfg.solver_cfg["arm"]
        tool_link = arm_solver.end_link_name
        link_from_tcp = torch.as_tensor(arm_solver.tcp, dtype=torch.float32)
        [grip_index] = robot.get_joint_ids("eef", remove_mimic=True)
        hand_config = robot.cfg.gripper_bindings["eef"]
        [native_joint] = [
            name
            for name, actual in hand_config["joint_names"].items()
            if actual == robot.joint_names[grip_index]
        ]
        grip_presets = hand_config["defaults"]["joint_commands"]
        active = robot.active_joint_ids
        self.context = context
        self.manager = manager
        self.scene = scene
        self.robot = robot
        self.arm = arm_solver
        self.link_from_tcp = link_from_tcp
        self.grip_index = grip_index
        self.grip_open = grip_presets["open"][native_joint]
        self.grip_close = grip_presets["grasp"][native_joint]
        self.cube = cube
        self.platform = platform
        self.active = active
        self.adapter = adapter
        self.compiled = adapter.compile(cfg.task_program)
        self.max_episode_steps = cfg.max_episode_steps
        self.substeps = substeps
        self.period = period
        self.video = video
        self.deployment = deployment
        self.sizes = {
            name: list(objects[name].shape.size) for name in ("cube", "platform")
        }
        self.physics_step = 0
        self.control_step = 0
        self.phase = "idle"
        self._recording = ExitStack()
        stack.callback(self._recording.close)
        self._stream = None
        self._writer = None
        self.recent: deque[dict[str, Any]] = deque(maxlen=12)
        self.segments: list[dict[str, Any]] = []

    def select_platform(self, path: str) -> None:
        self.manager.validate()
        if path not in self._platform_views:
            raise ValueError("Unbound placement destination")
        self._selected_platform = path
        self.platform = self.scene.get_object(path)

    def place_bridge(self, verified_state):
        """Ground a new Place workflow against the selected live support frame."""
        from dataclasses import replace

        segment = self._program.program.items[-1]
        if segment.name != "place":
            raise ValueError("Expected an explicitly registered Place segment")
        adapter = self._new_adapter()
        program = replace(self._program, program=segment)
        bridge = adapter.create_bridge(adapter.compile(program))
        bridge._runtime.adopt_verified_task_state(verified_state)
        return bridge

    def tcp(self) -> torch.Tensor:
        return (
            self.robot.get_link_pose(self.arm.end_link_name, to_matrix=True)[0]
            @ self.link_from_tcp
        )

    def observe(self, command: torch.Tensor | None = None) -> dict[str, Any]:
        return {
            "physics_step": self.physics_step,
            "control_step": self.control_step,
            "simulation_time_s": self.physics_step * _PHYSICS_DT,
            "phase": self.phase,
            "cube_pose": self.cube.get_world_pose().tolist(),
            "platform_pose": self.platform.get_world_pose().tolist(),
            "tcp_pose": self.tcp().tolist(),
            "qpos": self.robot.get_qpos()[0].tolist(),
            "qvel": self.robot.get_qvel()[0].tolist(),
            "target_qpos": None if command is None else command[0].tolist(),
        }

    def _sample(self, command: torch.Tensor | None = None) -> None:
        row = self.observe(command)
        row["physics_step"] -= self._recording_physics_origin
        row["control_step"] -= self._recording_control_origin
        row["simulation_time_s"] = row["physics_step"] * _PHYSICS_DT
        self.recent.append(row)
        self._stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _frame(self) -> None:
        if self._writer is not None:
            self._camera.render()
            rgb = np.array(self._camera.get_rgb_map(), copy=True)[..., :3]
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            self._writer.append_data(rgb)

    def record(self, output: Path, *, phase: str = "initial") -> None:
        """Open evidence files before the first native action."""
        if self._stream is not None:
            raise RuntimeError("An episode recorder is already active")
        self.output = output
        self._recording_physics_origin = self.physics_step
        self._recording_control_origin = self.control_step
        self._stream = self._recording.enter_context(
            (output / "ur5-trajectory.jsonl").open("w", encoding="utf-8")
        )
        if self.video:
            import imageio.v2 as imageio

            if self.live_paths is None:
                self.scene.set_environment_lighting(mode="constant", intensity=28.0)
            self._camera = self.scene.get_camera("demo")
            if self.live_paths is None:
                self._camera.look_at(
                    [-1.33, -1.02, 0.88], [-0.27, 0.07, 0.30], [0, 0, 1]
                )
            self._camera.open_camera()
            self._recording.callback(self._camera.close_camera)
            self._writer = self._recording.enter_context(
                imageio.get_writer(
                    output / "ur5-pick-place.mp4",
                    fps=1.0 / self.period,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=2,
                    ffmpeg_log_level="error",
                )
            )
        self.phase = phase
        self.recent.clear()
        self._sample()
        self._frame()

    def advance(self, phase: str, command: torch.Tensor) -> dict[str, Any]:
        """Apply one controller interval and sample every completed physics update."""
        if self.control_step >= self.max_episode_steps:
            raise RuntimeError("UR5 reached the configured step limit")
        self.manager.validate()
        self.phase = phase
        self.robot.set_qpos(command[:, self.active], joint_ids=self.active)
        self.control_step += 1
        for _ in range(self.substeps):
            self.context.world.update(_PHYSICS_DT)
            self.physics_step += 1
            if self._stream is not None:
                self._sample(command)
        self._frame()
        return {}

    def run_segment(
        self, segment: DemoSegment
    ) -> Generator[dict[str, Any], None, bool]:
        for action in segment.actions:
            yield self.advance(segment.name, action.value)
        accepted = bool(segment.validator().all())
        self.segments.append(
            {
                "name": segment.name,
                "accepted": accepted,
                "end_step": self.control_step,
                "runtime": segment.metadata.get("runtime"),
                "validation": segment.metadata.get("validation"),
            }
        )
        return accepted

    def settle(self) -> Generator[dict[str, Any], None, None]:
        hold = self.robot.get_qpos(target=True)
        for _ in range(round(_STABILITY_SECONDS / self.period)):
            yield self.advance("settle", hold)

    def close_recording(self) -> None:
        self._recording.close()
        self._stream = self._writer = None

    def finish(self, *, program_passed: bool) -> dict[str, Any]:
        """Finalize video and verify the complete physical episode."""
        self.close_recording()
        trace = self.output / "ur5-trajectory.jsonl"
        manifest = {
            "session_id": self.context.session_id,
            "deployment": str(self.deployment),
            "pose_convention": "4x4 homogeneous transforms in the single default arena; metres",
            "command_convention": "full position target in joint order; only commanded_joint_names are written",
            "program_id": self.compiled.program_id,
            "seed": _SEED,
            "physics_dt_s": _PHYSICS_DT,
            "substeps_per_control": self.substeps,
            "control_dt_s": self.period,
            "joint_names": list(self.robot.joint_names),
            "commanded_joint_names": [self.robot.joint_names[i] for i in self.active],
            "tcp_link": self.arm.end_link_name,
            "link_from_tcp": self.link_from_tcp.tolist(),
            "gripper": {
                "joint_index": self.grip_index,
                "open_position_m": self.grip_open,
                "grasp_position_m": self.grip_close,
            },
            "physics_clock": "completed World updates relative to the start of recording",
            "host_physics_step_origin": self._recording_physics_origin,
            "frames_per_control_step": int(self.video),
            "size_m": self.sizes,
            "segments": self.segments,
            "program_passed": program_passed,
        }
        manifest_path = self.output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        measurements = verify(trace, manifest)
        report = self.output / "verification.json"
        report.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")
        return {
            **measurements,
            "trajectory": str(trace),
            "manifest": str(manifest_path),
            "verification": str(report),
            **(
                {"video": str(self.output / "ur5-pick-place.mp4")} if self.video else {}
            ),
        }


def run(
    context: SessionContext,
    *,
    files: dict[str, Path],
    options: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Compose, execute and physically verify one UR5 pick-and-place episode."""
    with ExitStack() as stack:
        episode = _Episode(
            context, files["deployment"], options.get("video", True), stack
        )
        bridge = episode.adapter.create_bridge(episode.compiled)
        episode.record(output)
        for segment in bridge:
            work = episode.run_segment(segment)
            while True:
                try:
                    next(work)
                except StopIteration as done:
                    accepted = done.value
                    break
            if not accepted:
                break
        program_passed = bool(bridge.program_completed and bridge.completion_mask.all())
        if program_passed:
            for _ in episode.settle():
                pass
        return episode.finish(program_passed=program_passed)
