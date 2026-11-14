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

"""Measured reference/expansion comparison on a shared physics-free timeline."""

from __future__ import annotations

from contextlib import ExitStack, AbstractContextManager
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
import time
import numpy as np
from embodichain.data_pipeline.recording.media import _atomic_json as atomic_json
from embodichain.lab.expansion.persistence import _digest as file_digest
from embodichain.lab.visualization.markers import MarkerFramePublisher

__all__ = [
    "MeasuredTrajectory",
    "record_reference_trajectory",
    "preview_expansion_comparison",
]

GHOST_COLOR = (0.15, 0.8, 0.95, 0.35)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MeasuredTrajectory:
    """One arena's measured poses; all frames include the initial boundary."""

    name: str
    robot_uid: str
    joint_names: tuple[str, ...]
    timestamps: np.ndarray
    qpos: np.ndarray
    root_poses: np.ndarray
    object_poses: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        for name in ("timestamps", "qpos", "root_poses"):
            object.__setattr__(self, name, np.array(getattr(self, name), copy=True))
        object.__setattr__(
            self,
            "object_poses",
            MappingProxyType(
                {
                    key: np.array(value, copy=True)
                    for key, value in self.object_poses.items()
                }
            ),
        )
        count = len(self.timestamps)
        if (
            count < 2
            or self.timestamps.shape != (count,)
            or self.timestamps[0] != 0
            or not np.all(np.diff(self.timestamps) > 0)
            or not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or self.qpos.shape != (count, len(self.joint_names))
        ):
            raise ValueError("Preview requires a nonempty, ordered measured trajectory")
        arrays = (
            self.timestamps,
            self.qpos,
            self.root_poses,
            *self.object_poses.values(),
        )
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Preview trajectory must contain finite measurements")
        for poses in (self.root_poses, *self.object_poses.values()):
            if poses.shape != (count, 7) or not np.allclose(
                np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-5, rtol=0
            ):
                raise ValueError("Preview poses must be xyz + normalized xyzw")
        for array in (
            self.timestamps,
            self.qpos,
            self.root_poses,
            *self.object_poses.values(),
        ):
            array.flags.writeable = False

    def frame_at(self, elapsed: float) -> int:
        """Select the latest measured boundary, holding the final pose at the end."""
        return int(
            np.clip(
                np.searchsorted(self.timestamps, elapsed, side="right") - 1,
                0,
                len(self.timestamps) - 1,
            )
        )

    @classmethod
    def load_reference(cls, path: Path) -> MeasuredTrajectory:
        """Read a pickle-free reference captured by record_reference_trajectory."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                name="original",
                robot_uid=str(data["robot_uid"]),
                joint_names=tuple(data["joint_names"].tolist()),
                timestamps=data["timestamps"].copy(),
                qpos=data["qpos"].copy(),
                root_poses=data["root_poses"].copy(),
                object_poses={
                    str(uid): data[f"object_{index}"].copy()
                    for index, uid in enumerate(data["object_uids"])
                },
            )

    @classmethod
    def load_episode(cls, storage_id: str, *, robot_uid: str) -> MeasuredTrajectory:
        """Read only an accepted native episode, preserving full joint order."""
        from embodichain.lab.expansion import load_expansion_episode

        episode = load_expansion_episode(storage_id)
        if not episode.validation.accepted:
            raise ValueError("comparison requires a qualified native episode")
        metadata, observations = episode.metadata, episode.observations
        robot = next(
            item for item in metadata["initial_state"] if item["uid"] == robot_uid
        )
        root_key = next(
            field["key"]
            for field in metadata["scene_state_fields"]
            if field["entity_uid"] == robot_uid and field["field"] == "pose"
        )

        def array(value):
            return value.detach().cpu().numpy().copy()

        return cls(
            name=episode.episode_id,
            robot_uid=robot_uid,
            joint_names=tuple(robot["joint_names"]),
            timestamps=array(episode.timestamps),
            qpos=array(observations["full_joint_positions"]),
            root_poses=array(observations[root_key]),
            object_poses={
                key[len("rigid/") : -len("/state")]: array(value)[:, :7]
                for key, value in observations.items()
                if key.startswith("rigid/") and key.endswith("/state")
            },
        )


def record_reference_trajectory(env: object, path: Path) -> None:
    """Execute the unmodified expert once and archive its measured scene poses."""
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.sim.scene import SceneView, capture_state

    view = SceneView(env.sim, env_ids=(0,))
    frames = [capture_state(view).batches[0]]

    def observe(*_args):
        frames.append(capture_state(view).batches[0])

    episode = execute_demo_episode(env, step_observer=observe)
    atomic_json(path.with_suffix(".json"), episode.to_metadata())
    if not episode.completed or not episode.all_success:
        raise RuntimeError(f"Nominal preview expert failed: {episode.terminal_reason}")
    robot_index = next(
        index for index, item in enumerate(frames[0]) if item.uid == env.robot.uid
    )
    robot = frames[0][robot_index]
    objects = [
        (index, item.uid)
        for index, item in enumerate(frames[0])
        if item.kind == "rigid_object"
    ]
    arrays = {
        "robot_uid": np.asarray(robot.uid),
        "joint_names": np.asarray(robot.joint_names),
        "timestamps": np.arange(len(frames), dtype=np.float64) * env.step_dt,
        "qpos": np.asarray([frame[robot_index].qpos for frame in frames]),
        "root_poses": np.asarray([frame[robot_index].pose for frame in frames]),
        "object_uids": np.asarray([uid for _, uid in objects]),
    }
    arrays.update(
        {
            f"object_{ordinal}": np.asarray([frame[index].pose for frame in frames])
            for ordinal, (index, _) in enumerate(objects)
        }
    )
    np.savez_compressed(path, **arrays)
    # Validate the exact archive before another World is created.
    MeasuredTrajectory.load_reference(path)


def _object_prototype(obj):
    """Use the physical object's actual visual shape and scale exactly once."""
    from embodichain.lab.visualization.markers import MarkerPrototypeCfg
    from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, SphereCfg

    shape, scale = obj.cfg.shape, np.asarray(obj.cfg.body_scale)
    options = {"color": GHOST_COLOR, "scene_lighting": True, "unlit": False}
    if isinstance(shape, CubeCfg):
        return MarkerPrototypeCfg(
            shape="box", scale=tuple(scale * shape.size), **options
        )
    if isinstance(shape, SphereCfg):
        return MarkerPrototypeCfg(
            shape="sphere", scale=tuple(scale * 2 * shape.radius), **options
        )
    if isinstance(shape, MeshCfg) or obj.cfg.source is not None:
        # Use the simulator's materialized visual geometry: GLB loading may
        # convert axes, while USD sources can author their own mesh transforms.
        vertices = obj.get_vertices(env_ids=[0], scale=False)[0].cpu().numpy()
        faces = obj.get_triangles(env_ids=[0])[0].cpu().numpy()
        return MarkerPrototypeCfg(
            shape="mesh", vertices=vertices, faces=faces, scale=tuple(scale), **options
        )
    raise ValueError(f"Unsupported preview object shape for {obj.uid}")


def _precompute_link_poses(clip, robot, asset_path):
    """Compute complete-arm FK once; browsing never runs FK or loads episodes."""
    import torch
    import pytorch_kinematics as pk
    from scipy.spatial.transform import Rotation

    chain = pk.build_chain_from_urdf(Path(asset_path).read_bytes())
    indices = [clip.joint_names.index(name) for name in robot.joint_names]
    poses = []
    with torch.no_grad():
        for start in range(0, len(clip.timestamps), 256):
            qpos = torch.as_tensor(
                clip.qpos[start : start + 256, indices], dtype=torch.float32
            )
            fk = chain.forward_kinematics(qpos)
            poses.append(
                torch.stack(
                    [fk[name].get_matrix() for name in robot.link_names], dim=1
                ).numpy()
            )
    roots = np.broadcast_to(np.eye(4), (len(clip.timestamps), 4, 4)).copy()
    roots[:, :3, :3] = Rotation.from_quat(clip.root_poses[:, 3:]).as_matrix()
    roots[:, :3, 3] = clip.root_poses[:, :3]
    return roots[:, None] @ np.concatenate(poses)


class _ReferencePlayer:
    """Publish measured entities through the native render-state owner."""

    def __init__(self, env, reference):
        from embodichain.lab.sim.scene import SceneView, capture_state

        if tuple(env.robot.joint_names) != reference.joint_names:
            raise ValueError("Preview robot joint order differs from recording")
        self.view = SceneView(env.sim, env_ids=(0,))
        self.initial = capture_state(self.view)
        self.reference = reference

    def queue(self, frame):
        from dataclasses import replace
        from embodichain.lab.sim.scene import restore_render_state

        reference = self.reference
        entities = []
        for state in self.initial.batches[0]:
            if state.uid == reference.robot_uid:
                state = replace(
                    state,
                    pose=tuple(map(float, reference.root_poses[frame])),
                    qpos=tuple(map(float, reference.qpos[frame])),
                )
            elif state.uid in reference.object_poses:
                state = replace(
                    state,
                    pose=tuple(map(float, reference.object_poses[state.uid][frame])),
                )
            entities.append(state)
        restore_render_state(
            self.view, replace(self.initial, batches=(tuple(entities),))
        )

    def close(self):
        from embodichain.lab.sim.scene import restore_render_state

        restore_render_state(self.view, self.initial)


def _accepted_candidates(operators):
    from embodichain.lab.expansion import ArtifactEpisodeSink

    return [
        {
            "operator": name,
            "storage_id": receipt.storage_id,
            "episode_id": receipt.episode_id,
        }
        for name, report in operators.items()
        if "directory" in report
        for receipt in ArtifactEpisodeSink(report["directory"]).inspect()
    ]


def preview_expansion_comparison(
    environment_factory: Callable[[], AbstractContextManager],
    reference_path: Path,
    operators: dict[str, dict] | None,
    *,
    renderer: str,
    scene_entities: tuple[str, ...],
    output: Path,
    collect: Callable | None = None,
    max_frames: int | None = None,
    frame_observer: Callable | None = None,
    expected_trajectories: int | None = None,
) -> dict:
    """Browse accepted trajectories; N advances, Space pauses and Q closes.

    With a collection callback, warm model resources before opening the window,
    then delegate collection and load only its qualified measured trajectories.
    Only the selected expansion is visible; N preserves playback time and
    pause state. Rendering cannot alter episodes or advance physics.

    Args:
        environment_factory: Context manager factory for the caller's native
            environment. The viewer displays physical row zero; other rows are
            retained with World-wide physics disabled during playback.
        reference_path: Pickle-free measured reference produced by the recorder.
        operators: Named artifact directories containing accepted native receipts.
        renderer: Renderer label in viewer evidence; the factory configures it.
        scene_entities: Recorded objects to display with the selected robot ghost.
        output: Viewer manifest and interaction evidence directory.
        collect: Optional callback receiving simulation, window, output and status;
            returns named artifact directories. Collection ownership stays external.
        max_frames: Optional display-frame budget for bounded playback.
        frame_observer: Optional callback receiving environment, frame number,
            selected index and shared elapsed time after publication.
        expected_trajectories: Required preload count for early-window collection.

    Returns:
        Playback outcome, selection/frame counts and actual physics-step delta.
    """
    from collections import deque
    from dexsim.engine import InputControl
    from dexsim.types import InputKey
    from embodichain.lab.visualization.markers import (
        RobotMarkers,
        RobotMarkersAppearanceCfg,
        RobotMarkersCfg,
        VisualizationMarkers,
        VisualizationMarkersCfg,
    )
    import os

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if collect is not None and (
        type(expected_trajectories) is not int or expected_trajectories < 1
    ):
        raise ValueError("Early preview requires the expected trajectory count")
    with environment_factory() as env:
        sim = env.sim
        physics_enabled = sim.physics_enabled
        sim.enable_physics(False)
        physics_steps_before = sim.physics_step_index
        with ExitStack() as cleanup:
            cleanup.callback(sim.enable_physics, physics_enabled)
            players = cleanup.enter_context(ExitStack())
            markers = MarkerFramePublisher(sim, xray=True)
            cleanup.callback(markers.close)
            robot = RobotMarkers(
                RobotMarkersCfg(
                    name="expansion_robot",
                    asset_path=env.robot.cfg.fpath,
                    coordinate_space="env",
                    appearance=RobotMarkersAppearanceCfg(
                        color=GHOST_COLOR[:3], opacity=GHOST_COLOR[3]
                    ),
                ),
                markers,
            )
            objects = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="expansion_objects",
                    coordinate_space="env",
                    markers={
                        uid: _object_prototype(sim.get_rigid_object(uid))
                        for uid in scene_entities
                    },
                ),
                markers,
            )

            def open_window():
                if not sim.open_window():
                    raise RuntimeError(
                        "The requested native expansion preview did not open"
                    )
                cleanup.callback(sim.close_window)
                window = sim.get_world().get_windows()
                center = np.mean(
                    [env.robot.get_local_pose()[0, :3].detach().cpu().numpy()]
                    + [
                        sim.get_rigid_object(uid)
                        .get_local_pose()[0, :3]
                        .detach()
                        .cpu()
                        .numpy()
                        for uid in scene_entities
                    ],
                    axis=0,
                )
                center[2] += 0.2
                window.set_look_at(
                    center + [1.6, -1.8, 1.1], center, np.array([0.0, 0.0, 1.0])
                )
                return window

            window = None
            window_opened_at = None
            if collect is not None:
                # Allocate all native mesh instances before the render loop.
                # Their hidden placeholder poses require no recorded data/FK.
                robot.visualize_link_poses(
                    np.broadcast_to(
                        np.eye(4),
                        (1, expected_trajectories, len(robot.link_names), 4, 4),
                    ),
                    env_ids=(0,),
                )
                count = expected_trajectories * len(scene_entities)
                objects.visualize(
                    translations=np.zeros((count, 3)),
                    orientations=np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
                    marker_indices=np.tile(
                        np.arange(len(scene_entities)), expected_trajectories
                    ),
                )
                for name, width in (
                    ("expansion_robot", robot.num_visuals),
                    ("expansion_objects", len(scene_entities)),
                ):
                    markers.groups[name].update(
                        visible=np.zeros(expected_trajectories * width, dtype=bool),
                        env_ids=(0,),
                    )
                markers.publish()
                sim.sync_render_state()
                window = open_window()
                window_opened_at = _utc_now()
                status = {
                    "state": "collecting",
                    "renderer": renderer,
                    "pid": os.getpid(),
                    "window_opened_at": window_opened_at,
                    "window_open": True,
                    "expected_trajectories": expected_trajectories,
                    "preloaded_robots": expected_trajectories,
                }
                atomic_json(output / "window.json", status)
                window.set_window_title("Expansion preparing | reference | Q cancel")
                operators = collect(sim, window, output, status)
                atomic_json(output / "status.json", {**status, "state": "loading"})

            reference = MeasuredTrajectory.load_reference(reference_path)
            candidates = _accepted_candidates(operators)
            atomic_json(
                output / "manifest.json",
                {
                    "reference": str(reference_path),
                    "reference_sha256": file_digest(reference_path),
                    "candidates": candidates,
                },
            )
            if not candidates:
                result = {
                    "state": "empty",
                    "trajectories": 0,
                    "reason": "no_accepted_episodes",
                }
                atomic_json(output / "status.json", result)
                return result
            clips = [
                MeasuredTrajectory.load_episode(
                    candidate["storage_id"], robot_uid=reference.robot_uid
                )
                for candidate in candidates
            ]
            if any(
                set(clip.joint_names) != set(reference.joint_names) for clip in clips
            ):
                raise ValueError("Preview expansion has incompatible robot joints")
            duration = max(clip.timestamps[-1] for clip in [reference, *clips]) + 0.5
            player = _ReferencePlayer(env, reference)
            players.callback(player.close)
            selected = 0
            links = [
                _precompute_link_poses(clip, robot, env.robot.cfg.fpath)
                for clip in clips
            ]
            last_frame = None

            def show_frame(elapsed):
                nonlocal last_frame
                original_frame = reference.frame_at(elapsed)
                expanded_frames = [clip.frame_at(elapsed) for clip in clips]
                current = (original_frame, *expanded_frames, selected)
                if current == last_frame:
                    sim.get_world().update(0.0)
                    return original_frame, expanded_frames
                player.queue(original_frame)
                robot.visualize_link_poses(
                    np.stack(
                        [poses[frame] for poses, frame in zip(links, expanded_frames)]
                    )[None],
                    env_ids=(0,),
                )
                poses = np.stack(
                    [
                        clip.object_poses[uid][frame]
                        for clip, frame in zip(clips, expanded_frames)
                        for uid in scene_entities
                    ]
                )
                objects.visualize(
                    translations=poses[:, :3],
                    orientations=poses[:, 3:],
                    marker_indices=np.tile(np.arange(len(scene_entities)), len(clips)),
                )
                for name, width in (
                    ("expansion_robot", robot.num_visuals),
                    ("expansion_objects", len(scene_entities)),
                ):
                    markers.groups[name].update(
                        visible=np.repeat(np.arange(len(clips)) == selected, width),
                        env_ids=(0,),
                    )
                markers.publish()
                # Publish once, after full-arm, object and reference state writes.
                sim.sync_render_state()
                # Publication only flushes render uploads in the pinned SDK;
                # poll window input and present even when every frame changes.
                sim.get_world().update(0.0)
                last_frame = current
                return original_frame, expanded_frames

            # Early windows reuse the warmed meshes; recorded-only previews
            # create their first instances before starting the render loop.
            show_frame(0)
            if window is None:
                window = open_window()
                window_opened_at = _utc_now()
            switches, frames, paused = 0, 0, False
            elapsed, previous = 0.0, time.monotonic()
            running = True
            pressed, held, polled = deque(), set(), set()
            preview_keys = {
                key.value
                for key in (
                    InputKey.SCANCODE_Q,
                    InputKey.SCANCODE_ESCAPE,
                    InputKey.SCANCODE_N,
                    InputKey.SCANCODE_SPACE,
                )
            }

            class PreviewInput(InputControl):
                def on_key_down(self, key):
                    if key in preview_keys and key not in held:
                        held.add(key)
                        pressed.append(key)

                def on_key_up(self, key):
                    held.discard(key)

            control = PreviewInput()
            window.add_input_control(control)
            cleanup.callback(window.remove_input_control, control)
            result = {
                "state": "playing",
                "trajectories": len(candidates),
                "renderer": renderer,
                "robot_layers": 1,
                "object_instances": len(scene_entities),
                "preloaded_robots": len(clips),
                "preloaded_object_instances": len(clips) * len(scene_entities),
                "marker_render_mode": "overlay",
                "pid": os.getpid(),
                "world_object_id": id(sim.get_world()),
                "num_envs": sim.num_envs,
                "window_opened_at": window_opened_at,
            }

            def publish():
                result.update(
                    selected_index=selected,
                    operator=candidates[selected]["operator"],
                    episode_id=candidates[selected]["episode_id"],
                    switches=switches,
                    frames=frames,
                    state="paused" if paused else "playing",
                    original_frame=original_frame,
                    expanded_frame=expanded_frames[selected],
                    elapsed_s=elapsed,
                    expanded_frames=expanded_frames,
                    marker_publications=markers.publications,
                    solid_qpos=env.robot.get_qpos()[0].detach().cpu().tolist(),
                )
                atomic_json(output / "status.json", result)
                window.set_window_title(
                    f"Expansion {selected + 1}/{len(candidates)} | {candidates[selected]['operator']} | N next | Space pause | Q close"
                )

            while (
                running
                and sim.get_world().is_window_initialized()
                and (max_frames is None or frames < max_frames)
            ):
                started = time.monotonic()
                changed = frames == 0
                # Queue native key edges so a short tap survives a slow render
                # frame. A newly opened viewport may not dispatch control
                # callbacks before its first click, so retain held-key edges too.
                down = {key for key in preview_keys if sim.get_world().key_state(key)}
                for key in down - polled:
                    if key not in held:
                        held.add(key)
                        pressed.append(key)
                held.difference_update(polled - down)
                polled = down
                while pressed:
                    key = pressed.popleft()
                    if key in (
                        InputKey.SCANCODE_Q.value,
                        InputKey.SCANCODE_ESCAPE.value,
                    ):
                        running = False
                    elif key == InputKey.SCANCODE_N.value:
                        selected = (selected + 1) % len(candidates)
                        switches += 1
                        changed = True
                    elif key == InputKey.SCANCODE_SPACE.value:
                        paused, changed = not paused, True
                if not running:
                    break
                if not paused:
                    elapsed += started - previous
                previous = started
                elapsed %= duration
                original_frame, expanded_frames = show_frame(elapsed)
                frames += 1
                if frame_observer is not None:
                    frame_observer(env, frames, selected, elapsed)
                if not sim.get_world().is_window_initialized():
                    break
                if changed:
                    publish()
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - started)))
            result.update(
                state="closed",
                frames=frames,
                switches=switches,
                physics_steps=sim.physics_step_index - physics_steps_before,
            )
            atomic_json(output / "status.json", result)
    return result
