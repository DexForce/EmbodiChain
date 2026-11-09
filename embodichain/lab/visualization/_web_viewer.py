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

"""Creator-thread native WebRTC observation view; no scene export or stepping."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING
from dataclasses import asdict
from copy import deepcopy
import uuid

import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from .protocol import JointControlProvider

__all__: list[str] = []

_CAMERA_NAME = "__embodichain_web_viewer"


class _WebViewer:
    """Own an observer camera and HTTP routes, borrowing the manager's World."""

    def __init__(self, sim: SimulationManager) -> None:
        import dexsim

        self.sim = sim
        self.cfg = deepcopy(sim.sim_config.visualization.web)
        for name in ("width", "height"):
            if getattr(self.cfg, name) is None:
                value = getattr(sim.sim_config, name)
                setattr(self.cfg, name, value + value % 2)
        self.cfg.__post_init__()
        self.owner = threading.get_ident()
        self.run_id = uuid.uuid4().hex
        self.camera = None
        self.group = None
        self.application = None
        self.gizmo = None
        self.error: str | None = None
        self.title = "EmbodiChain"
        self.orbit = np.asarray(self.cfg.target, dtype=np.float32)
        self.home_view = (self.cfg.eye, self.cfg.target, (0, 0, 1))
        self.pan_enabled = True
        self.controls: list[object] = []
        self.keys: set[int] = set()
        self.key_deadline = 0.0
        self.joint_provider: JointControlProvider | None = None
        self.joint_commands = {}
        self.replay = {"step": 0, "max_step": 0, "visible": False}
        self.replay_command: int | None = None
        self.sequence = 0
        self.frames = 0
        self.render_seconds = 0.0
        self.next_render = 0.0
        self.dirty = True
        self.paused = False
        self.closed = False
        self.arena = sim.get_env()
        self.endpoint = f"http://127.0.0.1:{self.cfg.port}/"
        try:
            # Explicitly separate this group from sensor groups even at equal sizes.
            self.group = getattr(sim, "_web_camera_group", None)
            if self.group is None:
                self.group = sim.get_world().create_camera_group(
                    [self.cfg.width, self.cfg.height],
                    1,
                    True,
                    dexsim.render.ViewFlags.COLOR,
                )
                sim._web_camera_group = self.group
            self.camera = self.arena.create_camera(
                _CAMERA_NAME,
                self.cfg.width,
                self.cfg.height,
                True,
                dexsim.render.ViewFlags.COLOR,
                self.group,
            )
            self.camera.set_visual_overlay_enabled(True)
            self.camera.enable_control()
            self.camera.look_at(self.cfg.eye, self.cfg.target)
            assets_dir = Path(__file__).with_name("web_assets")
            assets = {
                route: (mime, (assets_dir / filename).read_bytes())
                for route, filename, mime in (
                    ("/", "index.html", "text/html; charset=utf-8"),
                    ("/app.js", "app.js", "text/javascript; charset=utf-8"),
                    ("/style.css", "style.css", "text/css; charset=utf-8"),
                )
            }
            self.application = dexsim.WebApplication(sim.get_world(), assets)
        except BaseException:
            self.close()
            raise

    def _assert_owner(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("WebRTC viewer must run on the World creator thread.")

    def _world_gizmo_pose(self, gizmo: object) -> list[list[float]]:
        pose = np.asarray(gizmo.get_control_pose()[0].tolist(), dtype=np.float32)
        pose[:3, 3] += np.asarray(self.sim.arena_offsets[0].tolist())
        return pose.tolist()

    def state(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "revision": self.sim._visualization_topology_revision,
            "camera_name": _CAMERA_NAME,
            "frames": self.frames,
            "paused": self.paused,
            "allow_commands": self.sim.sim_config.visualization.allow_commands,
            "gizmo": self.gizmo is not None,
            "camera_pose": np.asarray(self.camera.get_world_pose()).tolist(),
            "assets": self.sim.asset_uids,
            "error": self.error,
            "title": self.title,
            "pan_enabled": self.pan_enabled,
            "recording": self.sim.is_window_recording(),
            "camera_intrinsic": np.asarray(self.camera.get_intrinsic()).tolist(),
            "gizmos": (
                []
                if self.sim.spawn_result is None
                else [
                    dict(id=key, pose=self._world_gizmo_pose(gizmo))
                    for key, gizmo in self.sim.get_gizmo_items()
                    if gizmo.is_visible()
                ]
            ),
            "joints": (
                []
                if self.joint_provider is None
                else [
                    asdict(spec) for spec in self.joint_provider.joint_control_specs()
                ]
            ),
            "joint_states": (
                []
                if self.joint_provider is None
                else [
                    asdict(state)
                    for state in self.joint_provider.joint_control_states()
                ]
            ),
            "replay": self.replay,
        }

    @property
    def exporter(self) -> _WebViewer:
        """Identity boundary used by shared joint and replay controls."""
        return self

    @property
    def scene_revision(self) -> int:
        return self.sim._visualization_topology_revision

    @property
    def is_running(self) -> bool:
        return not self.closed

    def set_look_at(self, eye: object, look_at: object, up: object = (0, 0, 1)) -> None:
        self._assert_owner()
        self.camera.look_at(eye, look_at, up)
        self.home_view = (np.array(eye), np.array(look_at), np.array(up))
        self.orbit = np.array(look_at, dtype=np.float32)
        self.dirty = True

    def set_camera_orbit_pan_enabled(self, enabled: bool) -> None:
        self.pan_enabled = bool(enabled)

    def translate_camera_orbit(self, delta: object) -> None:
        delta = np.asarray(delta, dtype=np.float32)
        self.set_look_at(self.get_pose_matrix()[:3, 3] + delta, self.orbit + delta)

    def get_pose_matrix(self) -> np.ndarray:
        return np.asarray(self.camera.get_world_pose())

    def set_window_title(self, title: str) -> None:
        self.title = str(title)

    def native(self) -> _WebViewer:
        # The common consumers need key state, not an SDL window pointer.
        return self

    def key_state(self, key: object) -> bool:
        return (
            self.sim.sim_config.visualization.allow_commands and int(key) in self.keys
        )

    def add_input_control(self, control: object) -> None:
        self.controls.append(control)

    def remove_input_control(self, control: object) -> None:
        if control in self.controls:
            self.controls.remove(control)

    def set_joint_control_provider(self, provider: JointControlProvider | None) -> None:
        self.joint_provider = provider
        self.joint_commands.clear()

    def drain_joint_control_commands(self) -> tuple:
        commands = tuple(self.joint_commands.values())
        self.joint_commands.clear()
        return commands

    def publish_replay_control(
        self, *, step: int, max_step: int, visible: bool = True
    ) -> None:
        if not 0 <= step <= max_step:
            raise ValueError("Replay step must satisfy 0 <= step <= max_step.")
        self.replay = dict(step=step, max_step=max_step, visible=visible)

    def drain_replay_control_command(self) -> int | None:
        command, self.replay_command = self.replay_command, None
        return command

    def _request(self, path: str, raw: str) -> tuple[int, str]:
        try:
            if path == "/api/state":
                return 200, json.dumps(self.state())
            if path != "/api/command":
                return 404, json.dumps({"error": "Unknown route"})
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Command must be an object.")
            if (
                body.get("run_id") != self.run_id
                or body.get("revision") != self.sim._visualization_topology_revision
            ):
                return 409, json.dumps(
                    {"error": "Scene changed; refresh before controlling it."}
                )
            action = body.get("action")
            if action == "reset_view":
                self.camera.look_at(*self.home_view)
            elif action == "render":
                pass  # Deferred until after state synchronization, never inside HTTP handling.
            elif action == "pause":
                if type(body.get("enabled")) is not bool:
                    raise ValueError("enabled must be boolean.")
                self.paused = body["enabled"]
            elif action == "key":
                if not self.sim.sim_config.visualization.allow_commands:
                    return 403, json.dumps(
                        {"error": "Keyboard scene controls are disabled."}
                    )
                key = body.get("key")
                if (
                    type(key) is not int
                    or not 0 <= key < 512
                    or type(body.get("down")) is not bool
                ):
                    raise ValueError("Expected a scancode and boolean key state.")
                changed = (key not in self.keys) if body["down"] else (key in self.keys)
                self.key_deadline = time.monotonic() + 1.0
                if body["down"]:
                    self.keys.add(key)
                else:
                    self.keys.discard(key)
                if changed and self.sim.sim_config.visualization.allow_commands:
                    import dexsim

                    if body["down"] and key == dexsim.types.InputKey.SCANCODE_I.value:
                        for _, gizmo in self.sim.get_gizmo_items():
                            gizmo.toggle_visibility()
                    method = "on_key_down" if body["down"] else "on_key_up"
                    for control in tuple(self.controls):
                        getattr(control, method)(key)
            elif action in {"target", "joint"}:
                if not self.sim.sim_config.visualization.allow_commands:
                    return 403, json.dumps({"error": "Scene editing is disabled."})
                if action == "target":
                    gizmo = dict(self.sim.get_gizmo_items()).get(body.get("id"))
                    if gizmo is None:
                        raise ValueError("Unknown Gizmo.")
                    pose = np.asarray(body.get("pose"), dtype=np.float32)
                    if (
                        pose.shape != (4, 4)
                        or not np.isfinite(pose).all()
                        or not np.allclose(pose[3], [0, 0, 0, 1])
                    ):
                        raise ValueError("Expected a finite rigid pose.")
                    rotation = pose[:3, :3]
                    if (
                        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                        or np.linalg.det(rotation) < 0
                    ):
                        raise ValueError("Expected an orthonormal rotation.")
                    pose[:3, 3] -= np.asarray(self.sim.arena_offsets[0].tolist())
                    if not gizmo.request_local_pose(pose, source_id="web"):
                        return 409, json.dumps(
                            {"error": "Gizmo is controlled by another source."}
                        )
                else:
                    from .protocol import JointControlCommand

                    specs = (
                        ()
                        if self.joint_provider is None
                        else self.joint_provider.joint_control_specs()
                    )
                    if body.get("id") not in {spec.control_id for spec in specs}:
                        raise ValueError("Unknown joint control.")
                    self.sequence += 1
                    self.joint_commands[body["id"]] = JointControlCommand(
                        self.run_id,
                        self.scene_revision,
                        self.sequence,
                        "web",
                        body["id"],
                        float(body.get("value")),
                    )
            elif action == "seek":
                if not self.sim.sim_config.visualization.allow_commands:
                    return 403, json.dumps({"error": "Replay editing is disabled."})
                step = body.get("step")
                if (
                    not self.replay["visible"]
                    or type(step) is not int
                    or not 0 <= step <= self.replay["max_step"]
                ):
                    raise ValueError("Invalid replay step.")
                self.replay_command = step
            elif action == "gizmo":
                if not self.sim.sim_config.visualization.allow_commands:
                    return 403, json.dumps({"error": "Scene editing is disabled."})
                if type(body.get("enabled")) is not bool:
                    raise ValueError("enabled must be boolean.")
                if body["enabled"]:
                    import dexsim

                    config = dexsim.interaction.EntityGizmoConfig()
                    config.attach_on_click = True
                    config.selection_cache = False
                    self.gizmo = self.sim.enable_entity_gizmo(config)
                    self.gizmo.attach_camera(self.camera)
                else:
                    self.sim.disable_entity_gizmo()
                    self.gizmo = None
            else:
                raise ValueError("Unknown command.")
            self.dirty = True
            return 200, json.dumps(self.state())
        except (ValueError, TypeError, KeyError) as error:
            return 400, json.dumps({"error": str(error)})
        except RuntimeError as error:
            return 503, json.dumps({"error": str(error)})

    def capture(self, *, force: bool = False) -> bool:
        """Pump HTTP and render only this observer; never advance physics."""
        import dexsim

        self._assert_owner()
        if self.closed:
            return False
        self.application.process(self._request)
        now = time.monotonic()
        due = not self.paused and now >= self.next_render
        # Applying inputs is independent of frame rate and physical stepping.
        if self.keys and now > self.key_deadline:
            for key in tuple(self.keys):
                for control in tuple(self.controls):
                    control.on_key_up(key)
            self.keys.clear()
        if self.sim.spawn_result is not None:
            self.sim._enable_default_entity_gizmo()
            self.sim.physics.process_interaction()
            update_joints = getattr(self.joint_provider, "update", None)
            if self.joint_commands and callable(update_joints):
                update_joints()
            self.sim.update_gizmos()
            # Commands may write poses after observation cameras consumed this
            # frame. Re-publish before drawing the observer instead of reusing
            # the read-only frame's pre-command synchronization cache.
            if self.sim._render_state_published is not None:
                self.sim._render_state_published = False
        if not (force or self.dirty or due):
            return False
        started = time.monotonic()
        if self.sim.spawn_result is not None:
            self.sim._ensure_render_state()
        self.camera.render(dexsim.render.ViewFlags.COLOR, dexsim.render.ViewFlags.NULL)
        self.frames += 1
        self.render_seconds += time.monotonic() - started
        self.next_render = time.monotonic() + 1.0 / self.cfg.fps
        self.dirty = False
        return True

    def close(self) -> None:
        """Detach application before cameras; listener remains World-owned."""
        self._assert_owner()
        if self.closed:
            return
        if self.application is not None:
            self.application.close()
            self.application = None
        if self.gizmo is not None:
            self.sim.disable_entity_gizmo()
            self.gizmo = None
        if self.camera is not None:
            self.camera.disable_control()
            self.arena.remove_camera(_CAMERA_NAME)
            self.camera = None
        for _, gizmo in self.sim.get_gizmo_items():
            gizmo.cancel_interaction("web")
        self.keys.clear()
        self.controls.clear()
        self.joint_provider = None
        self.joint_commands.clear()
        self.replay_command = None
        self.group = None
        self.arena = None
        self.closed = True
