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
from collections.abc import Iterable
import uuid

import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.sensors import BaseSensor, SensorCfg

__all__: list[str] = []

_CAMERA_NAME = "__embodichain_web_viewer"


class _WebViewer:
    """Own an observer camera and HTTP routes, borrowing the manager's World."""

    def __init__(self, sim: SimulationManager) -> None:
        import dexsim

        existing = []
        for sensor in sim._sensors.values():
            _validate_sensor(sensor.cfg, existing)
            existing.append(sensor)
        self.sim = sim
        self.cfg = sim.sim_config.visualization.web
        self.owner = threading.get_ident()
        self.run_id = uuid.uuid4().hex
        self.camera = None
        self.group = None
        self.application = None
        self.gizmo = None
        self.error: str | None = None
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
            self.group = sim.get_world().create_camera_group(
                [self.cfg.width, self.cfg.height],
                1,
                True,
                dexsim.render.ViewFlags.COLOR,
            )
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
        }

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
                self.camera.look_at(self.cfg.eye, self.cfg.target)
            elif action == "render":
                pass  # Deferred until after state synchronization, never inside HTTP handling.
            elif action == "pause":
                if type(body.get("enabled")) is not bool:
                    raise ValueError("enabled must be boolean.")
                self.paused = body["enabled"]
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
        if self.sim.spawn_result is None or not (force or self.dirty or due):
            return False
        started = time.monotonic()
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
        self.group = None
        self.arena = None
        self.closed = True


def _validate_sensor(cfg: SensorCfg, existing: Iterable[BaseSensor]) -> None:
    """Reject unsupported native auto-stream sources before camera allocation."""
    if cfg.sensor_type == "ContactSensor":
        return
    if cfg.sensor_type != "Camera" or not cfg.enable_color:
        raise ValueError(
            "dexsim-web currently supports RGB Camera sensors only (plus ContactSensor)."
        )
    if any(
        type(v) is not int or not 32 <= v <= 4096 or v % 2
        for v in (cfg.width, cfg.height)
    ):
        raise ValueError("dexsim-web sensor dimensions must be even and in [32, 4096].")
    if sum(sensor.cfg.sensor_type == "Camera" for sensor in existing) >= 15:
        raise ValueError(
            "dexsim-web supports at most 15 sensor cameras plus its observer."
        )
