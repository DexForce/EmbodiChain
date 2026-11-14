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

"""Creator-thread WebRTC video and read-only initialization information."""

from __future__ import annotations

import json
import ctypes
from copy import deepcopy
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING
import uuid

import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__: list[str] = []

_CAMERA_NAME = "__embodichain_web_viewer"
# The SDK's empty selector permits every camera. Reserve a different name for
# this deny policy: sensor names end in _viewN; observer and recorder names are
# distinct. Direct native camera callers must also leave this name unused.
_NO_INPUT_CAMERA_NAME = "__embodichain_web_input_disabled"


def _require_webrtc_runtime() -> None:
    """Check the selected native streaming library before allocating a World."""
    import dexsim

    library = os.environ.get("DEXSIM_WEBRTC_LIBRARY") or str(
        Path(dexsim.__file__).parent / "lib" / "libdexsim_streaming.so"
    )
    try:
        ctypes.CDLL(library)
    except OSError as error:
        raise RuntimeError(
            f"dexsim-web cannot load the DexSim streaming runtime {library!r}: {error}. "
            "A WebApplication Python API alone does not qualify the SDK. "
            "Install a loadable WebRTC-enabled DexSim artifact."
        ) from error


class _WebViewer:
    """Borrow the existing World for video and a read-only information page."""

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
        self.error: str | None = None
        self.title = "EmbodiChain"
        self.orbit = np.asarray(self.cfg.target, dtype=np.float32)
        self.frames = 0
        self.render_seconds = 0.0
        self.next_render = 0.0
        self.dirty = True
        self.closed = False
        self.arena = sim.get_env()
        self.endpoint = f"http://127.0.0.1:{self.cfg.port}/"
        try:
            # Observer buffers never share a group with task sensors.
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
            # OptiX observers start closed. Explicitly open every renderer's
            # observer, keeping browser input disabled for this information page.
            self.camera.open_camera()
            self.camera.disable_control()
            self.camera.set_visual_overlay_enabled(True)
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
        """Read the same initialization rows used by terminal summaries."""
        from embodichain.lab.sim._startup_summary import scene_rows, simulation_rows

        self._assert_owner()
        summary = None
        if self.sim.sim_config.startup_summary != "off":
            summary = getattr(self.sim, "_web_initialization_summary", None)
            if summary is None:
                summary = {
                    "title": "Simulation initialized",
                    "rows": simulation_rows(self.sim) + scene_rows(self.sim),
                }
        return {
            "run_id": self.run_id,
            "revision": self.sim._visualization_topology_revision,
            "camera_name": _CAMERA_NAME,
            "frames": self.frames,
            "read_only": True,
            "error": self.error,
            "title": self.title,
            "summary": summary,
        }

    def set_look_at(self, eye: object, look_at: object, up: object = (0, 0, 1)) -> None:
        """Set the observer from its Python host on the World creator thread."""
        self._assert_owner()
        self.camera.look_at(eye, look_at, up)
        self.orbit = np.array(look_at, dtype=np.float32)
        self.dirty = True

    def set_camera_orbit_pan_enabled(self, enabled: bool) -> None:
        """Keep viewport compatibility; the information page has no input."""

    def translate_camera_orbit(self, delta: object) -> None:
        """Translate the observation camera from its Python host."""
        delta = np.asarray(delta, dtype=np.float32)
        self.set_look_at(self.get_pose_matrix()[:3, 3] + delta, self.orbit + delta)

    def get_pose_matrix(self) -> np.ndarray:
        """Return the observation camera pose without acquiring task images."""
        self._assert_owner()
        return np.asarray(self.camera.get_world_pose())

    def set_window_title(self, title: str) -> None:
        """Set the information-page title from its Python host."""
        self.title = str(title)

    def native(self) -> _WebViewer:
        """Expose the read-only key-state adapter for shared viewport callers."""
        return self

    def key_state(self, key: object) -> bool:
        """Return False because the browser never dispatches application keys."""
        return False

    def add_input_control(self, control: object) -> None:
        """Keep shared viewport compatibility without registering browser input."""

    def remove_input_control(self, control: object) -> None:
        """Keep shared viewport compatibility without registering browser input."""

    def _request(self, path: str, raw: str) -> tuple[int, str]:
        self._assert_owner()
        if path == "/api/command":
            return 403, json.dumps({"error": "The Web page is read-only."})
        if path != "/api/state":
            return 404, json.dumps({"error": "Unknown route"})
        try:
            return 200, json.dumps(self.state())
        except RuntimeError as error:
            return 503, json.dumps({"error": str(error)})

    def capture(self, *, force: bool = False) -> bool:
        """Pump information requests and render only the observer camera."""
        import dexsim

        self._assert_owner()
        if self.closed:
            return False
        self.application.process(self._request)
        if self.error is not None:
            return False
        if not (force or self.dirty or time.monotonic() >= self.next_render):
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
        """Detach application before cameras; the listener stays World-owned."""
        self._assert_owner()
        if self.closed:
            return
        if self.application is not None:
            self.application.close()
            self.application = None
        if self.camera is not None:
            self.camera.disable_control()
            self.camera.close_camera()
            # The SDK returns a borrowed group and deletes it when its final
            # camera is removed. Reopening must allocate a fresh group.
            self.sim._web_camera_group = None
            self.arena.remove_camera(_CAMERA_NAME)
            self.camera = None
        self.group = None
        self.arena = None
        self.closed = True
