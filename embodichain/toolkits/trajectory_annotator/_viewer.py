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

"""Session-owned, simulator-free Viser playback of two full joint recordings."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
from pathlib import Path
import socket
import threading
from typing import Any

import numpy as np

__all__ = []


def _load_models(report: dict[str, Any]) -> list[tuple[Any, Any, np.ndarray]]:
    """Build two independent visual scenes in each URDF's calibrated world frame."""
    import yourdfpy

    profile = report["robot_profile"]
    names = set(profile["joint_names"])
    groups: dict[Path, list[str]] = {}
    for part, solver in profile["solvers"].items():
        path = Path(solver["urdf_path"]).expanduser().resolve()
        groups.setdefault(path, []).append(part)
        expected = report.get("robot_model_sha256", {}).get(part)
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("URDF 在拟合后发生变化，请重新拟合后查看模型。")
    models, mapped = [], set()
    for path, parts in groups.items():
        description = yourdfpy.URDF.load(
            path,
            build_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
        )
        # Recorded mimic columns are measured independently: display their exact
        # qpos. Only unrecorded mimic joints follow the URDF's coupling rule.
        for name in names & description.joint_map.keys():
            description.joint_map[name].mimic = None
            mapped.add(name)
        model = yourdfpy.URDF(
            robot=description.robot,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
            load_collision_meshes=False,
        )
        if model.scene is None or not model.scene.geometry:
            raise ValueError(f"URDF 没有可显示的 visual mesh：{path}")
        bases = []
        for part in parts:
            root = profile["solvers"][part].get("root_link_name") or model.base_link
            base = np.asarray(profile["base_poses"].get(part, np.eye(4)), dtype=float)
            bases.append(base @ np.linalg.inv(model.get_transform(root)))
        if any(
            not np.allclose(base, bases[0], atol=1e-5, rtol=0) for base in bases[1:]
        ):
            raise ValueError(
                "同一 URDF 的机械臂基座标定不一致，无法叠加完整机器人模型。"
            )
        # Rebuild the second scene from the parsed robot description. URDF's
        # internal graph caches cannot be deep-copied, and sharing one graph
        # would make the two configurations overwrite each other.
        patched = yourdfpy.URDF(
            robot=deepcopy(model.robot),
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
            load_collision_meshes=False,
        )
        models.append((model, patched, bases[0]))
    if missing := names - mapped:
        raise ValueError(f"URDF 缺少记录中的关节：{sorted(missing)}")
    return models


class TrajectoryViewer:
    """Own one Viser server and a single latest-time slot on a private worker.

    Args:
        arrays: Immutable full-resolution fit output, including both qpos arrays.
        report: Fit provenance containing the resolved robot profile and clock.
    """

    def __init__(self, arrays: dict[str, np.ndarray], report: dict[str, Any]) -> None:
        self._times = arrays["timestamps"]
        self._qpos = (arrays["original_qpos"], arrays["patched_qpos"])
        self._report = report
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._closed = False
        self._pending: float | None = None
        self._state: dict[str, Any] = {"phase": "loading", "port": None, "error": None}
        self._thread = threading.Thread(
            target=self._run, name="trajectory-viser", daemon=True
        )

    def start(self) -> dict[str, Any]:
        """Load models on the owning worker and return ready/error metadata."""
        self._thread.start()
        self._ready.wait()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return server availability and the last displayed video timestamp."""
        with self._condition:
            return dict(self._state)

    def set_time(self, time_seconds: float) -> dict[str, Any]:
        """Queue the newest video time, replacing any unrendered request.

        Args:
            time_seconds: Finite video-relative seconds, before the fit offset.

        Returns:
            Current server state; rendering proceeds on its owning worker.
        """
        if isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float)):
            raise ValueError("time_seconds 必须是有限秒数。")
        if not np.isfinite(time_seconds):
            raise ValueError("time_seconds 必须是有限秒数。")
        with self._condition:
            if self._closed or self._state["phase"] != "ready":
                raise ValueError(self._state["error"] or "Viser 模型尚未就绪或已关闭。")
            self._pending = float(time_seconds)
            self._condition.notify()
            return dict(self._state)

    def _configuration(self, time_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        # Sample the original joint clock, independently of chart decimation.
        time = time_seconds + self._report["time_offset"]
        right = int(np.searchsorted(self._times, time, side="right"))
        right = min(max(right, 1), len(self._times) - 1)
        left = max(0, right - 1)
        span = self._times[right] - self._times[left]
        weight = 0.0 if span == 0 else np.clip((time - self._times[left]) / span, 0, 1)
        return tuple(q[left] * (1 - weight) + q[right] * weight for q in self._qpos)

    def _run(self) -> None:
        server = None
        try:
            import viser
            from viser.extras import ViserUrdf
            from viser.transforms import SO3

            models = _load_models(self._report)
            # Viser 1.0.21 reports port=0 literally, so choose a concrete free port;
            # its server retries a nearby port if this short handoff races a bind.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
            server.gui.configure_theme(
                dark_mode=True,
                show_logo=False,
                show_share_button=False,
                control_layout="floating",
                control_width="small",
            )
            server.scene.set_up_direction("+z")
            robots = []
            bounds = []
            initial = float(
                self._times[self._report["start"]] - self._report["time_offset"]
            )
            original_qpos, patched_qpos = self._configuration(initial)
            for index, (model, patched, base) in enumerate(models):
                for name, source, color in (
                    ("original", model, None),
                    ("patched", patched, (0.16, 0.47, 1.0, 0.35)),
                ):
                    root = f"/{name}/robot_{index}"
                    server.scene.add_frame(
                        root,
                        show_axes=False,
                        position=base[:3, 3],
                        wxyz=SO3.from_matrix(base[:3, :3]).wxyz,
                    )
                    robot = ViserUrdf(
                        server,
                        source,
                        root_node_name=root,
                        mesh_color_override=color,
                    )
                    columns = [
                        (joint, column)
                        for column, joint in enumerate(self._report["joint_names"])
                        if joint in model.joint_map
                    ]
                    robots.append((name, robot, columns))
                    qpos = original_qpos if name == "original" else patched_qpos
                    robot.update_cfg(
                        {joint: float(qpos[col]) for joint, col in columns}
                    )
                    # Fit the view to the loaded recording's pose, rather than
                    # the usually outstretched zero configuration of the URDF.
                    corners = np.array(np.meshgrid(*source.scene.bounds.T)).T.reshape(
                        -1, 3
                    )
                    bounds.extend(corners @ base[:3, :3].T + base[:3, 3])
            bounds = np.asarray(bounds)
            center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
            radius = max(0.2, np.linalg.norm(np.ptp(bounds, axis=0)) / 2)
            server.initial_camera.look_at = center
            server.initial_camera.position = (
                center + np.array([1.6, -2.0, 1.2]) * radius
            )
            server.initial_camera.near = 0.005
            server.initial_camera.fov = np.deg2rad(45)

            def render(time: float) -> None:
                original, patched = self._configuration(time)
                with server.atomic():
                    for name, robot, columns in robots:
                        qpos = original if name == "original" else patched
                        robot.update_cfg(
                            {joint: float(qpos[col]) for joint, col in columns}
                        )

            with self._condition:
                self._state.update(
                    phase="ready", port=server.get_port(), time_seconds=initial
                )
                self._ready.set()
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed or self._pending is not None
                    )
                    if self._closed:
                        break
                    time, self._pending = self._pending, None
                render(time)
                with self._condition:
                    self._state["time_seconds"] = time
        except Exception as error:
            with self._condition:
                self._state.update(
                    phase="error", error=f"Viser 模型加载或更新失败：{error}"
                )
        finally:
            if server is not None:
                server.stop()
            with self._condition:
                if self._closed:
                    self._state["phase"] = "closed"
                self._ready.set()

    def close(self) -> None:
        """Stop the worker and its HTTP/WebSocket server before releasing qpos."""
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._thread.ident is not None:
            self._thread.join()
        with self._condition:
            self._state["phase"] = "closed"
