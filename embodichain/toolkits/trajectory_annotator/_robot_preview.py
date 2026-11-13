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

"""Lazy, simulator-free URDF ghosts with one owner and a latest-time mailbox."""

from __future__ import annotations

from functools import partial
import math
from pathlib import Path
import socket
import threading
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

__all__ = []


class RobotGhostPreview:
    """Own one loopback Viser server until its fitting result is replaced.

    Only bounded drawing samples are retained. Model loading, scene mutations
    and teardown all run on the same worker. Cursor updates replace one pending
    time; they cannot queue a backlog while the user scrubs the video.
    """

    def __init__(self, preview: dict[str, Any]) -> None:
        self._condition = threading.Condition()
        self._closed = False
        self._dirty = True
        self._time = 0.0
        self._options: dict[str, Any] = dict(original=True, ghost=True)
        report = preview["report"]
        self._offset = float(report["time_offset"])
        self._times = np.asarray(preview["timestamps"], dtype=np.float64)
        self._names = tuple(report["joint_names"][: len(preview["joint_names"])])
        self._positions = {
            name: np.asarray(preview[f"{name}_qpos"], dtype=np.float64)
            for name in ("original", "native", "patched", "proposed")
        }
        self._profile = report.get("robot_profile", {})
        self._digests = report.get("robot_model_sha256", {})
        self._options["series"] = "patched" if report["accepted"] else "proposed"
        self._accepted = bool(report["accepted"])
        skeleton = np.asarray(preview["original_links"], dtype=np.float64)
        # A serial chain's first node may be the world/base origin, far below
        # the shoulders. Frame the moving links for a useful arm comparison.
        points = (skeleton[..., 1:, :] if skeleton.shape[-2] > 2 else skeleton).reshape(
            -1, 3
        )
        lower, upper = points.min(axis=0), points.max(axis=0)
        self._look_at = (lower + upper) / 2
        self._radius = max(0.4, float(np.linalg.norm(upper - lower)) / 2)
        self._state: dict[str, Any] = dict(phase="loading", port=None, error=None)
        self._thread = threading.Thread(
            target=self._run, name="trajectory-robot-preview", daemon=True
        )
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        """Return model readiness and the last displayed physical sample."""
        with self._condition:
            return dict(self._state)

    def set_time(self, time_seconds: float) -> None:
        """Replace the pending video time without waiting for scene updates."""
        if (
            type(time_seconds) not in (float, int)
            or not math.isfinite(time_seconds)
            or time_seconds < 0
        ):
            raise ValueError("Video time must be finite and nonnegative.")
        with self._condition:
            if self._closed:
                raise ValueError("Robot preview is closed.")
            self._time = float(time_seconds)
            self._dirty = True
            self._condition.notify()

    def close(self) -> None:
        """Join the owner after closing its Viser server and websocket threads."""
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()

    def _change(self, **values: Any) -> None:
        # Viser GUI callbacks only publish options; they never mutate the scene.
        with self._condition:
            self._options.update(values)
            self._dirty = True
            self._condition.notify()

    def _models(self) -> list[tuple[Any, np.ndarray, list[int]]]:
        import yourdfpy
        from embodichain.data_pipeline.skill_fitting import file_digest

        if not self._profile.get("solvers"):
            raise ValueError("Robot profile has no URDF model for mesh preview.")
        loaded: dict[Path, Any] = {}
        models: list[tuple[Any, np.ndarray, list[int]]] = []
        for part, solver in self._profile["solvers"].items():
            path = Path(solver["urdf_path"])
            if path not in loaded:
                expected = self._digests.get(part)
                if expected and file_digest(path) != expected:
                    raise ValueError(
                        "Robot URDF changed after fitting; refit before previewing."
                    )
                # yourdfpy otherwise warns and silently omits missing meshes.
                for element in ET.parse(path).findall(".//visual/geometry/mesh"):
                    mesh = Path(
                        yourdfpy.filename_handler_magic(
                            element.attrib["filename"], dir=str(path.parent)
                        )
                    )
                    if not mesh.is_file():
                        raise ValueError(
                            f"机器人可视模型缺失：{mesh}。请将 URDF 引用的 visual/meshes 一并放在模型目录。"
                        )
                loaded[path] = yourdfpy.URDF.load(
                    path,
                    load_meshes=True,
                    load_collision_meshes=False,
                    build_scene_graph=True,
                    build_collision_scene_graph=False,
                    filename_handler=partial(
                        yourdfpy.filename_handler_magic, dir=str(path.parent)
                    ),
                )
            model = loaded[path]
            missing = set(model.actuated_joint_names) - set(self._names)
            if missing:
                raise ValueError(
                    f"Joint recording lacks model joints: {sorted(missing)}"
                )
            base = np.asarray(self._profile.get("base_poses", {}).get(part, np.eye(4)))
            mount = base @ np.linalg.inv(model.get_transform(solver["root_link_name"]))
            if any(
                previous is model and np.allclose(mount, transform, atol=1e-6)
                for previous, transform, _ in models
            ):
                continue
            models.append(
                (
                    model,
                    mount,
                    [self._names.index(name) for name in model.actuated_joint_names],
                )
            )
        return models

    def _run(self) -> None:
        server = None
        try:
            import viser
            from viser.extras import ViserUrdf
            from viser.transforms import SO3

            models = self._models()
            # Viser 1.0.21 reports the requested port, rather than getsockname(),
            # when passed zero. Select an ephemeral loopback port explicitly.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            server = viser.ViserServer(
                host="127.0.0.1", port=port, label="轨迹机械臂叠加", verbose=False
            )
            server.gui.configure_theme(
                control_layout="collapsible",
                control_width="small",
                dark_mode=True,
                show_logo=False,
                show_share_button=False,
            )
            server.scene.set_up_direction("+z")
            server.scene.add_grid("/ground", width=3, height=3, cell_size=0.1)
            # The saved FK skeleton bounds provide the calibrated camera target.
            server.initial_camera.position = self._look_at + self._radius * np.array(
                [1.5, -1.4, 0.8]
            )
            server.initial_camera.look_at = self._look_at
            choices = {
                ("连续回填" if self._accepted else "连续回填（保留原始）"): "patched",
                "原生技能（未拼接）": "native",
            }
            if not self._accepted:
                choices["校正候选（未通过）"] = "proposed"
            initial = next(
                label
                for label, name in choices.items()
                if name == self._options["series"]
            )
            choice = server.gui.add_dropdown(
                "青色虚影轨迹", options=list(choices), initial_value=initial
            )
            original = server.gui.add_checkbox("原始机械臂实体", initial_value=True)
            ghost = server.gui.add_checkbox("轨迹虚影", initial_value=True)
            choice.on_update(lambda _: self._change(series=choices[choice.value]))
            original.on_update(lambda _: self._change(original=original.value))
            ghost.on_update(lambda _: self._change(ghost=ghost.value))
            robots = []
            for index, (model, mount, columns) in enumerate(models):
                root = server.scene.add_frame(
                    f"/model_{index}",
                    show_axes=False,
                    position=mount[:3, 3],
                    wxyz=SO3.from_matrix(mount[:3, :3]).wxyz,
                )
                real = ViserUrdf(
                    server,
                    model,
                    root_node_name=f"{root.name}/original",
                    mesh_color_override=(0.56, 0.61, 0.70),
                )
                shadow = ViserUrdf(
                    server,
                    model,
                    root_node_name=f"{root.name}/ghost",
                    mesh_color_override=(0.25, 0.92, 0.82, 0.35),
                )
                robots.append((real, shadow, columns))
            with self._condition:
                self._state.update(
                    phase="ready", port=server.get_port(), model_count=len(models)
                )
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._dirty)
                    if self._closed:
                        break
                    timestamp, options = self._time + self._offset, dict(self._options)
                    self._dirty = False
                right = int(np.searchsorted(self._times, timestamp))
                index = min(right, len(self._times) - 1)
                if index > 0 and abs(self._times[index - 1] - timestamp) < abs(
                    self._times[index] - timestamp
                ):
                    index -= 1
                with server.atomic():
                    for real, shadow, columns in robots:
                        real.update_cfg(self._positions["original"][index, columns])
                        shadow.update_cfg(
                            self._positions[options["series"]][index, columns]
                        )
                        real.show_visual = options["original"]
                        shadow.show_visual = options["ghost"]
                with self._condition:
                    self._state.update(
                        sample_index=index,
                        sample_time_seconds=float(self._times[index]),
                        video_time_seconds=float(self._times[index] - self._offset),
                        ghost_series=options["series"],
                    )
        except Exception as error:
            with self._condition:
                self._state.update(phase="error", error=str(error))
        finally:
            if server is not None:
                server.stop()
            with self._condition:
                if self._closed:
                    self._state.update(phase="closed", port=None)
