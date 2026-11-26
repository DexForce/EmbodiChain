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

"""One background scene-import job independent of video decoding and fitting."""

from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any

from ._models import ConflictError

__all__ = []


class ScenePreparationSession:
    """Own a bounded worker while collision cache entries survive the session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._state: dict[str, Any] = dict(
            phase="idle", message="加载场景后准备碰撞网格。"
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe progress/result snapshot."""
        with self._lock:
            return deepcopy(self._state)

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start one scene import without occupying the request or decoder worker."""
        if (
            not isinstance(payload, dict)
            or payload.keys() != {"scene", "joints"}
            or any(not isinstance(value, str) for value in payload.values())
        ):
            raise ValueError("场景加载需要 scene、joints 路径字符串。")
        with self._lock:
            if self._closed:
                raise ConflictError("场景加载会话已关闭。")
            if self._state["phase"] == "running":
                if self._state["inputs"] == payload:
                    return self.snapshot()
                raise ConflictError("已有场景正在加载，请等待完成。")
            self._state = dict(
                phase="running", message="正在读取重建场景…", inputs=deepcopy(payload)
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(deepcopy(payload),),
                name="scene-collision",
                daemon=True,
            )
            self._thread.start()
            return self.snapshot()

    def _progress(self, text: str) -> None:
        with self._lock:
            self._state["message"] = text

    def _run(self, payload: dict[str, str]) -> None:
        from embodichain.toolkits.skill_fitting._collision import (
            prepare_scene_collision,
            resolve_scene_path,
        )

        try:
            path = resolve_scene_path(payload["scene"], payload["joints"])
            if path is None:
                with self._lock:
                    self._state.update(
                        phase="idle", message="未找到重建场景；可填写场景目录后加载。"
                    )
                return
            result = prepare_scene_collision(path, progress=self._progress)
            with self._lock:
                self._state.update(
                    phase="done",
                    result=result,
                    message=(
                        f"场景已就绪：{result['object_count']} 个物体，"
                        f"{result['prepared_count']} 份碰撞网格，{result['hull_count']} 个凸块；"
                        f"复用 {result['cache_hits']} 份缓存。"
                    ),
                )
        except Exception as error:
            with self._lock:
                self._state.update(
                    phase="error", message="场景碰撞网格准备失败", error=str(error)
                )

    def close(self) -> None:
        """Finish the single cache publication before closing the host."""
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._thread.join()
