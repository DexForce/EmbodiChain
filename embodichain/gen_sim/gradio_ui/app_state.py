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

"""Per-session runtime state for the Scene and Action engines."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app_config import PHASE_DEFINITIONS

__all__ = [
    "PHASES",
    "Phase",
    "RuntimeState",
    "SessionRuntimeRegistry",
    "runtime_lock",
    "runtime_registry",
    "set_runtime_phase_locked",
]


@dataclass(frozen=True)
class Phase:
    progress: int
    label: str


PHASES = {key: Phase(*value) for key, value in PHASE_DEFINITIONS.items()}


@dataclass
class RuntimeState:
    """Mutable Scene/Action UI state owned by one Gradio session."""

    is_busy: bool = False
    scene_engine_is_running: bool = False
    phase_key: str = "idle"
    status: str = "Idle."
    task_text: str = ""
    image_path: Path | None = None
    video_path: Path | None = None
    last_sent_video_signature: tuple[str, int] | None = None
    last_error: str | None = None
    log_lines: deque[str] = field(default_factory=deque)


class SessionRuntimeRegistry:
    """Own one Scene/Action UI runtime for each Gradio session hash."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, RuntimeState] = {}

    def get(self, session_id: str) -> RuntimeState:
        """Return the existing state for a session, creating it when absent.

        Args:
            session_id: Stable Gradio session identifier.

        Returns:
            Runtime state owned exclusively by ``session_id``.
        """
        with self._lock:
            return self._states.setdefault(session_id, RuntimeState())

    def reset(self, session_id: str) -> None:
        """Discard only one session's UI runtime state.

        Args:
            session_id: Stable Gradio session identifier.
        """
        with self._lock:
            self._states.pop(session_id, None)


runtime_registry = SessionRuntimeRegistry()
runtime_lock = threading.Lock()


def set_runtime_phase_locked(runtime: RuntimeState, new_phase_key: str) -> None:
    """Set the current UI phase while the caller holds ``runtime_lock``."""
    runtime.phase_key = new_phase_key
