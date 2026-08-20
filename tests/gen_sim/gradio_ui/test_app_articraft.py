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

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_articraft  # noqa: E402


def test_preview_selects_another_port_when_preferred_port_is_occupied() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    occupied_port = int(listener.getsockname()[1])
    previews = app_articraft._ArticraftViserPreview(occupied_port)
    try:
        selected_port = previews._select_available_port()
    finally:
        listener.close()

    assert selected_port != occupied_port
    assert selected_port > 0


def test_codex_checkout_cannot_be_the_embodichain_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_articraft,
        "ARTICRAFT_ROOT",
        app_articraft.EMBODICHAIN_ROOT,
    )

    assert "dedicated nested or external Git checkout" in (
        app_articraft._articraft_isolation_error() or ""
    )
