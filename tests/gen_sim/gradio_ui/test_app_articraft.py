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

import sys
from pathlib import Path

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_articraft  # noqa: E402

VIEWER_PORT = 54_321


@pytest.mark.parametrize("hostname", ["127.0.0.1", "localhost"])
def test_articraft_viewer_port_accepts_local_http_url(hostname: str) -> None:
    viewer_output = f"Viewer URL: http://{hostname}:{VIEWER_PORT}/"

    assert app_articraft._articraft_viewer_port(viewer_output) == VIEWER_PORT


@pytest.mark.parametrize(
    "viewer_output",
    [
        f"Viewer URL: https://localhost:{VIEWER_PORT}",
        f"Viewer URL: http://example.com:{VIEWER_PORT}",
        "Viewer URL: http://localhost:not-a-port",
        "Articraft viewer is starting",
    ],
)
def test_articraft_viewer_port_rejects_untrusted_or_invalid_output(
    viewer_output: str,
) -> None:
    assert app_articraft._articraft_viewer_port(viewer_output) is None


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
