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

"""Packaged browser assets for expansion playback and distribution reports."""

from __future__ import annotations

from importlib.resources import files

__all__: list[str] = []


def _preview_assets() -> dict[str, tuple[str, bytes]]:
    """Load same-origin assets for the native WebRTC preview."""
    directory = files(__name__)
    assets = {
        "/": (
            "text/html; charset=utf-8",
            directory.joinpath("preview.html").read_bytes(),
        ),
        "/preview.css": (
            "text/css; charset=utf-8",
            directory.joinpath("preview.css").read_bytes(),
        ),
    }
    for name in (
        "preview-text.js",
        "preview-controls.js",
        "preview-charts.js",
        "preview-heatmap.js",
    ):
        assets[f"/{name}"] = (
            "text/javascript; charset=utf-8",
            directory.joinpath(name).read_bytes(),
        )
    return assets
