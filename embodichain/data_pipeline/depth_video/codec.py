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

"""Codec availability detection for depth video encoding.

Adapted from lerobot v0.6.0 ``pyav_utils.py`` (Apache-2.0). Probes the bundled
FFmpeg build through PyAV so the writer can pick a working encoder and degrade
gracefully (to numeric Parquet depth, PR #422) when HEVC is unavailable.
"""

from __future__ import annotations

import functools

import av

__all__ = ["DepthCodecError", "detect_depth_encoder", "resolve_depth_vcodec"]


class DepthCodecError(RuntimeError):
    """Raised when no suitable depth video codec is available."""


@functools.cache
def get_codec(vcodec: str) -> av.codec.Codec | None:
    """PyAV write-mode ``Codec`` for *vcodec*, or ``None`` if unavailable."""
    try:
        return av.codec.Codec(vcodec, "w")
    except Exception:
        return None


def detect_depth_encoder(vcodec: str | None = None) -> str | None:
    """Return an available depth video encoder name, or ``None``.

    Args:
        vcodec: Preferred codec name. If ``None`` or unavailable, fall back to
            the default preference list (``libx265`` then ``hevc``).

    Returns:
        The first available encoder name, or ``None`` if none are available.
    """
    preference: list[str] = []
    if vcodec:
        preference.append(vcodec)
    for name in ("libx265", "hevc"):
        if name not in preference:
            preference.append(name)

    for name in preference:
        codec = get_codec(name)
        if codec is not None and codec.type == "video":
            return name
    return None


def resolve_depth_vcodec(vcodec: str | None = None) -> str:
    """Return an available depth video encoder name, raising on failure.

    Args:
        vcodec: Preferred codec name.

    Returns:
        An available encoder name.

    Raises:
        DepthCodecError: If no depth video encoder is available in the bundled
            FFmpeg build.
    """
    name = detect_depth_encoder(vcodec)
    if name is None:
        raise DepthCodecError(
            "No HEVC video encoder (libx265/hevc) is available in the bundled "
            "FFmpeg build; cannot write compressed depth videos. Install an "
            "FFmpeg build with libx265, or disable depth video to fall back to "
            "numeric depth features (PR #422)."
        )
    return name
