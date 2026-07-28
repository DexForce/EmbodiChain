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

"""Configuration for compressed depth sidecar storage."""

from __future__ import annotations

from typing import Literal

from embodichain.utils import configclass

from .depth_utils import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
)

__all__ = ["DepthVideoCfg"]

# Codebase version stamped into ``depth_meta.json`` so a future migration to the
# official lerobot 0.6.0 depth pipeline (issue #424, Path B) can recognise and
# convert sidecar videos produced by this writer.
EMBODI_DEPTH_CODEBASE_VERSION = "embodi-depth-v1"


@configclass
class DepthVideoCfg:
    """Configuration for the compressed depth sidecar writer.

    Depth is quantized to 12-bit codes and encoded as a single-channel
    ``gray12le`` video. With ``lossless=True`` (default) the 12-bit codes are
    preserved bit-exactly by HEVC; the only error is the configurable
    float32 -> 12-bit quantization step.

    Attributes:
        enable: If False, depth is stored as numeric LeRobot features (PR #422).
        vcodec: Video codec for depth. Defaults to ``"libx265"`` (HEVC), which
            supports 12-bit grayscale losslessly on typical FFmpeg builds.
        lossless: If True, encode with HEVC lossless mode so 12-bit codes are
            bit-exact. If False, use ``crf`` for lossy encoding.
        crf: Constant rate factor for lossy mode (ignored when ``lossless=True``).
            Lower is higher quality; 0 is lossless for libx265.
        depth_min: Depth (metres) mapped to quantum 0.
        depth_max: Depth (metres) mapped to quantum ``DEPTH_QMAX``.
        shift: Pre-log offset (metres) for numerical stability near zero.
        use_log: Logarithmic (True) or linear (False) quantization.
        pix_fmt: Pixel format for the depth video. ``gray12le`` carries the
            12-bit codes in a single channel.
        input_unit: Unit of the incoming depth arrays (``"auto"`` infers from
            dtype: float -> metres, int -> millimetres).
        output_unit: Unit returned by the reader (``"m"`` or ``"mm"``).
        keep_numeric_fallback: If True, also keep depth as a numeric LeRobot
            feature (exact raw values, ~2x depth storage). If False, depth lives
            only in the sidecar videos.
    """

    enable: bool = False
    vcodec: str = "libx265"
    lossless: bool = True
    crf: int = 0
    depth_min: float = DEFAULT_DEPTH_MIN
    depth_max: float = DEFAULT_DEPTH_MAX
    shift: float = DEFAULT_DEPTH_SHIFT
    use_log: bool = DEFAULT_DEPTH_USE_LOG
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT
    input_unit: Literal["auto", "m", "mm"] = "auto"
    output_unit: Literal["m", "mm"] = "m"
    keep_numeric_fallback: bool = False
