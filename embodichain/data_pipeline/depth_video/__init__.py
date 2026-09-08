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

"""Compressed depth sidecar storage for LeRobot datasets on Python 3.10--3.12.

This package implements issue #424 *Path A*: an EmbodiChain-owned depth writer
that stores camera depth as ``gray12le``/HEVC sidecar videos alongside a
LeRobot dataset without modifying the installed LeRobot package.

Depth quantization math is vendored from lerobot v0.6.0 so that sidecar videos
remain binary-compatible with the official reader.
"""

from __future__ import annotations

from .cfg import DepthVideoCfg
from .codec import DepthCodecError, detect_depth_encoder, resolve_depth_vcodec
from .depth_utils import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
    dequantize_depth,
    quantize_depth,
)
from .reader import (
    DepthVideoLibrary,
    DepthVideoReader,
    load_depth_dataset,
    load_depth_meta,
)
from .writer import DepthSidecarManager, DepthVideoWriter

__all__ = [
    "DepthVideoCfg",
    "DepthCodecError",
    "detect_depth_encoder",
    "resolve_depth_vcodec",
    "DEFAULT_DEPTH_MIN",
    "DEFAULT_DEPTH_MAX",
    "DEFAULT_DEPTH_SHIFT",
    "DEFAULT_DEPTH_USE_LOG",
    "DEFAULT_DEPTH_PIX_FMT",
    "DEPTH_METER_UNIT",
    "DEPTH_MILLIMETER_UNIT",
    "DEPTH_QMAX",
    "quantize_depth",
    "dequantize_depth",
    "DepthVideoWriter",
    "DepthSidecarManager",
    "DepthVideoReader",
    "DepthVideoLibrary",
    "load_depth_dataset",
    "load_depth_meta",
]
