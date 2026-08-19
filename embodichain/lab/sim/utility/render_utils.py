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

import torch

from embodichain.utils import logger

__all__ = ["select_default_renderer"]

# GPU name fragments that map to a fully ray-traced ('fast-rt') default. These
# are datacenter accelerators where ray tracing throughput is preferred over the
# rasterization fast-path used on consumer (RTX) cards.
_FAST_RT_GPU_KEYWORDS = ("A100", "A800", "H100", "H800", "H200", "H20")


def select_default_renderer(gpu_id: int = 0) -> str:
    """Select the default renderer backend based on the detected GPU.

    The selection rule is:

    - If :data:`embodichain.lab.sim.cfg.DEFAULT_RENDERER` is set to a concrete
      value (anything other than ``"auto"``), that value is honored. This lets
      callers (e.g. test fixtures) force a renderer regardless of hardware.
    - RTX-series cards use ``"hybrid"`` (ray tracing for shadows/reflections with
      rasterized primary rendering).
    - Datacenter cards (A100/A800, H100/H800/H200/H20) use ``"fast-rt"`` (fully
      ray-traced rendering).
    - If no CUDA device is available or the GPU name is unrecognized, falls back
      to ``"hybrid"``.

    Args:
        gpu_id: The CUDA device index to query for selecting the renderer.

    Returns:
        The resolved renderer name, one of ``"hybrid"``, ``"fast-rt"``, or ``"rt"``.
    """
    from embodichain.lab.sim import cfg

    # Explicit override takes precedence over hardware auto-detection.
    if cfg.DEFAULT_RENDERER != "auto":
        return cfg.DEFAULT_RENDERER

    if not torch.cuda.is_available():
        logger.log_info("No CUDA device available; defaulting renderer to 'hybrid'.")
        return "hybrid"

    try:
        device_name = torch.cuda.get_device_name(gpu_id)
    except Exception as exc:
        logger.log_warning(
            f"Failed to query GPU name for device {gpu_id} ({exc}). "
            "Defaulting renderer to 'hybrid'."
        )
        return "hybrid"

    upper_name = device_name.upper()
    if any(keyword in upper_name for keyword in _FAST_RT_GPU_KEYWORDS):
        logger.log_info(
            f"Detected datacenter GPU '{device_name}'; selecting 'fast-rt' renderer."
        )
        return "fast-rt"

    if "RTX" in upper_name:
        logger.log_info(
            f"Detected RTX GPU '{device_name}'; selecting 'hybrid' renderer."
        )
        return "hybrid"

    logger.log_info(
        f"Unrecognized GPU '{device_name}'; defaulting renderer to 'hybrid'."
    )
    return "hybrid"
