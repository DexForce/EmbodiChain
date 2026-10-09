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

from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    import warp as wp

__all__ = ["standardize_device_string", "current_warp_stream"]


def current_warp_stream(device: torch.device) -> wp.Stream | None:
    """Use the current Torch stream for Warp operations.

    Args:
        device: Device used by the caller's tensors.

    Returns:
        The registered Warp stream when it shares Torch's current native handle,
        otherwise a Warp wrapper of the current Torch stream. CPU returns None.
    """
    if device.type != "cuda":
        return None
    import warp as wp

    current = torch.cuda.current_stream(device)
    stream = wp.get_stream(str(device))
    # A temporary second Warp wrapper unregisters the shared native handle on
    # destruction in Warp 1.17, invalidating an enclosing capture.
    if stream.cuda_stream == current.cuda_stream:
        return stream
    return wp.stream_from_torch(current)


def standardize_device_string(device: Union[str, torch.device]) -> str:
    """Standardize the device string for Warp compatibility.

    Args:
        device (Union[str, torch.device]): The device specification.

    Returns:
        str: The standardized device string.
    """
    if isinstance(device, str):
        device_str = device
    else:
        device_str = str(device)

    if device_str == "cuda":
        device_str = "cuda:0"

    return device_str
