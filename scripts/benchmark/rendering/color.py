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

"""Device-preserving IEC sRGB delivery conversions, without photometric fitting."""

from __future__ import annotations

from typing import Any

__all__ = ["linear_to_srgb", "linear_u8_to_srgb_u8"]


def linear_to_srgb(value: Any, torch: Any) -> Any:
    """Apply the IEC 61966-2-1 transfer to nonnegative linear float tensors.

    Args:
        value: Scene-linear RGB tensor on the caller's existing device.
        torch: The backend's already loaded Torch module.

    Returns:
        Floating-point display-sRGB values on the same device. Values above one
        remain above one; output clipping and quantization belong to delivery.
    """
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def linear_u8_to_srgb_u8(value: Any, torch: Any) -> Any:
    """Encode linear UNORM8 RGB on its existing device.

    Args:
        value: Linear uint8 RGB tensor.
        torch: The backend's already loaded Torch module.

    Returns:
        Display-sRGB uint8 tensor. No host readback occurs. This conversion
        cannot recover precision lost during native linear-UNORM8 packing.
    """
    linear = value.to(torch.float32) / 255.0
    encoded = linear_to_srgb(linear, torch)
    return (encoded.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
