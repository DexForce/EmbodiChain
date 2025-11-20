# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch

from typing import Union


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

    if device_str.startswith("cuda"):
        device_str = "cuda:0"

    return device_str
