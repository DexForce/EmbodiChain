# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
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
