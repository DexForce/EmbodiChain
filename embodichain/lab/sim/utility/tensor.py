# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
import numpy as np

from typing import Union, Optional


def to_tensor(
    arr: Union[torch.Tensor, np.ndarray, list],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert input to torch.Tensor with specified dtype and device.

    Supports torch.Tensor, np.ndarray, and list.

    Args:
        arr (Union[torch.Tensor, np.ndarray, list]): Input array.
        dtype (torch.dtype, optional): Desired tensor dtype. Defaults to torch.float32.
        device (torch.device, optional): Desired device. If None, uses current device.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype, device=device) if device else arr.to(dtype=dtype)
    elif isinstance(arr, np.ndarray):
        return (
            torch.from_numpy(arr).to(dtype=dtype, device=device)
            if device
            else torch.from_numpy(arr).to(dtype=dtype)
        )
    elif isinstance(arr, list):
        return (
            torch.tensor(arr, dtype=dtype, device=device)
            if device
            else torch.tensor(arr, dtype=dtype)
        )
    else:
        raise TypeError("Input must be a torch.Tensor, np.ndarray, or list.")
