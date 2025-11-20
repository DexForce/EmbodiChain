# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
import numpy as np

from typing import Union, Optional

from embodichain.lab.sim.types import Array, Device


def to_tensor(array: Array, device: Optional[Device] = None):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v) for k, v in array.items()}
    if torch.cuda.is_available():
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, torch.Tensor):
            ret = array
        else:
            ret = torch.tensor(array)
        if device is None:
            if ret.device.type == "cpu":
                return ret.cuda()
            # keep same device if already on GPU
            return ret
        else:
            return ret.to(device)
    else:
        if isinstance(array, np.ndarray):
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            if array.dtype == np.uint32:
                array = array.astype(np.int64)
            ret = torch.from_numpy(array)
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif isinstance(array, list) and isinstance(array[0], np.ndarray):
            ret = torch.from_numpy(np.array(array))
            if ret.dtype == torch.float64:
                ret = ret.float()
        elif np.iterable(array):
            ret = torch.Tensor(array)
        else:
            ret = torch.Tensor(array)
        if device is None:
            return ret
        else:
            return ret.to(device)
