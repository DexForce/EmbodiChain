# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from .configclass import configclass, is_configclass


GLOBAL_SEED = 1024


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set. If -1, a random seed will be generated.
        deterministic (bool): If True, sets the environment to deterministic mode for reproducibility.
    """
    import random
    import numpy as np
    import torch
    import os
    import warp as wp

    if seed == -1 and deterministic:
        seed = GLOBAL_SEED
    elif seed == -1:
        seed = np.random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    wp.rand_init(seed)

    if deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
