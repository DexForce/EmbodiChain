# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from embodichain.utils import configclass


@configclass
class AlgorithmCfg:
    """Minimal algorithm configuration shared across RL algorithms."""

    device: str = "cuda"
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
