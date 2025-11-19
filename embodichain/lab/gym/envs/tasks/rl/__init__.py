# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy
from embodichain.lab.gym.utils import registration as env_registry
from embodichain.lab.gym.envs.rl_env_cfg import RLEnvCfg


def build_env(env_id: str, base_env_cfg: RLEnvCfg):
    """Create env from registry id, auto-inferring cfg class (EnvName -> EnvNameCfg)."""
    env = env_registry.make(env_id, cfg=deepcopy(base_env_cfg))
    return env


__all__ = [
    "build_env",
]
