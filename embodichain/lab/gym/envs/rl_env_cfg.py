# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from typing import Any, Dict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.utils import configclass


@configclass
class RLEnvCfg(EmbodiedEnvCfg):
    """Extended configuration for RL environments built from gym-style specs."""

    env_id: str = ""
    extensions: Dict[str, Any] = {}

    @classmethod
    def from_dict(cls, d):
        """Create an instance from a dictionary."""
        return cls(**d)
