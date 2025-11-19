# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import gymnasium as gym


class NoFailWrapper(gym.Wrapper):
    """A wrapper that alter the env's is_task_success method to make sure all the is_task_success determination return True.

    Args:
        env (gym.Env): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def is_task_success(self, *args, **kwargs):
        return True
