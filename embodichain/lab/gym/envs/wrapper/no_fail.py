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
