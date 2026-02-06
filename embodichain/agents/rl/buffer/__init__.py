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

"""
Buffer module for RL training.

Provides two buffer implementations:
- RolloutBuffer: Standard PPO buffer (single rollout, use and discard)
- VLABuffer: VLA buffer (FIFO multi-rollout accumulation for slow inference)
"""

from .vla_buffer import VLABuffer
from .standard_buffer import RolloutBuffer

__all__ = ["RolloutBuffer", "VLABuffer"]
