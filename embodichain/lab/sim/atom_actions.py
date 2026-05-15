# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Backward-compatible import path for Agent atomic skills.

The implementation now lives under ``embodichain.lab.sim.agent.atom_actions``.
This module keeps the historical ``embodichain.lab.sim.atom_actions`` import
path working for existing tutorials, docs, and downstream user code.
"""

from embodichain.lab.sim.agent.atom_actions import *  # noqa: F401,F403
