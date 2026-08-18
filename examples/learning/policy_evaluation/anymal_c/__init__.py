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

"""Register the ANYmal-C velocity Motion Profile."""

from __future__ import annotations

from embodichain.learning.rl.policy_evaluation import register_motion_profile

from .profile import PROFILE_ID, build_profile

__all__ = ["register"]


def register() -> None:
    """Register the ANYmal-C velocity Profile for the example process."""
    register_motion_profile(PROFILE_ID, build_profile)
