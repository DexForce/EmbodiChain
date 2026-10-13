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

"""Fluid replay deployment for Task Program-generated pouring trajectories."""

from __future__ import annotations
from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["PourCupFluidReplayEnv"]


@register_env("PourCupFluidReplay-Franka-v1", max_episode_steps=2000)
class PourCupFluidReplayEnv(FluidReplayEnv):
    """Evaluate a qualified Franka pouring trajectory with live SPH fluids."""
