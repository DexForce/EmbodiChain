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

"""Three-camera stationary demo used to benchmark long LeRobot episodes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

__all__ = ["StayStillSave3CamEnv"]


@register_env("StayStillSave3Cam-v1", max_episode_steps=301)
class StayStillSave3CamEnv(EmbodiedEnv):
    """Hold the UR10 pose for 300 steps while recording three RGB cameras."""

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Yield one deterministic 300-step stationary demonstration segment."""

        del kwargs
        init_pose = self.robot.get_qpos().clone()

        def actions() -> Iterable[Any]:
            for _ in range(300):
                yield init_pose.clone()

        logger.log_info("Generated 300 hold-still demo actions for three cameras.")
        yield DemoSegment(
            actions=actions(),
            name="hold_still_300_steps",
            instruction="Hold still while recording three RGB cameras",
            metadata={"segment_index": 0, "segment_count": 1, "steps": 300},
            progress_total_steps=300,
        )
