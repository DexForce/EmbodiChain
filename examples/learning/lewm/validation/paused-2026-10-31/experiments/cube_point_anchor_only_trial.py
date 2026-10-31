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

"""Private control changing only the point force moment arms."""

from __future__ import annotations
from unittest.mock import patch
from embodichain_tasks.manipulation.ogbench_cube import OGBenchCubeEnv


class PointAnchorOnlyCube(OGBenchCubeEnv):
    def _setup_robot(self, **kwargs):
        original = self.sim.create_rigid_constraint

        def create(cfg, *args, **extra):
            cfg.linear_drive_point_anchors = True
            return original(cfg, *args, **extra)

        with patch.object(self.sim, "create_rigid_constraint", create):
            return super()._setup_robot(**kwargs)
