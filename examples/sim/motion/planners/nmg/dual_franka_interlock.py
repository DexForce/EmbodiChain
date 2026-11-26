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

"""Activate a shared interlock by holding both FR3 TCPs in their sensor zones."""

from __future__ import annotations

from examples.sim.motion.planners.nmg._dual_franka import run_dual_franka

if __name__ == "__main__":
    run_dual_franka(
        description="Dual Franka FR3 coordinated proximity interlock",
        offsets={
            "left_arm": [
                [0.15, 0.04, 0.02, 0.04, 0.0, -0.02, 0.0],
                [0.24, 0.14, 0.04, 0.12, -0.05, -0.09, 0.05],
                [0.32, 0.30, 0.04, 0.17, -0.05, -0.19, 0.05],
                [0.24, 0.14, 0.04, 0.12, -0.05, -0.09, 0.05],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "right_arm": [
                [-0.09, 0.06, -0.02, 0.05, 0.03, -0.04, 0.0],
                [-0.28, 0.14, -0.07, 0.14, 0.05, -0.06, -0.03],
                [-0.41, 0.29, -0.12, 0.20, 0.09, -0.22, -0.11],
                [-0.28, 0.14, -0.07, 0.14, 0.05, -0.06, -0.03],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        },
        modalities={
            "left_arm": ("joint", "pose", "pose", "pose", "joint"),
            "right_arm": ("joint", "pose", "pose", "pose", "joint"),
        },
        trigger_slot=2,
    )
