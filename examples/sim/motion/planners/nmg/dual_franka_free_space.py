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

"""Move two FR3 arms through mixed joint and Cartesian waypoint pairs."""

from __future__ import annotations

from examples.sim.motion.planners.nmg._dual_franka import run_dual_franka

if __name__ == "__main__":
    run_dual_franka(
        description="Dual Franka FR3 synchronized free-space motion",
        offsets={
            "left_arm": [
                [0.18, -0.14, 0.10, -0.10, 0.05, 0.06, 0.03],
                [0.58, 0.14, -0.03, 0.20, 0.06, -0.16, 0.17],
                [0.45, 0.37, 0.13, 0.06, -0.16, -0.18, 0.22],
                [-0.17, 0.27, -0.14, 0.17, 0.08, -0.10, -0.18],
                [-0.50, -0.18, 0.02, 0.24, -0.10, -0.08, -0.08],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "right_arm": [
                [-0.29, -0.05, -0.07, -0.06, -0.04, 0.10, -0.03],
                [-0.61, 0.20, 0.06, 0.15, -0.10, -0.20, -0.13],
                [-0.37, 0.31, -0.10, 0.12, 0.11, -0.07, -0.19],
                [0.12, 0.31, 0.11, 0.18, -0.05, -0.12, 0.12],
                [0.49, -0.10, -0.02, 0.19, 0.13, -0.06, 0.09],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        },
        modalities={
            "left_arm": ("joint", "pose", "pose", "joint", "pose", "joint"),
            "right_arm": ("pose", "joint", "pose", "pose", "joint", "joint"),
        },
    )
