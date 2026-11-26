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

"""Franka FR3 wide-workspace constraints and NMG feedback recovery."""

from __future__ import annotations

from examples.sim.motion.planners.nmg._common import run

if __name__ == "__main__":
    run(
        "franka",
        offsets=[
            [0.32, -0.18, 0.10, -0.10, 0.05, 0.06, 0.03],
            [0.78, 0.16, -0.05, 0.20, 0.06, -0.22, 0.17],
            [0.55, 0.14, 0.13, 0.08, -0.15, -0.18, 0.22],
            [0.05, 0.37, -0.14, 0.16, 0.08, -0.10, -0.18],
            [-0.60, 0.18, -0.18, 0.25, -0.10, -0.10, -0.08],
            [-0.79, -0.30, 0.05, -0.20, 0.18, 0.24, 0.20],
            [-0.31, -0.21, 0.19, 0.13, -0.20, 0.11, -0.19],
            [0.00, 0.04, 0.00, -0.08, 0.00, 0.10, 0.10],
        ],
        modalities=[
            "joint",
            "pose",
            "position",
            "pose",
            "joint",
            "pose",
            "orientation",
            "pose",
        ],
        look_at=((1.11, -1.02, 0.9), (0.32, 0, 0.4), (0, 0, 1)),
    )
