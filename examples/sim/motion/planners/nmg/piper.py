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

"""PiPER V100 mixed free-space targets and NMG feedback recovery."""

from __future__ import annotations

from examples.sim.motion.planners.nmg._common import run

if __name__ == "__main__":
    run(
        "piper",
        offsets=[
            [0.32, -0.15, 0.15, 0.08, -0.10, 0.04],
            [0.70, -0.36, 0.26, 0.15, 0.22, 0.14],
            [0.72, 0.24, -0.26, -0.26, 0.17, 0.05],
            [0.18, 0.24, -0.30, 0.10, -0.25, -0.14],
            [-0.60, 0.15, -0.10, 0.14, 0.17, 0.11],
            [-0.76, -0.28, 0.25, -0.17, 0.18, -0.18],
            [-0.43, -0.32, -0.03, 0.17, -0.13, 0.11],
            [-0.07, 0.04, 0.06, -0.08, -0.08, 0.03],
        ],
        modalities=[
            "joint",
            "pose",
            "position",
            "pose",
            "joint",
            "pose",
            "position",
            "pose",
        ],
        look_at=((0.84, -0.79, 0.73), (0.24, 0, 0.33), (0, 0, 1)),
    )
