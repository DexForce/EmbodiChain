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

"""PiPER V100 Cartesian waypoint sequence using the published NMG v1 policy."""

from __future__ import annotations

from examples.sim.motion.planners.nmg._common import run

if __name__ == "__main__":
    run(
        "piper",
        offsets=[
            [0.20, -0.08, 0.10, 0.10, -0.05, 0.10],
            [-0.20, 0.08, -0.10, -0.10, 0.08, -0.10],
            [0.10, -0.12, 0.08, 0.12, -0.08, 0.15],
        ],
        modalities=["pose", "pose", "pose"],
        look_at=((1.0, -1.0, 0.85), (0.20, 0.0, 0.27), (0.0, 0.0, 1.0)),
    )
