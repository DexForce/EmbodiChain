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

"""Analyze the DexForce W1 left-arm workspace by sampling Cartesian positions."""

from __future__ import annotations

import torch

from embodichain.lab.sim.motion.workspace import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.configs import (
    SamplingConfig,
    SamplingStrategy,
    VisualizationConfig,
)

if __package__:
    from ._common import parse_args, workspace_scene, show_results
else:
    from _common import parse_args, workspace_scene, show_results


def main() -> None:
    """Run the cartesian workspace API example."""
    args = parse_args(
        __doc__, samples=50000, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    with workspace_scene(args, reference_pose=True) as (sim, robot, reference_pose):
        analyzer = WorkspaceAnalyzer(
            robot=robot,
            config=WorkspaceAnalyzerConfig(
                mode=AnalysisMode.CARTESIAN_SPACE,
                sampling=SamplingConfig(strategy=SamplingStrategy.RANDOM),
                visualization=VisualizationConfig(
                    show_unreachable_points=True,
                    point_size=8.0,
                ),
                control_part_name="left_arm",
            ),
            sim_manager=sim,
        )
        results = analyzer.analyze(num_samples=args.num_samples, visualize=True)
        show_results(
            sim,
            results,
            title="Cartesian Space",
            count_key="num_reachable",
            count_label="Reachable points",
            no_wait=args.no_wait,
        )


if __name__ == "__main__":
    main()
