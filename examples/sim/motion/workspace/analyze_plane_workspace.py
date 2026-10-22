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

"""Analyze the DexForce W1 left-arm workspace by sampling a plane."""

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
    """Run the plane workspace API example."""
    args = parse_args(__doc__, samples=1500, device="cpu")
    with workspace_scene(args, reference_pose=True) as (sim, robot, reference_pose):
        analyzer = WorkspaceAnalyzer(
            robot=robot,
            config=WorkspaceAnalyzerConfig(
                mode=AnalysisMode.PLANE_SAMPLING,
                plane_normal=torch.tensor(
                    [0.0, 0.0, 1.0],
                    dtype=torch.float32,
                    device=robot.device,
                ),
                plane_point=torch.tensor(
                    [0.0, 0.0, 1.2],
                    dtype=torch.float32,
                    device=robot.device,
                ),
                reference_pose=reference_pose[0],
                sampling=SamplingConfig(strategy=SamplingStrategy.RANDOM),
                visualization=VisualizationConfig(
                    show_unreachable_points=True,
                    vis_type="axis",
                ),
                control_part_name="left_arm",
            ),
            sim_manager=sim,
        )
        results = analyzer.analyze(num_samples=args.num_samples, visualize=True)
        show_results(
            sim,
            results,
            title="Plane Sampling",
            count_key="num_reachable",
            count_label="Reachable points",
            no_wait=args.no_wait,
        )


if __name__ == "__main__":
    main()
