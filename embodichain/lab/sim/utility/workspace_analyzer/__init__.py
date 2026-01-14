# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

"""
Workspace Analyzer for Robotic Manipulation

A comprehensive tool for analyzing robot workspace reachability, computing
performance metrics, and generating visualizations.

Supports two analysis modes:
    1. Joint Space Mode: Sample joint configurations → FK → Workspace points
    2. Cartesian Space Mode: Sample Cartesian positions → IK → Verify reachability

Example:
    >>> from embodichain.lab.sim.utility.workspace_analyzer import (
    ...     WorkspaceAnalyzer,
    ...     AnalysisMode
    ... )
    >>> # Joint space analysis (default)
    >>> analyzer = WorkspaceAnalyzer(robot=robot)
    >>> results = analyzer.analyze(num_samples=10000)
    >>>
    >>> # Cartesian space analysis
    >>> from embodichain.lab.sim.utility.workspace_analyzer import WorkspaceAnalyzerConfig
    >>> config = WorkspaceAnalyzerConfig(mode=AnalysisMode.CARTESIAN_SPACE)
    >>> analyzer = WorkspaceAnalyzer(robot=robot, config=config)
    >>> results = analyzer.analyze(num_samples=10000)
"""

from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
    AnalysisMode,
)

# Import submodules for convenience
from embodichain.lab.sim.utility.workspace_analyzer import configs
from embodichain.lab.sim.utility.workspace_analyzer import samplers
from embodichain.lab.sim.utility.workspace_analyzer import caches
from embodichain.lab.sim.utility.workspace_analyzer import visualizers
from embodichain.lab.sim.utility.workspace_analyzer import metrics
from embodichain.lab.sim.utility.workspace_analyzer import constraints

__all__ = [
    "WorkspaceAnalyzer",
    "WorkspaceAnalyzerConfig",
    "AnalysisMode",
    "configs",
    "samplers",
    "caches",
    "visualizers",
    "metrics",
    "constraints",
]
