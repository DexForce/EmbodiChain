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

"""Source-independent real-to-simulation trajectory preparation."""

from __future__ import annotations

from .contracts import (
    JointSpec,
    NormalizationReport,
    NormalizedReal2SimUnit,
    PhaseAnnotation,
    Real2SimTrajectory,
    TrajectoryTimeMap,
)
from .normalize import normalize_trajectory
from .prototype import (
    SkillMotionPrototype,
    build_motion_prototype,
    retarget_motion_prototype,
)
from .io import load_prototype, load_trajectory, prepare_prototype, save_trajectory

__all__ = [
    "JointSpec",
    "NormalizationReport",
    "NormalizedReal2SimUnit",
    "PhaseAnnotation",
    "Real2SimTrajectory",
    "TrajectoryTimeMap",
    "SkillMotionPrototype",
    "build_motion_prototype",
    "load_trajectory",
    "load_prototype",
    "normalize_trajectory",
    "prepare_prototype",
    "retarget_motion_prototype",
    "save_trajectory",
]
