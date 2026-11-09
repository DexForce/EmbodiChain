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

"""Parallel-jaw grasp APIs with optional backends loaded on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "for_axis_region",
    "select_supported_end",
    "AntipodalGraspPoseGenerator",
    "AntipodalGraspPoseGeneratorCfg",
    "AntipodalSampler",
    "AntipodalSamplerCfg",
    "ConvexCollisionChecker",
    "ConvexCollisionCheckerCfg",
    "GraspAnnotationCfg",
    "FixedGripperCollisionGeometry",
    "get_parallel_jaw_collision_geometry",
    "GripperCollisionCfg",
    "GripperCollisionChecker",
    "GraspPoseGenerator",
    "GraspRegionSelector",
    "ParallelJawGraspPoseGenerator",
    "ParallelJawGraspCollisionCfg",
    "ParallelJawGripperModelCfg",
]


# Include the explicit-import names previously exposed by the star imports,
# while retaining the public star-import contract above.
_EXPORT_MODULES = {
    "GraspPoseGenerator": "..pose_generator",
    "ParallelJawGraspPoseGenerator": "..pose_generator",
    "ParallelJawGripperModelCfg": "..pose_generator",
    "get_parallel_jaw_gripper_model": "..pose_generator",
    "AntipodalSamplerCfg": ".antipodal_sampler",
    "AntipodalSampler": ".antipodal_sampler",
    "ConvexCollisionCheckerCfg": ".collision_checker",
    "ConvexCollisionChecker": ".collision_checker",
    "GripperCollisionCfg": ".gripper_collision_checker",
    "GripperCollisionChecker": ".gripper_collision_checker",
    "FixedGripperCollisionGeometry": ".fixed_geometry",
    "get_parallel_jaw_collision_geometry": ".model_geometry",
    "AntipodalGraspPoseGenerator": ".pose_generator",
    "AntipodalGraspPoseGeneratorCfg": ".pose_generator",
    "GraspRegionSelector": ".pose_generator",
    "GraspAnnotationCfg": ".pose_generator",
    "ParallelJawGraspCollisionCfg": ".pose_generator",
    "for_axis_region": ".regions",
    "select_supported_end": ".regions",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORT_MODULES.keys())
