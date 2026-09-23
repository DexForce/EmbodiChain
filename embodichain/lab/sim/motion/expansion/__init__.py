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

"""Host-independent grasp/qpos augmentation, coverage, and generation bookkeeping.

Execution, initial-state restoration, physical validation, and durable storage
are supplied by host integrations. These algorithms do not call Gym or a
simulation backend, but the public import follows normal ``lab.sim``
initialization and requires the simulation package dependencies.
"""

from __future__ import annotations

from .cfg import (
    SPATIAL_METHODS,
    TrajectoryAugmentationCfg,
    TrajectoryGenerationJobCfg,
)
from .contracts import (
    CandidateIdentity,
    CandidateSpec,
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpertEpisode,
    MotionSnapshot,
    SceneCase,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)
from .coverage import CoverageIndex, TrajectoryDescriptor, describe_trajectory
from .manipulability import (
    GuidedResidual,
    ManipulabilityBands,
    ManipulabilityProfile,
    describe_manipulability,
    manipulability_guided_residual,
)
from .variants import (
    NOMINAL_OPERATOR,
    TrajectoryVariant,
    TrajectoryVariantSet,
    apply_trajectory_variant,
    default_variant_factors,
    expand_trajectory_variants,
    plan_trajectory_variants,
    sample_approach_cone,
)
from .operators import (
    TIMING_PROFILES,
    ProposalRejected,
    joint_residual,
    nullspace_residual,
    perturb_approach_direction,
    retime,
    rotate_grasp_about_object_axis,
    validate_motion_limits,
    via_points,
)
from .session import GenerationSession
from .source import SourceAdapter, SourceContext

__all__ = [
    "SPATIAL_METHODS",
    "CandidateIdentity",
    "CandidateSpec",
    "CandidateTrajectoryBatch",
    "CommitReceipt",
    "ExpertEpisode",
    "MotionSnapshot",
    "SceneCase",
    "TrajectoryAugmentationCfg",
    "TrajectoryGenerationJobCfg",
    "TrajectoryPhase",
    "TrajectoryTemplate",
    "ValidationCheck",
    "ValidationResult",
    "CoverageIndex",
    "TrajectoryDescriptor",
    "describe_trajectory",
    "GuidedResidual",
    "ManipulabilityBands",
    "ManipulabilityProfile",
    "describe_manipulability",
    "manipulability_guided_residual",
    "NOMINAL_OPERATOR",
    "TrajectoryVariant",
    "TrajectoryVariantSet",
    "apply_trajectory_variant",
    "default_variant_factors",
    "expand_trajectory_variants",
    "plan_trajectory_variants",
    "sample_approach_cone",
    "TIMING_PROFILES",
    "ProposalRejected",
    "joint_residual",
    "nullspace_residual",
    "perturb_approach_direction",
    "retime",
    "rotate_grasp_about_object_axis",
    "validate_motion_limits",
    "via_points",
    "GenerationSession",
    "SourceAdapter",
    "SourceContext",
]
