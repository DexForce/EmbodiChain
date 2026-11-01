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

"""Task-conditioned joint scene and robot installation feasibility search.

Geometry/kinematics/planning evidence remains distinct from measured task
success. Importing this package never imports Newton or creates a simulator.
"""

from __future__ import annotations

from .cfg import SceneRobotFeasibilityCfg
from .contracts import (
    PoseSearchVariable,
    PhaseTarget,
    HeldObject,
    TaskPhase,
    TaskOperationMode,
    TaskFeasibilityRequest,
    RobotDeploymentCandidate,
    JointSceneCandidate,
    FeasibilityCheck,
    CandidateEvaluation,
)
from .budget import FeasibilityBudget, FeasibilityBudgetExhausted
from .collision import (
    CollisionBody,
    NewtonCollisionChecker,
    UnsupportedFeasibilityError,
)
from .planning import (
    TaskPlanningBackend,
    TaskPlan,
    plan_task_mode,
    RobotFeasibilityBackend,
)
from .search import (
    FeasibilityEvaluator,
    PreparedFeasibilitySearch,
    FeasibilityObservation,
    FeasibleRegion,
    FeasibilityAtlas,
    prepare_feasibility,
    search_feasibility,
    sample_feasibility,
    validate_feasibility,
    binomial_interval,
)
from .persistence import (
    request_from_mapping,
    save_feasibility_atlas,
    load_feasibility_atlas,
)

__all__ = [
    "RobotWorkspaceProposalSource",
    "SceneRobotFeasibilityCfg",
    "PoseSearchVariable",
    "PhaseTarget",
    "HeldObject",
    "TaskPhase",
    "TaskOperationMode",
    "TaskFeasibilityRequest",
    "RobotDeploymentCandidate",
    "JointSceneCandidate",
    "FeasibilityCheck",
    "CandidateEvaluation",
    "FeasibilityBudget",
    "FeasibilityBudgetExhausted",
    "CollisionBody",
    "NewtonCollisionChecker",
    "UnsupportedFeasibilityError",
    "TaskPlanningBackend",
    "TaskPlan",
    "plan_task_mode",
    "RobotFeasibilityBackend",
    "FeasibilityEvaluator",
    "PreparedFeasibilitySearch",
    "FeasibilityObservation",
    "FeasibleRegion",
    "FeasibilityAtlas",
    "prepare_feasibility",
    "search_feasibility",
    "sample_feasibility",
    "validate_feasibility",
    "binomial_interval",
    "request_from_mapping",
    "save_feasibility_atlas",
    "load_feasibility_atlas",
]

from .proposals import RobotWorkspaceProposalSource
