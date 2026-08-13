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

"""Visual motion-policy evaluation through DexSim Motion Policy Kit."""

from __future__ import annotations

from .bridge import (
    MotionEvaluationResult,
    create_motion_profile_evaluator,
    evaluate_motion_profile,
)
from .checkpoint import load_policy_state_dict
from .manifest import RunManifest, write_run_manifest
from .profile import (
    MotionProfile,
    MotionProfileRequest,
    build_motion_profile,
    get_motion_profile_names,
    register_motion_profile,
)
from .report import write_evaluation_report

__all__ = [
    "MotionEvaluationResult",
    "MotionProfile",
    "MotionProfileRequest",
    "RunManifest",
    "build_motion_profile",
    "create_motion_profile_evaluator",
    "evaluate_motion_profile",
    "get_motion_profile_names",
    "load_policy_state_dict",
    "register_motion_profile",
    "write_run_manifest",
    "write_evaluation_report",
]
