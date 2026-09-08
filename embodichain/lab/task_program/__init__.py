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

"""Declarative orchestration and execution language for embodied tasks.

Task Program is the provider-independent language between human or model
authored task intent and executable Atomic Skills.  Semantic Calls and their
scene, profile, effect, and evidence contracts are internal language
semantics; provider-specific assembly lives in :mod:`.integrations`, while
Gym owns only the environment execution bridge.
"""

from __future__ import annotations

from .language import (
    ArticulationJointPositionValidatorCfg,
    BarrierCfg,
    CyclicPoseTargetCfg,
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    ParallelCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from .compiler import CompiledTaskProgram, TaskProgramCompileError, TaskProgramCompiler
from .language import (
    ConfigPath,
    ConfigPathPart,
    TaskProgramConfigError,
    TaskProgramDecodeError,
    TaskProgramValidationContext,
    TaskProgramValidationError,
    SceneReferenceRole,
    decode_task_program,
    render_config_path,
    validate_task_program,
)
from .language import (
    load_task_program,
    loads_task_program_json,
    parse_task_program_json,
)

__all__ = [
    "ArticulationJointPositionValidatorCfg",
    "BarrierCfg",
    "CompiledTaskProgram",
    "ConfigPath",
    "ConfigPathPart",
    "CyclicPoseTargetCfg",
    "TaskProgramCfg",
    "TaskProgramCompileError",
    "TaskProgramCompiler",
    "TaskProgramConfigError",
    "TaskProgramDecodeError",
    "TaskProgramIntegrationCfg",
    "TaskProgramValidationContext",
    "TaskProgramValidationError",
    "HandOverCfg",
    "InvokeCfg",
    "ObjectNearTargetValidatorCfg",
    "ParallelCfg",
    "PickCfg",
    "PlaceCfg",
    "PoseCfg",
    "RegisteredSemanticCallCfg",
    "RepeatCfg",
    "SceneReferenceRole",
    "SegmentCfg",
    "SequenceCfg",
    "TargetRefCfg",
    "WaitStablePostCfg",
    "decode_task_program",
    "load_task_program",
    "loads_task_program_json",
    "parse_task_program_json",
    "render_config_path",
    "validate_task_program",
]
