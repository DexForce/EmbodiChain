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

"""Task Program source schema, strict decoding, validation, and loading."""

from __future__ import annotations

from .decoder import (
    ConfigPath,
    ConfigPathPart,
    SceneReferenceRole,
    TaskProgramConfigError,
    TaskProgramDecodeError,
    TaskProgramValidationContext,
    TaskProgramValidationError,
    decode_task_program,
    render_config_path,
    validate_task_program,
)
from .loader import load_task_program, loads_task_program_json, parse_task_program_json
from .schema import (
    ArticulationJointPositionValidatorCfg,
    BarrierCfg,
    CyclicPoseTargetCfg,
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
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
    WaitStablePostCfg,
)

__all__ = [
    "ArticulationJointPositionValidatorCfg",
    "BarrierCfg",
    "ConfigPath",
    "ConfigPathPart",
    "CyclicPoseTargetCfg",
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
    "TaskProgramCfg",
    "TaskProgramConfigError",
    "TaskProgramDecodeError",
    "TaskProgramIntegrationCfg",
    "TaskProgramValidationContext",
    "TaskProgramValidationError",
    "WaitStablePostCfg",
    "decode_task_program",
    "load_task_program",
    "loads_task_program_json",
    "parse_task_program_json",
    "render_config_path",
    "validate_task_program",
]
