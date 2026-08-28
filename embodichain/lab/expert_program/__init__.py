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

"""Provider-independent Expert Program schema, decoding, and compilation.

Expert Program is the task-level action-generation entry point. Environment
and simulator adapters live in :mod:`embodichain.lab.gym.envs.expert_program`.
"""

from __future__ import annotations

from .cfg import (
    ArticulationJointPositionValidatorCfg,
    BarrierCfg,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
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
from .compiler import CompiledProgram, ExpertProgramCompileError, ExpertProgramCompiler
from .decoder import (
    ConfigPath,
    ConfigPathPart,
    ExpertProgramConfigError,
    ExpertProgramDecodeError,
    ExpertProgramValidationContext,
    ExpertProgramValidationError,
    SceneReferenceRole,
    decode_expert_program,
    render_config_path,
    validate_expert_program,
)
from .loader import (
    load_expert_program,
    loads_expert_program_json,
    parse_expert_program_json,
)

__all__ = [
    "ArticulationJointPositionValidatorCfg",
    "BarrierCfg",
    "CompiledProgram",
    "ConfigPath",
    "ConfigPathPart",
    "CyclicPoseTargetCfg",
    "ExpertProgramCfg",
    "ExpertProgramCompileError",
    "ExpertProgramCompiler",
    "ExpertProgramConfigError",
    "ExpertProgramDecodeError",
    "ExpertProgramIntegrationCfg",
    "ExpertProgramValidationContext",
    "ExpertProgramValidationError",
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
    "decode_expert_program",
    "load_expert_program",
    "loads_expert_program_json",
    "parse_expert_program_json",
    "render_config_path",
    "validate_expert_program",
]
