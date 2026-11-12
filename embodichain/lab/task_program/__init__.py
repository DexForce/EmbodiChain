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

Public owners load on access; data-only evidence imports create no runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PoseTargetCfg",
    "SceneRelativePoseTargetCfg",
    "MoveHeldObjectCfg",
    "HoldPostCfg",
    "NormalizedTaskProgram",
    "normalize_task_program",
    "ArticulationJointPositionValidatorCfg",
    "BarrierCfg",
    "ConditionValidatorCfg",
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
    "ObjectNearRelativeTargetValidatorCfg",
    "ObjectNearTargetValidatorCfg",
    "ObjectLiftedValidatorCfg",
    "ObjectRelativePositionValidatorCfg",
    "ObjectUprightValidatorCfg",
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

_EXPORTS = {
    "ArticulationJointPositionValidatorCfg": (
        ".language",
        "ArticulationJointPositionValidatorCfg",
    ),
    "BarrierCfg": (".language", "BarrierCfg"),
    "ConditionValidatorCfg": (".language", "ConditionValidatorCfg"),
    "CompiledTaskProgram": (".compiler", "CompiledTaskProgram"),
    "ConfigPath": (".language", "ConfigPath"),
    "ConfigPathPart": (".language", "ConfigPathPart"),
    "CyclicPoseTargetCfg": (".language", "CyclicPoseTargetCfg"),
    "TaskProgramCfg": (".language", "TaskProgramCfg"),
    "TaskProgramCompileError": (".compiler", "TaskProgramCompileError"),
    "TaskProgramCompiler": (".compiler", "TaskProgramCompiler"),
    "TaskProgramConfigError": (".language", "TaskProgramConfigError"),
    "TaskProgramDecodeError": (".language", "TaskProgramDecodeError"),
    "TaskProgramIntegrationCfg": (".language", "TaskProgramIntegrationCfg"),
    "TaskProgramValidationContext": (".language", "TaskProgramValidationContext"),
    "TaskProgramValidationError": (".language", "TaskProgramValidationError"),
    "HandOverCfg": (".language", "HandOverCfg"),
    "InvokeCfg": (".language", "InvokeCfg"),
    "ObjectNearRelativeTargetValidatorCfg": (
        ".language",
        "ObjectNearRelativeTargetValidatorCfg",
    ),
    "ObjectNearTargetValidatorCfg": (".language", "ObjectNearTargetValidatorCfg"),
    "ObjectLiftedValidatorCfg": (".language", "ObjectLiftedValidatorCfg"),
    "ObjectRelativePositionValidatorCfg": (
        ".language",
        "ObjectRelativePositionValidatorCfg",
    ),
    "ObjectUprightValidatorCfg": (".language", "ObjectUprightValidatorCfg"),
    "ParallelCfg": (".language", "ParallelCfg"),
    "PoseTargetCfg": (".language", "PoseTargetCfg"),
    "SceneRelativePoseTargetCfg": (".language", "SceneRelativePoseTargetCfg"),
    "MoveHeldObjectCfg": (".language", "MoveHeldObjectCfg"),
    "HoldPostCfg": (".language", "HoldPostCfg"),
    "NormalizedTaskProgram": (".language", "NormalizedTaskProgram"),
    "normalize_task_program": (".language", "normalize_task_program"),
    "PickCfg": (".language", "PickCfg"),
    "PlaceCfg": (".language", "PlaceCfg"),
    "PoseCfg": (".language", "PoseCfg"),
    "RegisteredSemanticCallCfg": (".language", "RegisteredSemanticCallCfg"),
    "RepeatCfg": (".language", "RepeatCfg"),
    "SceneReferenceRole": (".language", "SceneReferenceRole"),
    "SegmentCfg": (".language", "SegmentCfg"),
    "SequenceCfg": (".language", "SequenceCfg"),
    "TargetRefCfg": (".language", "TargetRefCfg"),
    "WaitStablePostCfg": (".language", "WaitStablePostCfg"),
    "decode_task_program": (".language", "decode_task_program"),
    "load_task_program": (".language", "load_task_program"),
    "loads_task_program_json": (".language", "loads_task_program_json"),
    "parse_task_program_json": (".language", "parse_task_program_json"),
    "render_config_path": (".language", "render_config_path"),
    "validate_task_program": (".language", "validate_task_program"),
}


def __getattr__(name: str) -> Any:
    """Resolve the existing public object only when its capability is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
