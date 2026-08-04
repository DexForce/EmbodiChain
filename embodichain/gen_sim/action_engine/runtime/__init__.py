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

"""Runtime API for the compositional Action Engine."""

from __future__ import annotations

from .executor import ProgramExecutor
from .loader import load_agent_execution_program, load_execution_program
from .models import ExecutionProgram, ExecutionResult
from .predicates import PREDICATE_TYPES, evaluate_predicate

__all__ = [
    "ExecutionProgram",
    "PREDICATE_TYPES",
    "ExecutionResult",
    "ProgramExecutor",
    "evaluate_predicate",
    "load_agent_execution_program",
    "load_execution_program",
]
