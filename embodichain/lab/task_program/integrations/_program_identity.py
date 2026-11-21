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

"""Shared identity for compiled-program integration bindings."""

from __future__ import annotations

import hashlib
import json

from embodichain.lab.task_program.compiler.program import CompiledTaskProgram
from .catalog import _canonical_value

__all__: list[str] = []


def task_program_fingerprint(program: CompiledTaskProgram) -> str:
    """Return the canonical SHA-256 shared by compiled-program bindings.

    Args:
        program: Exact compiled program whose full declaration is identified.

    Returns:
        Lowercase digest of program ID, instruction, integration selection and
        expanded segments, using the established nominal-source identity format.

    Raises:
        TypeError: If the input is not an exact CompiledTaskProgram.
    """
    if type(program) is not CompiledTaskProgram:
        raise TypeError("program must be a CompiledTaskProgram")
    payload = _canonical_value(
        (
            program.program_id,
            program.instruction,
            program.integration,
            tuple(program.iter_segments()),
        )
    )
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
