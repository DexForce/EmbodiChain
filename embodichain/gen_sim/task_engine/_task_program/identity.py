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

"""GenSim bundle identity from complete validated runtime declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import canonical_hash

if TYPE_CHECKING:
    from embodichain.lab.task_program.integrations.catalog import (
        SimulationTaskProgramRegistration,
    )

__all__: list[str] = []


def bundle_registration_fingerprint(
    registration: SimulationTaskProgramRegistration,
) -> str:
    """Validate declaration drift and hash the complete canonical snapshot."""
    return canonical_hash(registration.fingerprint_payload())
