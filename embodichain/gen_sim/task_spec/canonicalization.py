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

"""TaskSpec canonical identities, using the shared JSON canonicalization contract."""

from __future__ import annotations
from typing import Final
from embodichain._json_contract import (
    canonicalize,
    canonical_json,
    canonical_hash,
    json_snapshot,
)

__all__ = [
    "CANONICALIZATION_VERSION",
    "canonical_hash",
    "canonical_json",
    "canonicalize",
    "json_snapshot",
]
CANONICALIZATION_VERSION: Final = "task_spec_c14n/v1"
