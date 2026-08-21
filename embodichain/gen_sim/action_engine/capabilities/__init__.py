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

"""Semantic operator and atomic-action capability registry."""

from __future__ import annotations

from .atomic import (
    ACTION_CONTRACT_VERSION,
    AtomicCapability,
    AtomicCapabilityRegistry,
    ResolvedActionContract,
    ResourceClaim,
    StateAtom,
    StateEffect,
    build_atomic_capability_registry,
    capability_precondition,
)
from .builtins import build_default_registry
from .registry import (
    ActionCapability,
    ActionTemplate,
    CapabilityRegistry,
    OperatorCapability,
    PhaseTemplate,
)

__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ActionCapability",
    "ActionTemplate",
    "AtomicCapability",
    "AtomicCapabilityRegistry",
    "CapabilityRegistry",
    "OperatorCapability",
    "PhaseTemplate",
    "ResolvedActionContract",
    "ResourceClaim",
    "StateAtom",
    "StateEffect",
    "build_atomic_capability_registry",
    "build_default_registry",
    "capability_precondition",
]
