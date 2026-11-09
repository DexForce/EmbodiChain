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

"""Offline Task Program evidence protocol validation and acquisition declarations."""

from __future__ import annotations

from .control_evidence import normalize_control_evidence, validate_control_declaration
from .handover_evidence import HandOverEvidenceBinding, normalize_handover_evidence
from .coordinated_evidence import (
    CoordinatedHoldEvidenceBinding,
    normalize_coordinated_hold_evidence,
)
from .transport_evidence import (
    CoordinatedTransportEvidenceBinding,
    normalize_coordinated_transport_evidence,
)
from .observation_requests import (
    held_observation_request,
    sampled_support_window,
    directional_observation_axes,
    resolve_hold_observation_condition,
)
from .placement_evidence import (
    normalize_placement_evidence,
    normalize_pour_release_evidence,
    validate_placement_declaration,
)

__all__ = [
    "normalize_control_evidence",
    "validate_control_declaration",
    "HandOverEvidenceBinding",
    "normalize_handover_evidence",
    "CoordinatedHoldEvidenceBinding",
    "normalize_coordinated_hold_evidence",
    "CoordinatedTransportEvidenceBinding",
    "normalize_coordinated_transport_evidence",
    "held_observation_request",
    "resolve_hold_observation_condition",
    "sampled_support_window",
    "directional_observation_axes",
    "normalize_placement_evidence",
    "normalize_pour_release_evidence",
    "validate_placement_declaration",
]
