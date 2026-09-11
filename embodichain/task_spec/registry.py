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

"""Versioned semantic vocabulary; this module registers no execution services."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = ["registry_snapshot"]

_OBJECTS = ["object", "container", "support"]
_PREDICATES = {
    "object_at_target": {
        "roles": {"object": _OBJECTS, "target": ["target"]},
        "quantities": {"tolerance": "length"},
        "meaning": "Object origin is within tolerance of a bound target origin.",
    },
    "relative_position": {
        "roles": {"object": _OBJECTS, "reference": _OBJECTS},
        "quantities": {"margin": "length"},
        "meaning": "Origin separation along scene X (right) or Y (front), at least margin; Z is up.",
    },
    "upright": {
        "roles": {"object": _OBJECTS},
        "quantities": {"max_tilt": "angle"},
        "meaning": "Object local +Z tilt from scene +Z is at most max_tilt.",
    },
    "stack_supported": {
        "roles": {"object": _OBJECTS, "support": _OBJECTS},
        "quantities": {
            "max_tilt": "angle",
            "max_gap": "length",
            "max_offset": "length",
        },
        "meaning": "Upright bounding-box support approximation, not contact or force proof.",
    },
    "held_stable": {
        "roles": {"object": _OBJECTS, "holder": ["manipulator"]},
        "quantities": {
            "position_tolerance": "length",
            "angular_tolerance": "angle",
            "duration": "time",
        },
        "meaning": "Observed held-relative pose drift over duration, not continuous contact proof.",
    },
    "released": {
        "roles": {"object": _OBJECTS, "holder": ["manipulator"]},
        "quantities": {},
        "meaning": "Observed detachment from the named holder, not global absence of contact.",
    },
    "joint_position": {
        "roles": {"joint": ["joint"]},
        "quantities": {"position": "angle", "tolerance": "angle"},
        "meaning": "Revolute joint position within tolerance, without angle wrapping.",
    },
}


def registry_snapshot() -> dict[str, Any]:
    """Return a detached semantic vocabulary snapshot.

    Returns:
        Closed role, asset-capability and predicate definitions for semantic
        version 0.1. Predicate revisions identify meaning, not installed
        checkers; hosts must report unavailable when observations are absent.
    """
    return {
        "semantic_version": "0.1",
        "max_roles": 6,
        "role_kinds": [
            "object",
            "container",
            "support",
            "target",
            "manipulator",
            "joint",
        ],
        "capabilities": [
            "graspable",
            "placeable",
            "orientable",
            "handover",
            "dual_graspable",
            "rigid",
        ],
        "predicates": {
            name: dict(deepcopy(spec), revision="1")
            for name, spec in _PREDICATES.items()
        },
    }
