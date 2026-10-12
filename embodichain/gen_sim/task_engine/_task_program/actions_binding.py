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

"""Resource binding resolution shared by the v40 GenSim state boundary."""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_bound_resource", "resolve_shared_resource"]


def resolve_bound_resource(binding: Any, *, participant: str) -> str:
    """Resolve and validate one bound primary motion/grasp resource."""
    motion = binding.endpoint("primary", "motion")
    grasp = binding.endpoint("primary", "grasp")
    motion_key = getattr(motion, "task_state_key", None)
    grasp_key = getattr(grasp, "task_state_key", None)
    if not isinstance(motion_key, str) or not motion_key:
        raise ValueError(f"{participant} requires a non-empty motion resource key.")
    if motion_key != grasp_key:
        raise ValueError(
            f"{participant} requires motion and grasp to share one state resource."
        )
    return motion_key


def resolve_shared_resource(request: Any, *, participant: str) -> str:
    """Resolve and validate the primary motion/grasp resource for one call."""
    return resolve_bound_resource(request.binding, participant=participant)
