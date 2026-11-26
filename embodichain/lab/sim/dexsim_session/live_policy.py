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

"""Bind operator-approved trained policies on a qualified live Scene."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__: list[str] = []


def create_policy(context, binding: dict[str, Any], output: Path):
    profile = binding["profile"]
    catalog = Path(os.environ["EMBODICHAIN_LIVE_POLICY_CATALOG"]).resolve(strict=True)
    recipe = json.loads(catalog.read_text())[profile]
    files = {
        k: (catalog.parent / p).resolve(strict=True) for k, p in recipe["files"].items()
    }
    roles = binding["paths"]
    if (
        context.scene.backend != "newton"
        or not context.world.is_physics_manually_update()
    ):
        raise ValueError("Policy requires a manual Newton World")
    floor = context.scene.get_object(roles["floor"])
    if floor.get_actor_type().name.lower() != "static":
        raise ValueError("Expected an existing static floor")
    if profile == "anymal-c":
        from embodichain.extensions.models.policies.anymal_c.borrowed import (
            AttachedVelocity,
        )

        policy = AttachedVelocity(
            context, files, roles["robot"], (0.4, 0.0, 0.0), output
        )
    else:
        from .rp1_policy import AttachedCarry

        policy = AttachedCarry(
            context,
            files,
            roles,
            output,
            clearance=recipe.get("lead_clearance_m", 0.06),
        )
    policy.profile = profile
    policy.roles = dict(roles)
    return policy


def operations(policy):
    return {
        "capabilities": lambda: dict(
            profile=policy.profile, roles=policy.roles, **policy.progress()
        ),
        "start_policy": policy.start,
        "cancel_run": policy.jobs.cancel,
    }
