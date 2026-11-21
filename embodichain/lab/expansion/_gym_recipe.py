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

"""Build source-pinned immutable recipes independently of World mutation."""

from __future__ import annotations
import json

from .recipes import (
    ExpansionRecipe,
    ExpansionContext,
    RecipeOperation,
    UniformRangeOperator,
)

from ._gym_context import GymHostContext

__all__: list[str] = []


class GymRecipeBuilder:
    """Build source-pinned immutable recipes independently of World mutation."""

    def __init__(self, context: GymHostContext) -> None:
        self.context = context

    def recipe(self, ordinal: int) -> ExpansionRecipe:
        """Sample an immutable logical recipe before assigning a physical slot."""
        ctx = self.context
        cfg = ctx.cfg
        context = ExpansionContext(
            cfg.source, ctx.report["recipe_revision"], cfg.seed, ordinal
        )
        physical = {
            kind: dict(UniformRangeOperator(kind, getattr(cfg, kind)).sample(context))
            for kind in (
                "mass_scale",
                "friction",
                "joint_stiffness_scale",
                "joint_damping_scale",
            )
            if kind in {"mass_scale", "friction"} or getattr(cfg, kind)
        }
        if cfg.center_of_mass:
            physical["center_of_mass"] = {
                uid: [
                    context.random("center_of_mass", "1", f"{uid}/{axis}").uniform(
                        *bounds
                    )
                    for axis, bounds in enumerate(axes)
                ]
                for uid, axes in cfg.center_of_mass.items()
            }
        operations = [
            RecipeOperation(
                "physical_parameters",
                "1",
                ctx.supported_operations["physical_parameters"],
                json.dumps(physical),
            )
        ]
        if ctx.scene_source:
            candidate = ctx.scene_source.candidate(ordinal)
            operations.append(
                RecipeOperation(
                    "joint_scene",
                    "1",
                    ctx.supported_operations["joint_scene"],
                    json.dumps(
                        {"candidate_id": candidate.candidate_id, "ordinal": ordinal}
                    ),
                )
            )
        if cfg.visual_profile:
            operations.append(
                RecipeOperation(
                    "visual",
                    "1",
                    ctx.supported_operations["visual"],
                    json.dumps(ctx.visual_registry[cfg.visual_profile]),
                )
            )
        motion = {
            "recipe_index": (
                ordinal % len(ctx.motion_recipes) if ctx.motion_recipes else None
            ),
            "source_recipe": (
                ctx.profile.recipes[ordinal % len(ctx.profile.recipes)]
                if ctx.profile
                else None
            ),
        }
        operations.append(
            RecipeOperation(
                "motion", "1", ctx.supported_operations["motion"], json.dumps(motion)
            )
        )
        order = ("joint_scene", "physical_parameters", "visual", "motion")
        operations.sort(key=lambda operation: order.index(operation.name))
        return ExpansionRecipe(context, ctx.compatibility, tuple(operations))
