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

"""FK-only reference model construction without a World, Scene or solver."""

from __future__ import annotations

from typing import Any

import newton

from ._scene_assets import make_reference_articulation

__all__: list[str] = []


def reference_ground_options(friction: dict[str, Any] | None) -> dict[str, float]:
    """Keep the reference teacher's shared plane explicit in both build paths."""
    return dict(
        ke=1e5,
        kd=1e3,
        kf=1e3,
        mu=friction["ground_mu"] if friction else 0.5,
        margin=1e-5,
        gap=0.05,
        mu_torsional=0.0,
        mu_rolling=0.0,
    )


def make_reference_model(
    robot: str,
    device: str,
    num_envs: int,
    physics_profile: str,
    friction: dict[str, Any] | None = None,
) -> Any:
    """Compile the same authored geometry as native reference deployment.

    Reuse the SDK's static descriptor lowering rather than duplicating its
    transform decomposition and inertia rounding. The SDK-internal lowering
    boundary is isolated here; strict physical-signature admission and native
    parity tests protect upgrades. With ``reuse_render=True`` the reference
    descriptors need no skeleton, render body, World or Scene. No solver is
    constructed. This adapter is not a general-purpose importer.
    """
    from dexsim.spawn.adapters.newton_articulation_adapter import (
        _translate_explicit_articulation,
    )
    from dexsim.engine.newton_physics.articulation.articulation import (
        NewtonArticulation,
    )
    from dexsim.engine.newton_physics._runtime import create_model_builder

    descriptor = make_reference_articulation(robot, physics_profile)
    if friction is not None:
        for link in descriptor.links:
            for collision in link.collisions:
                collision.newton.mu *= friction["material_mu_scale"]
    template = create_model_builder()
    specs, joints = _translate_explicit_articulation(
        None, None, descriptor, {}, "featherstone", reuse_render=True
    )
    articulation = NewtonArticulation(
        template,
        None,
        descriptor.name,
        descriptor.root_link_name,
        specs,
        joints,
        build_device=device,
    )
    articulation.finalize(
        descriptor.pose,
        fixed_base=descriptor.fixed_base,
        disable_self_collision=True,
    )
    builder = create_model_builder(gravity=(0.0, 0.0, -9.81))
    if robot == "ant":
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(**reference_ground_options(friction))
        )
    for _ in range(num_envs):
        builder.add_world(template)
    return builder.finalize(device=device)
