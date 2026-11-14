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

"""Shared physical scene declarations for reference tasks and validation."""

from __future__ import annotations
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from ._scene_assets import _ant_reference_friction, make_reference_articulation

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import NewtonSolverCfg
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects.articulation import Articulation

__all__: list[str] = []


def reference_friction(
    robot: str,
    physics_profile: str,
    solver_cfg: Mapping[str, Any] | NewtonSolverCfg | None,
) -> dict[str, Any] | None:
    """Resolve the model-recorded contact domain, validating coupled friction."""
    if not isinstance(solver_cfg, dict):
        solver_cfg = (
            dict(solver_cfg)
            if isinstance(solver_cfg, Mapping)
            else {
                "solver_type": getattr(solver_cfg, "solver_type", None),
                "options": getattr(solver_cfg, "options", {}),
            }
        )
    featherstone = solver_cfg.get("solver_type") == "featherstone"
    neural = solver_cfg.get("solver_type") == "nerd"
    friction_mapping = None
    if robot == "ant" and physics_profile == "warp_reference":
        if featherstone:
            friction_mapping = _ant_reference_friction()
            delta = friction_mapping["friction_smoothing"]
            requested = solver_cfg.setdefault("friction_smoothing", delta)
            if isinstance(requested, bool) or requested != delta:
                raise ValueError(
                    "Ant warp_reference requires the coupled contact mapping: "
                    f"friction_smoothing={delta} with scaled shape/ground mu."
                )
        elif neural:
            # Older exports predate the Featherstone friction conversion.
            # Their original material domain must still be reconstructed;
            # the provider then validates the actual physics signature.
            options = solver_cfg.get("options", {})
            checkpoint = Path(options["checkpoint"])
            exported = json.loads((checkpoint / "manifest.json").read_text())
            recorded = exported.get("reference_friction_mapping")
            if recorded is not None:
                expected = _ant_reference_friction()
                if not isinstance(recorded, Mapping) or any(
                    recorded.get(key) != value
                    for key, value in expected.items()
                    if key != "interpretation"
                ):
                    raise ValueError("Unsupported Ant reference friction mapping")
                friction_mapping = dict(recorded)
    return friction_mapping


def declare_reference_scene(
    sim: SimulationManager,
    robot: str,
    physics_profile: str,
    friction_mapping: Mapping[str, Any] | None,
) -> Articulation:
    """Declare the same robot and ground for neural and analytical solvers."""
    if robot == "ant" and physics_profile == "warp_reference":
        # The old ModelBuilder's shared plane has different material
        # defaults from SimulationManager's ordinary default plane.
        from dexsim.spawn import (
            CollisionDesc,
            GeometryDesc,
            NewtonCollisionDesc,
            ObjectDesc,
            RigidBodyPhysicsDesc,
        )

        collision = CollisionDesc.from_geometry(GeometryDesc.plane(1000.0))
        collision.newton = NewtonCollisionDesc(
            ke=1e5,
            kd=1e3,
            kf=1e3,
            mu=(friction_mapping["ground_mu"] if friction_mapping else 0.5),
            margin=1e-5,
            gap=0.05,
            mu_torsional=0.0,
            mu_rolling=0.0,
        )
        sim.add_rigid_object_descriptor(
            ObjectDesc(
                name="reference_ground",
                collisions=[collision],
                physics=RigidBodyPhysicsDesc.static(),
                per_env=False,
            )
        )
    descriptor = make_reference_articulation(robot, physics_profile)
    if friction_mapping is not None:
        # Scale every freshly authored Ant shape, including the shared
        # plane above, before Spawn finalizes the physical model.
        for link in descriptor.links:
            for collision in link.collisions:
                collision.newton.mu *= friction_mapping["material_mu_scale"]
    return sim.add_articulation_descriptor(descriptor)
