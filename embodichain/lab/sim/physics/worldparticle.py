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
"""WorldParticle native inference adapter for DexSim's provider lifecycle."""

from __future__ import annotations
from typing import Any

__all__ = ["WorldParticleProvider"]
import math
from dataclasses import replace

from dexsim.engine.newton_physics.solver_extensions import (
    NewtonSolverCapabilities,
    NewtonSolverExtension,
)


class WorldParticleProvider:
    """Select one of the four local official WorldParticle networks with explicit static scene inputs.

    Options: ``checkpoint`` and ``config`` paths, an ``inputs`` NPZ with raw
    sample-ordered raw features/boundaries/CSR topology, and optional
    ``model_to_world`` rotation. Cloth uses its authored surface; duck
    deformation uses an unconnected-particle carrier, not an SPH model.
    One CUDA particle world and static network boundaries are supported. This
    provider supplies inference and native reset, not functional gradients,
    native contact/force reports, dynamic rigid bodies, or CUDA Graph capture.
    Boundary points are model inputs, not automatically generated collisions.
    """

    capabilities = NewtonSolverCapabilities(
        devices=("cuda",),
        reset=True,
        cuda_graph=False,
        reduced_coordinates=False,
        external_wrenches=False,
        supported_control_fields=(),
        runtime_controls=False,
        supported_particle_kinds=("fluid",),
    )

    def register_custom_attributes(self, builder: Any, options: dict[str, Any]) -> None:
        """Declare the authored topology kind before particle materialization."""
        from embodichain_models.dynamics.worldparticle._runtime import load_config

        cfg = load_config(options.get("config"))
        kind = "cloth" if cfg["data"] == "cloth_on_ball" else "fluid"
        self.capabilities = replace(
            type(self).capabilities, supported_particle_kinds=(kind,)
        )

    def create(self, context: Any, options: dict[str, Any]) -> NewtonSolverExtension:
        """Validate the scene before allocating network weights on the GPU."""
        import numpy as np
        import torch
        from embodichain_models.dynamics.worldparticle import _runtime
        from embodichain_models.dynamics.worldparticle.model import WorldParticleModel

        load_assets = _runtime.load_assets

        model = context.model
        if not context.device.is_cuda:
            raise ValueError("WorldParticle requires CUDA")
        if model.requires_grad:
            raise ValueError(
                "WorldParticle currently supports inference only (requires_grad=False)"
            )
        if (
            model.body_count
            or model.joint_count
            or model.shape_count
            or model.world_count > 1
        ):
            raise ValueError(
                "WorldParticle supports one particle world without rigid bodies, joints or native collision shapes"
            )
        if not model.particle_count:
            raise ValueError("WorldParticle requires a nonempty fluid particle set")
        cfg, arrays, rotation = load_assets(options)
        if len(arrays[0]) != model.particle_count:
            raise ValueError(
                "Input features must match the flattened Newton particle order/count"
            )
        if not math.isclose(context.dt, cfg["timestep"], rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("Newton substep dt must match the checkpoint timestep")
        expected_gravity = rotation @ np.asarray(cfg["gravity"], dtype=np.float32)
        gravity = model.gravity.numpy().reshape(-1, 3)
        if not np.allclose(gravity, expected_gravity, atol=1e-6):
            raise ValueError(
                "Newton gravity must equal model_to_world @ checkpoint gravity"
            )
        inverse_mass = model.particle_inv_mass.numpy()
        if not np.isfinite(inverse_mass).all() or np.any(inverse_mass <= 0):
            raise ValueError("Initially fixed/massless particles are not supported")
        from newton import ParticleFlags

        if np.any((model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0):
            raise ValueError("Initially inactive particles are not supported")
        from dexsim.engine.newton_physics.particle_inference import (
            ParticleInferenceSolver,
        )

        def create_predictor(device):
            inputs = tuple(torch.as_tensor(value, device=device) for value in arrays)
            network = _runtime.load_network(cfg, options["checkpoint"], device)
            return WorldParticleModel(network, inputs, rotation, cfg)

        solver = ParticleInferenceSolver(model, create_predictor, context.dt)
        return NewtonSolverExtension(solver, self.capabilities, close=solver.close)
