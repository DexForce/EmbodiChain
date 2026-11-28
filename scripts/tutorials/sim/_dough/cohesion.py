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

"""Adapt the DexSim bread example's cohesive material to a tabletop fixture."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import dexsim
import numpy as np
import warp as wp
from dexsim.engine.newton_physics import GraphRuntimeControl


def reference_example(directory: Path | None):
    """Load the original example functions from a matching DexSim source checkout."""
    if directory is None and os.getenv("DEXSIM_EXAMPLES"):
        directory = Path(os.environ["DEXSIM_EXAMPLES"])
    if directory is None:
        for parent in Path(dexsim.__file__).resolve().parents:
            candidate = parent / "examples/python/physics/multiphys"
            if (candidate / "demo_dexuni_mjmpm_w1_bread_cut.py").is_file():
                directory = candidate
                break
    if (
        directory is None
        or not (directory / "demo_dexuni_mjmpm_w1_bread_cut.py").is_file()
    ):
        raise FileNotFoundError(
            "Supply --dexsim-examples pointing to DexSim's examples/python/physics/multiphys directory."
        )
    sys.path.insert(0, str(directory.resolve()))
    try:
        return importlib.import_module("demo_dexuni_mjmpm_w1_bread_cut")
    finally:
        sys.path.pop(0)


class BreadCut(GraphRuntimeControl):
    """Run the reference swept-edge damage and implicit per-substep cohesion."""

    def __init__(
        self,
        ref: ModuleType,
        points: np.ndarray,
        damage: bool,
        *,
        mass: float,
        spacing: float,
        yield_stress: float,
        tensile_yield_ratio: float,
    ) -> None:
        self.ref = ref
        self.points = points
        self.enabled = damage
        self.mass = mass
        self.spacing = spacing
        self.yield_stress = yield_stress
        self.tensile_yield_ratio = tensile_yield_ratio

    def initialize(self, context):
        start, n = context.particle_range("arena_0/bread")
        if n != len(self.points):
            raise ValueError("Unexpected bread sample count.")
        self.region = slice(start, start + n)
        self.tool = context.body_index("arena_0/fixture", "utensil")
        self.local = wp.transform_identity()
        self.damage = wp.zeros(n, dtype=float, device=context.device)
        self.materials = tuple(
            zip(
                context.mpm_material_views("arena_0/bread", "yield_stress"),
                context.mpm_material_views("arena_0/bread", "tensile_yield_ratio"),
                strict=True,
            )
        )
        initial = wp.transform(*context.current_state.body_q.numpy()[self.tool])
        self.network = self.ref.CuttableCohesion(
            self.points,
            mass=self.mass,
            spacing=self.spacing,
            device=context.device,
            tool_pose=initial,
            plastic_zone=self.damage,
            plastic_yield=0.005,
            plastic_limit=0.6,
        )

    def __call__(self, context, substep_index, substep_count, substep_dt):
        state = SimpleNamespace(
            particle_q=context.current_state.particle_q[self.region],
            particle_qd=context.current_state.particle_qd[self.region],
            body_q=context.current_state.body_q,
        )
        if self.enabled and substep_index == 0:
            self.network.cut(state, self.tool, self.local)
            wp.launch(
                self.ref.damage_at_blade,
                len(self.points),
                inputs=[
                    state.particle_q,
                    state.body_q,
                    self.tool,
                    self.local,
                    0.0055,
                    self.damage,
                ],
                device=context.device,
            )
            for shear, tension in self.materials:
                wp.launch(
                    self.ref.apply_damage,
                    len(self.points),
                    inputs=[
                        self.damage,
                        self.yield_stress,
                        self.tensile_yield_ratio,
                        shear,
                        tension,
                    ],
                    device=context.device,
                )
        self.network.apply(state, substep_dt)
