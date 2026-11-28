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

"""Single-arena DexUni setup for the procedural material-point examples."""

from __future__ import annotations

import numpy as np
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects.deformable import DeformableObject
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    VolumeDeformableObjectCfg,
)

SPACING = 0.008
DT = 1 / 60


def create_scene(headless: bool) -> SimulationManager:
    physics = NewtonPhysicsCfg(
        device="cuda:0",
        physics_dt=DT,
        num_substeps=8,
        collision_cfg=None,
        solver_cfg={
            "solver_type": "dexuni",
            "particle_options": {
                "mpm_options": {
                    "grid_type": "sparse",
                    "grid_padding": 0,
                    "voxel_size": 2 * SPACING,
                    "max_active_cell_count": 8192,
                    "collider_basis": "pic128",
                }
            },
        },
    )
    return SimulationManager(
        SimulationManagerCfg(
            physics_cfg=physics,
            headless=headless,
            startup_summary="off",
            render_cfg=RenderCfg(renderer="no-render" if headless else "rt"),
        )
    )


def add_lattice(
    sim: SimulationManager,
    uid: str,
    dims: tuple[int, int, int],
    origin: tuple[float, float, float],
    density: float,
    **material: float,
) -> DeformableObject:
    """Sample a uniform volume and assign its physical mass from density."""
    return sim.add_deformable_object(
        VolumeDeformableObjectCfg.from_dict(
            {
                "uid": uid,
                "particle_radius": SPACING / 2,
                "model": {
                    "kind": "material_point",
                    "sampling": {
                        "kind": "lattice",
                        "dims": dims,
                        "origin": origin,
                        "spacing": SPACING,
                    },
                    "physics": {"particle_masses": density * SPACING**3, **material},
                },
            }
        )
    )


def open_view(
    sim: SimulationManager,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    """Frame this small tabletop scene in the native viewer."""
    sim.open_window()
    sim.viewer.set_look_at(
        eye=np.array(eye), look_at=np.array(target), up=np.array([0.0, 0.0, 1.0])
    )
