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
"""Compare installed kernels on frozen S04 topology and measured deformed states.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/verify_sdk_kernel.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import newton
import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from dexsim.engine.newton_physics import solver_runtime
from scripts.benchmark.tech_reports.physics.experiments.dexuni.remeasure_sdk import (
    AUDIT,
    ROOTS,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    write_json,
)

__all__ = ["main"]


def main() -> None:
    """Check every color, all five EC runs and rest/mid/final states without timing."""
    solver_runtime.ensure_solver_available("dexuni")
    module = importlib.import_module("newton._src.solvers.mjvbd_v2.vbd.solver_vbd")
    kernels = importlib.import_module(
        "newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels"
    )
    attempts = [
        a
        for a in json.loads((AUDIT / "remeasurement_attempts.json").read_text())
        if a["task"] == "s04" and a["platform"] == "embodichain"
    ]
    source = ROOTS["s04"] / attempts[0]["run"] / "model.npz"
    data = dict(np.load(source))
    builder = newton.ModelBuilder()
    module.SolverVBD.register_custom_attributes(builder)
    for pos, mass in zip(data["particle_q"], data["particle_mass"]):
        builder.add_particle(wp.vec3(pos), wp.vec3(), float(mass))
    for tet, material in zip(data["tet_indices"], data["tet_materials"]):
        builder.add_tetrahedron(*map(int, tet), *map(float, material))
    for tri in data["tri_indices"]:
        builder.add_triangle(*map(int, tri), tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
    builder.color(include_bending=True)
    model = builder.finalize(device="cuda:0")
    solver = module.SolverVBD(model, particle_enable_uniform_tet_specialization=True)
    assert solver.particle_uniform_tet_status == "specialized"
    np.testing.assert_array_equal(model.particle_q.numpy(), data["particle_q"])
    np.testing.assert_array_equal(model.tet_indices.numpy(), data["tet_indices"])
    np.testing.assert_array_equal(model.tet_materials.numpy(), data["tet_materials"])
    rng = np.random.default_rng(134)
    solver.particle_forces.assign(
        rng.normal(0, 0.001, size=(441, 3)).astype(np.float32)
    )
    count = 0
    max_delta = max_hessian = 0.0
    covered = set()
    for attempt in attempts:
        states = np.load(ROOTS["s04"] / attempt["run"] / "states.npz")
        for frame in (0, 360, 720):
            solver.particle_q_prev.assign(states["particles"][max(0, frame - 1)])
            solver.inertia.assign(states["particles"][frame])
            for perturbation in (0.0, 0.0001, 0.001):
                pos = wp.array(
                    states["particles"][frame]
                    + rng.normal(0, perturbation, size=(441, 3)).astype(np.float32),
                    dtype=wp.vec3,
                    device=model.device,
                )
                for group in model.particle_color_groups:
                    covered.update(map(int, group.numpy()))
                    results = []
                    seed = rng.normal(0, 0.00001, size=(441, 3)).astype(np.float32)
                    for kernel in (
                        kernels.solve_elasticity_tile,
                        solver._particle_elasticity_kernel,
                    ):
                        delta = wp.array(seed, dtype=wp.vec3, device=model.device)
                        hessian = wp.zeros(441, dtype=wp.mat33, device=model.device)
                        inputs = [
                            1 / 600,
                            group,
                            solver.particle_q_prev,
                            pos,
                            model.particle_mass,
                            solver.inertia,
                            model.particle_flags,
                            model.tri_indices,
                            model.tri_poses,
                            model.tri_materials,
                            model.tri_areas,
                            model.edge_indices,
                            model.edge_rest_angle,
                            model.edge_rest_length,
                            model.edge_bending_properties,
                            model.tet_indices,
                            model.tet_poses,
                            model.tet_materials,
                            solver.particle_adjacency,
                            solver.particle_forces,
                            solver.particle_hessians,
                            True,
                            hessian,
                            delta,
                        ]
                        wp.launch(
                            kernel,
                            dim=group.size * 16,
                            block_dim=16,
                            inputs=inputs,
                            device=model.device,
                        )
                        results.append((delta.numpy(), hessian.numpy()))
                    assert all(
                        np.isfinite(x).all() for result in results for x in result
                    )
                    max_delta = max(
                        max_delta, float(np.abs(results[0][0] - results[1][0]).max())
                    )
                    max_hessian = max(
                        max_hessian, float(np.abs(results[0][1] - results[1][1]).max())
                    )
                    for baseline, specialized in zip(*results):
                        np.testing.assert_array_equal(baseline, specialized)
                    count += 1
        states.close()
    assert covered == set(range(441))
    write_json(
        AUDIT / "frozen_state_kernel_validation.json",
        {
            "scope": "Particle-only reconstruction from frozen world-space positions/topology/masses/materials; rest poses and adjacency recomputed with Newton builder. Uses recorded deformed particle states; external forces identical in both arms. This is a local kernel comparison, not whole-trajectory bitwise equivalence.",
            "runs": 5,
            "state_frames_per_run": [0, 360, 720],
            "perturbation_m": [0, 0.0001, 0.001],
            "color_groups": len(model.particle_color_groups),
            "covered_particles": len(covered),
            "kernel_comparisons": count,
            "max_displacement_error": max_delta,
            "max_hessian_error": max_hessian,
            "kernel_file": kernels.__file__,
            "kernel_sha256": hashlib.sha256(
                Path(kernels.__file__).read_bytes()
            ).hexdigest(),
            "passed": True,
        },
    )
    print(
        f"Passed {count} color/state/perturbation comparisons; max errors {max_delta}, {max_hessian}"
    )


if __name__ == "__main__":
    main()
