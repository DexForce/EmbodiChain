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

"""Measure intact stack geometry and unforced motion from saved states.

Run: imported by p02_stack_sweep; never initializes a simulator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.benchmark.tech_reports.reporting.p02_analysis import projected_overlap

__all__ = ["measure_stack"]


def measure_stack(path: Path, meta: dict) -> tuple[dict, dict]:
    """Return geometric events and continuous motion; no drift/angle quality gate."""
    with np.load(path) as d:
        t, state, size = d["t"], d["state"], d["body_size"]
        inertia = d["body_inertia"]
        initial = d["requested_initial_poses"]
        support_top = float(d["surface_origin"][2])
    assert np.isfinite(state).all()
    np.testing.assert_allclose(state[0, :, :3], initial[:, :3], atol=1e-6)
    np.testing.assert_array_equal(
        np.load(path.with_name(path.stem + "_state.npy"), mmap_mode="r"), state
    )
    bodies = state.shape[1]
    edges = np.ones((len(t), max(0, bodies - 1)), dtype=bool)
    for i in range(bodies - 1):
        edges[:, i] = projected_overlap(
            state[:, i, :7], size, state[:, i + 1, :7], size
        ) & (state[:, i + 1, 2] > state[:, i, 2])
    broken = (~edges).sum(axis=1)
    longest = np.ones(len(t), dtype=np.int32)
    run = longest.copy()
    for i in range(bodies - 1):
        run = np.where(edges[:, i], run + 1, 1)
        longest = np.maximum(longest, run)
    event = np.flatnonzero(broken > 0)
    rotation = (
        Rotation.from_quat(state[..., 3:7].reshape(-1, 4))
        .as_matrix()
        .reshape(len(t), bodies, 3, 3)
    )
    speed = np.linalg.norm(state[..., 7:10], axis=-1)
    angular_local = np.einsum("tbij,tbi->tbj", rotation, state[..., 10:13])
    kinetic = (
        0.5 * 0.1 * (speed**2) + 0.5 * (angular_local**2 * inertia).sum(-1)
    ).sum(-1)
    potential = 0.1 * 9.81 * state[..., 2].sum(-1)
    energy = kinetic + potential
    radius_z = np.abs(rotation[:, :, 2, :]) @ (size / 2)
    top_height = (state[..., 2] + radius_z).max(axis=1) - support_top
    bottom_gap = state[:, 0, 2] - radius_z[:, 0] - support_top
    drift = np.linalg.norm(state[..., :2] - state[0, :, :2], axis=-1)
    tilt = np.rad2deg(np.arccos(np.clip(rotation[..., 2, 2], -1, 1)))
    result = dict(
        run_id=meta["run_id"],
        platform=meta["platform"],
        solver=meta["solver"],
        initial=int(path.stem.split("_")[-1]),
        layers=bodies,
        dt_s=meta["dt_s"],
        duration_s=float(t[-1]),
        source=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=meta["status"],
        first_geometry_event_s=float(t[event[0]]) if len(event) else None,
        intact_12s=not bool(np.any(broken[t <= 12 + 1e-10])),
        intact_full_window=not bool(len(event)),
        broken_neighbor_pairs_max=int(broken.max()),
        broken_neighbor_pairs_final=int(broken[-1]),
        longest_ordered_chain_min=int(longest.min()),
        longest_ordered_chain_final=int(longest[-1]),
        max_horizontal_drift_mm=float(drift.max() * 1000),
        max_tilt_deg=float(tilt.max()),
        max_speed_m_s=float(speed.max()),
        peak_kinetic_energy_J=float(kinetic.max()),
        initial_mechanical_energy_J=float(energy[0]),
        max_mechanical_energy_gain_J=float((energy - energy[0]).max()),
        max_bottom_gap_mm=float(bottom_gap.max() * 1000),
        final_tower_envelope_height_mm=float(top_height[-1] * 1000),
        diagnostic_wall_s=meta.get("diagnostic_wall_s_per_episode", [None])[0],
    )
    return result, dict(
        t=t,
        broken_pairs=broken,
        longest_ordered_chain=longest,
        top_envelope_height_m=top_height,
        bottom_gap_m=bottom_gap,
        max_speed_m_s=speed.max(-1),
        kinetic_J=kinetic,
        mechanical_energy_J=energy,
        max_horizontal_drift_m=drift.max(-1),
        max_tilt_deg=tilt.max(-1),
    )
