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
"""Verify E8's fixed-mass source-to-actor conversion without native writes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.lab.sim._inertia import _principal_inertia_matrix

__all__: list[str] = []


def verify_scaled_mass(art: Any, binding: Any) -> dict[str, Any]:
    """Fail closed if a backend ignores or repeats the deployment conversion."""
    from .twist_mass_source import adjacent_manifest_path, qualify_mass_source_copy

    runtime_source = Path(art.cfg.fpath)
    lineage_digest = getattr(binding, "mass_lineage_sha256", "")
    mass_copy = None
    if lineage_digest:
        mass_copy = qualify_mass_source_copy(
            runtime_source, manifest_sha256=lineage_digest
        )
        if (
            getattr(art.cfg, "body_scale_mass_policy", "native") != "native"
            or mass_copy.scale != float(binding.scale)
            or mass_copy.runtime_sha256 != binding.source_sha256
        ):
            raise ValueError("E8 mass deployment conflicts with its scale or policy.")
    elif adjacent_manifest_path(runtime_source).exists():
        raise ValueError("E8 deployment mass source requires bound lineage.")
    if (
        mass_copy is None
        and getattr(art.cfg, "body_scale_mass_policy", "native") != "fixed_mass"
    ):
        return {"policy": "native", "verified": False, "scope": "not_requested"}
    source = runtime_source if mass_copy is None else mass_copy.source
    expected_digest = (
        binding.source_sha256 if mass_copy is None else mass_copy.source_sha256
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("E8 mass verification source hash differs from its binding.")
    from dexsim.kit.usd import parse_usd

    scene = parse_usd(str(source))
    if len(scene.articulations) != 1:
        raise ValueError("E8 mass verification requires one source articulation.")
    scale = float(binding.scale)
    records = []
    for link in scene.articulations[0].links:
        body = link.rigid_body
        if body is None:
            continue
        if body.mass is None or body.com_position is None or body.inertia is None:
            raise ValueError(
                "E8 fixed-mass verification requires authored mass, COM and inertia."
            )
        mass_rows = art.get_mass(link.name).detach().cpu().numpy()
        inertia_rows = art.get_inertia(link.name).detach().cpu().numpy()
        com_rows = art.get_com_pose(link.name).detach().cpu().numpy()
        if (
            mass_rows.ndim != 2
            or mass_rows.shape[1:] != (1,)
            or inertia_rows.shape != (*mass_rows.shape, 3)
            or com_rows.shape != (*mass_rows.shape, 7)
            or not mass_rows.shape[0]
            or mass_rows.shape[0] != art.num_instances
        ):
            raise ValueError(
                "E8 mass-property readback tensor shapes are inconsistent."
            )
        mass = mass_rows[:, 0]
        moments = inertia_rows[:, 0]
        com = com_rows[:, 0]
        expected_com = np.asarray(body.com_position, dtype=float) * scale
        expected_inertia = np.asarray(body.inertia, dtype=float) * scale**2
        if (
            not np.isfinite(body.mass)
            or body.mass <= 0
            or not np.isfinite(expected_com).all()
            or expected_inertia.shape != (3, 3)
            or not np.isfinite(expected_inertia).all()
            or not np.allclose(expected_inertia, expected_inertia.T, atol=1e-12)
            or np.linalg.eigvalsh(expected_inertia).min() <= 0
        ):
            raise ValueError("E8 authored mass properties must be positive and finite.")
        # The SDK drops nearly diagonal off-diagonals at 1e-5 of the tensor
        # scale, with a float32 epsilon floor, while retaining its chosen frame.
        inertia_atol = 1e-5 * max(
            float(np.abs(expected_inertia).max()), float(np.finfo(np.float32).eps)
        )
        if len(mass) != len(com) or len(mass) != len(moments) or not len(mass):
            raise ValueError("E8 mass-property readback rows are inconsistent.")
        for env, (m, diagonal, pose) in enumerate(zip(mass, moments, com, strict=True)):
            values = np.r_[m, diagonal, pose]
            if (
                not np.isfinite(values).all()
                or m <= 0
                or (diagonal <= 0).any()
                or not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-5)
            ):
                raise ValueError(
                    "E8 mass-property readbacks must be finite and normalized."
                )
            actual_inertia = _principal_inertia_matrix(diagonal, pose[3:])
            accepted = (
                np.isclose(m, body.mass, rtol=1e-5, atol=1e-8)
                and np.allclose(pose[:3], expected_com, rtol=1e-5, atol=1e-7)
                and np.allclose(
                    actual_inertia, expected_inertia, rtol=0, atol=inertia_atol
                )
            )
            if not accepted:
                raise ValueError(
                    f"E8 scaled mass readback mismatch for {link.name!r}, env {env}: "
                    f"mass={float(m)} expected={body.mass}, "
                    f"COM={pose[:3].tolist()} expected={expected_com.tolist()}, "
                    f"inertia={actual_inertia.tolist()} expected={expected_inertia.tolist()}."
                )
            records.append(
                {
                    "link": link.name,
                    "env_id": env,
                    "mass": float(m),
                    "com_local": pose[:3].tolist(),
                    "inertia_body_about_com": actual_inertia.tolist(),
                }
            )
    if not records:
        raise ValueError("E8 mass verification found no authored physical links.")
    result = {
        "policy": "fixed_mass",
        "verified": True,
        "scale": scale,
        "source_sha256": binding.source_sha256,
        "source_edited": False,
        "scope": "native mass-property readback; not collision cooking or task success",
        "links": records,
    }
    if mass_copy is not None:
        result.update(
            conversion_owner="gen_sim_deployment_source",
            origin_source_sha256=mass_copy.source_sha256,
            mass_lineage_sha256=lineage_digest,
        )
    return result
