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
"""Select DexUni's rigid articulation branch and translate its reset state."""

from __future__ import annotations

import copy
import types
from typing import Any

import newton
import numpy as np
import torch
import warp as wp
from newton import Model

__all__ = [
    "import_dexuni_mjcf",
    "select_all_articulations",
    "refresh_collision_filters",
    "DexUniStateAdapter",
]


def import_dexuni_mjcf(builder: Any, path: str) -> dict[str, Any]:
    """Parse MJCF after DexUni has registered its private solver attributes.

    Newton 1.7's parser registers standalone solver callbacks, which conflict
    with DexUni's callbacks. A function-local globals copy selects the private
    class while retaining the parser code and flags. Parser-only attributes
    absent from the private schema are copied from a fresh public builder.
    """
    from newton._src.solvers.mjvbd_v2.mujoco.solver_mujoco import (
        SolverMuJoCo as PrivateMuJoCo,
    )
    from newton._src.utils.import_mjcf import parse_mjcf

    reference = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(reference)
    additions = []
    for name, attribute in reference.custom_attributes.items():
        if name not in builder.custom_attributes:
            builder.add_custom_attribute(copy.deepcopy(attribute))
            additions.append(name)
    parser = types.FunctionType(
        parse_mjcf.__code__,
        {**parse_mjcf.__globals__, "SolverMuJoCo": PrivateMuJoCo},
        parse_mjcf.__name__,
        parse_mjcf.__defaults__,
        parse_mjcf.__closure__,
    )
    parser.__kwdefaults__ = parse_mjcf.__kwdefaults__
    parser(
        builder,
        path,
        ctrl_direct=True,
        parse_visuals=False,
        parse_sites=True,
        collapse_fixed_joints=False,
    )
    empty_references = {}
    for kind in ("joint", "wrap"):
        addresses = builder.custom_attributes[f"mujoco:tendon_{kind}_adr"].values
        counts = builder.custom_attributes[f"mujoco:tendon_{kind}_num"].values
        changed = 0
        for index, count in enumerate(counts):
            if count == 0 and addresses[index] != -1:
                # A zero-length range has no referenced row. Keeping address
                # zero makes the compact view reject fixed tendons when the
                # spatial-wrap domain is empty.
                addresses[index] = -1
                changed += 1
        empty_references[kind] = changed
    return {
        "schema_additions": additions,
        "empty_tendon_reference_corrections": empty_references,
        "parser_global_state_modified": False,
    }


def select_all_articulations(backend: Any) -> None:
    """Assign every imported articulation, including free objects, to DexUni.

    Call after adding all worlds and before finalizing the DexSim backend.
    DexUni's pure_mujoco dispatch then keeps hand-object contacts and tendons
    within one private solver. A standalone MJWarp solver is never constructed.
    """
    if backend.solver_type != "dexuni":
        raise ValueError("Expected a DexUni backend")
    count = int(backend.builder.articulation_count)
    if count <= 0:
        raise ValueError("The manipulation scene contains no articulations")
    backend.cfg.solver_cfg.mujoco_articulations = tuple(range(count))


def refresh_collision_filters(solver: Any) -> dict[str, Any]:
    """Rebuild collision caches after restoring host MJCF contact masks.

    MuJoCo Warp precomputes eligible pairs and collision-kernel type counts in
    ``put_model``. Copy these derived fields while retaining the existing
    world-expanded dynamics arrays and data allocations.
    """
    model = solver.mj_model
    for body_field, geom_field in (
        ("body_contype", "geom_contype"),
        ("body_conaffinity", "geom_conaffinity"),
    ):
        values = getattr(model, body_field)
        values[:] = 0
        np.bitwise_or.at(values, model.geom_bodyid, getattr(model, geom_field))
    previous = solver.mjw_model.nxn_geom_pair_filtered.numpy()
    with wp.ScopedDevice(solver.model.device):
        fresh = solver._mujoco_warp.put_model(model)
    updated = fresh.nxn_geom_pair_filtered.numpy()
    old_pairs = set(map(tuple, previous.tolist()))
    new_pairs = set(map(tuple, updated.tolist()))
    added = sorted(new_pairs - old_pairs)
    if added:
        raise ValueError(
            "Source collision-mask restoration adds pairs; existing contact "
            f"allocations require a separate capacity check: {added[:8]}"
        )
    fields = (
        "geom_contype",
        "geom_conaffinity",
        "body_contype",
        "body_conaffinity",
        "nxn_geom_pair",
        "nxn_geom_pair_filtered",
        "nxn_pairid",
        "nxn_pairid_filtered",
        "geom_pair_type_count",
    )
    for field in fields:
        setattr(solver.mjw_model, field, getattr(fresh, field))
    solver.mjw_model.opt.broadphase = fresh.opt.broadphase
    return {
        "status": "refreshed",
        "fields": list(fields) + ["opt.broadphase"],
        "eligible_pairs_before": len(old_pairs),
        "eligible_pairs_after": len(new_pairs),
        "removed_pairs": len(old_pairs - new_pairs),
        "added_pairs": len(added),
    }


class DexUniStateAdapter:
    """Scatter private MuJoCo state into the owning DexSim state buffers.

    The compact-to-scene maps come from NewtonBackend's public facade. Physical
    stepping remains owned by ``backend.step`` and the outer ``SolverDexUni``.
    """

    def __init__(self, backend: Any) -> None:
        outer = backend.solver
        if backend.solver_type != "dexuni" or type(outer).__name__ != "SolverDexUni":
            raise TypeError("Requested DexUni did not create SolverDexUni")
        if outer.features.backend != "pure_mujoco":
            raise ValueError(
                f"Expected DexUni pure_mujoco dispatch, got {outer.features.backend}"
            )
        if outer.features.vbd_solve_enabled or not outer.features.mujoco_solve_enabled:
            raise ValueError("Unexpected active DexUni solver branches")
        self.backend = backend
        self.solver = backend.mujoco_solver
        if self.solver is None or not hasattr(self.solver, "mjw_data"):
            raise TypeError("DexUni has no active GPU MuJoCo data")
        if ".mjvbd_v2.mujoco." not in type(self.solver).__module__:
            raise TypeError("DexUni did not select its private MuJoCo implementation")
        self.local_state = self.solver.model.state()
        parent = backend.model
        joint_world = wp.to_torch(parent.joint_world).long()
        coord_world = torch.repeat_interleave(
            joint_world, torch.diff(wp.to_torch(parent.joint_q_start)).long()
        )
        dof_world = torch.repeat_interleave(
            joint_world, torch.diff(wp.to_torch(parent.joint_qd_start)).long()
        )
        body_world = wp.to_torch(parent.body_world).long()
        frequencies = (
            ("joint_q", Model.AttributeFrequency.JOINT_COORD, coord_world),
            ("joint_qd", Model.AttributeFrequency.JOINT_DOF, dof_world),
            ("body_q", Model.AttributeFrequency.BODY, body_world),
            ("body_qd", Model.AttributeFrequency.BODY, body_world),
        )
        self.fields = []
        for name, frequency, worlds in frequencies:
            mapping = backend.get_mujoco_index_map(frequency)
            if mapping is None:
                raise ValueError(f"Missing DexUni {name} compact-to-scene map")
            indices = wp.to_torch(mapping).long()
            local = wp.to_torch(getattr(self.local_state, name))
            if local.shape[0] != indices.numel():
                raise ValueError(f"DexUni {name} state and index map disagree")
            self.fields.append((name, indices, worlds[indices]))

    def publish_reset(self, data: Any, env_ids: torch.Tensor) -> None:
        """Publish selected MJCF coordinate resets into both scene buffers."""
        selected = [
            (name, indices, torch.isin(worlds, env_ids))
            for name, indices, worlds in self.fields
        ]
        for parent_state in (self.backend.state_0, self.backend.state_1):
            for name, indices, _ in selected:
                wp.to_torch(getattr(self.local_state, name)).copy_(
                    wp.to_torch(getattr(parent_state, name))[indices]
                )
            self.solver._update_newton_state(
                self.solver.model,
                self.local_state,
                data,
                state_prev=self.local_state,
            )
            for name, indices, mask in selected:
                wp.to_torch(getattr(parent_state, name))[indices[mask]] = wp.to_torch(
                    getattr(self.local_state, name)
                )[mask]
