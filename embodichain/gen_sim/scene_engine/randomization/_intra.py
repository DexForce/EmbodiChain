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

"""GenSim hierarchy adaptation, canonical poses and lazy v3 candidates."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from embodichain.compute.geometry._hierarchical import _sample_hierarchy
from embodichain.lab.sim.scene.expansion.contracts import _variant_id
from ._batch import _CandidatePool, _LoadedVariants
from ._intra_geometry import _prepare_geometry, _load_geometry
from .baseline import PreparedSceneBaseline
from .cfg import SceneRandomizationCfg, _INTRA_GENERATOR, _intra_payload

from embodichain.lab.sim.scene.expansion._layout_numeric import (
    _hierarchy_batch_limit,
    _numeric_geometry,
    _sample_poses,
    _specification,
    _stream,
    _target_indices,
    _with_table,
)

__all__: list[str] = []


def _metadata(
    data: dict, geometry: dict, ordinal: int, *, algorithm: str | None = None
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "generator": _INTRA_GENERATOR if algorithm is None else algorithm,
                "baseline_sha256": data["data_sha256"],
                "support_cache_sha256": geometry["data_sha256"],
                "proposal_ordinal": str(ordinal),
                "geometry_scope": "hierarchical_support_grid_xy_yaw",
            }.items()
        )
    )


class _IntraPool(_CandidatePool):
    def __init__(
        self,
        baseline: PreparedSceneBaseline,
        geometry_path: Path,
        geometry: dict,
        deltas: np.ndarray,
        selected: np.ndarray,
        ordinals: np.ndarray,
        parameters: dict,
        entities: tuple,
    ) -> None:
        self._data = deepcopy(baseline.data)
        self._geometry = deepcopy(geometry)
        self._geometry_path = geometry_path
        self._parameters = deepcopy(parameters)
        self._deltas, self._selected, self._ordinals = (
            np.array(x, copy=True) for x in (deltas, selected, ordinals)
        )
        self._entities = deepcopy(entities)
        self._uids, self._poses, self._centers = _numeric_geometry(
            geometry,
            _with_table(parameters, baseline.data),
            self._deltas,
            self._selected,
        )
        # Parent-before-child cache order is sorted by roots, as are layout centers.
        self._data["groups"] = sorted(self._data["groups"], key=lambda g: g["root_id"])
        for array in (
            self._deltas,
            self._selected,
            self._ordinals,
            self._poses,
            self._centers,
        ):
            array.setflags(write=False)

    def _metadata_at(self, index: int) -> tuple[tuple[str, str], ...]:
        return _metadata(self._data, self._geometry, int(self._ordinals[index]))

    def _records(
        self, baseline: PreparedSceneBaseline, parameters: dict, algorithm: str
    ) -> list[dict]:
        if (
            parameters != self._parameters
            or algorithm != _INTRA_GENERATOR
            or sorted(baseline.data["groups"], key=lambda g: g["root_id"])
            != self._data["groups"]
            or baseline.data["data_sha256"] != self._data["data_sha256"]
        ):
            raise ValueError(
                "Hierarchical candidate provenance differs from its snapshot."
            )
        from .cfg import _decode_intra

        _load_geometry(
            self._geometry_path,
            baseline.data,
            _decode_intra(parameters["intra_group"]),
            expected_sha256=self._geometry["data_sha256"],
        )
        _numeric_geometry(
            self._geometry,
            _with_table(parameters, baseline.data),
            self._deltas,
            self._selected,
        )
        records = []
        roots = [n["uid"] for n in self._geometry["nodes"] if n["parent"] < 0]
        targets = [
            self._geometry["nodes"][i]["uid"]
            for i in _target_indices(self._geometry, parameters)
        ]
        for index, ordinal in enumerate(self._ordinals):
            ordinal = int(ordinal)
            metadata = self._metadata_at(index)
            identity = _variant_id(
                self._data["source_scene"]["content_sha256"],
                parameters["seed"],
                index,
                tuple(zip(self._uids, self._poses[index].tolist(), strict=True)),
                metadata,
            )
            records.append(
                {
                    "candidate_id": f"candidate_{ordinal:06d}",
                    "proposal_ordinal": ordinal,
                    "variant_id": identity,
                    "variant_ordinal": index,
                    "variant_metadata": dict(metadata),
                    "root_xy_deltas": dict(
                        zip(roots, self._deltas[index].tolist(), strict=True)
                    ),
                    "member_poses_in_parent": dict(
                        zip(targets, self._selected[index].tolist(), strict=True)
                    ),
                }
            )
        return records


class _LoadedIntraVariants(_LoadedVariants):
    def __init__(
        self, baseline: dict, geometry: dict, records: list[dict], policy: dict
    ) -> None:
        roots = [n["uid"] for n in geometry["nodes"] if n["parent"] < 0]
        targets = [
            geometry["nodes"][i]["uid"] for i in _target_indices(geometry, policy)
        ]
        deltas, selected = [], []
        for record in records:
            if set(record["root_xy_deltas"]) != set(roots) or set(
                record["member_poses_in_parent"]
            ) != set(targets):
                raise ValueError(
                    "Saved roots/selected members do not match the support hierarchy."
                )
            delta = [record["root_xy_deltas"][uid] for uid in roots]
            poses = [record["member_poses_in_parent"][uid] for uid in targets]
            if any(
                not isinstance(row, list)
                or len(row) != 2
                or any(type(v) not in (int, float) for v in row)
                for row in delta
            ):
                raise ValueError("Root XY translations must be numeric pairs.")
            if any(
                not isinstance(pose, list)
                or len(pose) != 4
                or any(
                    not isinstance(row, list)
                    or len(row) != 4
                    or any(type(v) not in (int, float) for v in row)
                    for row in pose
                )
                for pose in poses
            ):
                raise ValueError("Member poses must contain numeric 4x4 matrices.")
            deltas.append(delta)
            selected.append(poses)
        self._source, self._seed = (
            baseline["source_scene"]["content_sha256"],
            policy["seed"],
        )
        self._uids, self._poses, _ = _numeric_geometry(
            geometry,
            _with_table(policy, baseline),
            np.asarray(deltas, dtype=np.float64),
            np.asarray(selected, dtype=np.float64),
        )
        self._poses.setflags(write=False)
        self._metadata = []
        for index, record in enumerate(records):
            metadata = _metadata(
                baseline,
                geometry,
                record["proposal_ordinal"],
                algorithm=record["algorithm_version"],
            )
            if (
                dict(metadata) != record["variant_metadata"]
                or _variant_id(
                    self._source,
                    self._seed,
                    index,
                    tuple(zip(self._uids, self._poses[index].tolist(), strict=True)),
                    metadata,
                )
                != record["variant_id"]
            ):
                raise ValueError(
                    "Reconstructed hierarchical variant identity or provenance does not match."
                )
            self._metadata.append(metadata)


def _randomize_intra(
    cfg: SceneRandomizationCfg, baseline: PreparedSceneBaseline, parameters: dict
):
    from .generation import SceneRandomizationResult, _runtime_entities

    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (
            device.index is not None and device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                "CUDA hierarchical generation requested but device is unavailable."
            )
        device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
    path, geometry, _ = _prepare_geometry(baseline, cfg.intra_group)
    batch_limit = (
        _hierarchy_batch_limit(
            cfg.batch_size, sum(n["parent"] < 0 for n in geometry["nodes"])
        )
        if cfg.randomize_groups
        else cfg.batch_size
    )
    resolved = {
        **parameters,
        "backend": "batched",
        "device": str(device),
        "batch_size": batch_limit,
        "randomize_groups": cfg.randomize_groups,
        "intra_group": _intra_payload(cfg.intra_group),
    }
    spec = _specification(geometry, resolved, device)
    table = torch.tensor(
        _with_table(resolved, baseline.data)["_table"],
        dtype=torch.float64,
        device=device,
    )
    deltas, samples, ordinals = [], [], []
    accepted = attempts = 0
    while (
        attempts < parameters["max_attempts"]
        and accepted < parameters["candidate_count"]
    ):
        count = min(batch_limit, parameters["max_attempts"] - attempts)
        delta, sample, valid = _sample_hierarchy(
            *spec,
            table,
            start=attempts,
            count=count,
            randomize_groups=cfg.randomize_groups,
            sibling_margin=cfg.intra_group.sibling_margin_m,
            group_margin=parameters["collision_margin_m"],
            support_gap=cfg.intra_group.support_gap_m,
        )
        chosen = torch.nonzero(valid, as_tuple=False).flatten()[
            : parameters["candidate_count"] - accepted
        ]
        picked = chosen.cpu().numpy() + attempts
        deltas.append(delta[chosen].cpu().numpy())
        samples.append(sample[chosen].cpu().numpy())
        ordinals.append(picked)
        accepted += len(picked)
        attempts = (
            int(picked[-1]) + 1
            if accepted == parameters["candidate_count"]
            else attempts + count
        )
    sample_array = np.concatenate(samples)
    selected = _sample_poses(geometry, resolved, sample_array)
    from .persistence import _verify_baseline
    from .baseline import _read_json, _rigid_entries

    export = _verify_baseline(baseline)
    _, config = _read_json(export / "scene_config.json")
    entities = _runtime_entities(_rigid_entries(config), export)
    pool = _IntraPool(
        baseline,
        path,
        geometry,
        np.concatenate(deltas),
        selected,
        np.concatenate(ordinals),
        resolved,
        entities,
    )
    return SceneRandomizationResult(
        baseline,
        pool,
        attempts,
        (
            "completed"
            if accepted == parameters["candidate_count"]
            else "budget_exhausted"
        ),
        {"solved": accepted, "no_solution": attempts - accepted},
        resolved,
        _INTRA_GENERATOR,
    )
