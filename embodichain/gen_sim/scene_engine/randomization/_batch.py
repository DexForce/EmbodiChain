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

"""Numeric grouped candidates with deferred runtime-configuration construction."""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.compute.geometry._sampling import _sample_aabb_layouts
from embodichain.lab.sim.scene.composition import SceneFragment
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.sim.scene.expansion.contracts import _variant_id
from embodichain.lab.sim.scene.layout import LayoutSolution
from embodichain.lab.sim.scene.procedural import GeneratedScene
from embodichain.lab.sim.scene.validation import SceneCheck, SceneCheckReport

from .baseline import PreparedSceneBaseline, _read_pose
from .cfg import _BATCH_GENERATOR, SceneRandomizationCfg

if TYPE_CHECKING:
    from .generation import SceneRandomizationResult

__all__: list[str] = []


def _check_centers(data: dict, centers: np.ndarray, margin: float) -> None:
    """Validate bounded batches without constructing individual pose contracts."""
    groups = data["groups"]
    half = np.asarray([g["aabb_2d"]["size_xy"] for g in groups]) / 2
    table_center = np.asarray(data["table"]["center_xy"])
    table_half = np.asarray(data["table"]["size_xy"]) / 2
    if centers.shape[1:] != (len(groups), 2) or not np.isfinite(centers).all():
        raise ValueError("Group centers must contain finite XY coordinates.")
    first, second = np.triu_indices(len(groups), k=1)
    for start in range(0, len(centers), 4096):
        xy = centers[start : start + 4096]
        if np.any(np.abs(xy - table_center) + half > table_half + 1e-7):
            raise ValueError("Candidate group lies outside the table bounds.")
        gap = np.abs(xy[:, first] - xy[:, second]) - half[first] - half[second]
        if np.any(gap.max(axis=-1) < margin - 1e-7):
            raise ValueError("Candidate group clearance is below the required margin.")


def _group_poses(data: dict, centers: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    """Expand original root/member matrices using canonical CPU float64 arithmetic.

    Saving and loading use this same operation order for bit-stable identities;
    GPU matmul rounding must not change a persisted proposal's identity.
    """
    poses = {}
    for index, group in enumerate(data["groups"]):
        root = np.broadcast_to(
            _read_pose(group["root_pose"]), (len(centers), 4, 4)
        ).copy()
        root[:, :2, 3] += centers[:, index] - group["aabb_2d"]["center_xy"]
        poses[group["root_id"]] = root
        for member in group["members"]:
            poses[member["entity_id"]] = root @ _read_pose(member["pose_in_root"])
    uids = tuple(sorted(poses))
    return uids, np.stack([poses[uid] for uid in uids], axis=1)


def _metadata(data: dict, proposal_ordinal: int) -> tuple[tuple[str, str], ...]:
    """Return canonical provenance independent of batch partitioning."""
    return tuple(
        sorted(
            (
                ("generator", _BATCH_GENERATOR),
                ("baseline_sha256", data["data_sha256"]),
                ("proposal_ordinal", str(proposal_ordinal)),
                ("geometry_scope", "fixed_orientation_group_aabb_xy"),
            )
        )
    )


class _CandidatePool(Sequence[GeneratedScene]):
    """Own compact numeric candidates; materialize copied objects only on access."""

    def __init__(
        self,
        baseline: PreparedSceneBaseline,
        centers: np.ndarray,
        ordinals: np.ndarray,
        parameters: dict,
        entities: tuple,
    ) -> None:
        self._data = deepcopy(baseline.data)
        self._parameters = dict(parameters)
        self._centers = np.array(centers, dtype=np.float64, copy=True)
        self._ordinals = np.array(ordinals, dtype=np.int64, copy=True)
        self._entities = deepcopy(entities)
        self._uids, self._poses = _group_poses(self._data, self._centers)
        for array in (self._centers, self._ordinals, self._poses):
            array.setflags(write=False)

    def _metadata_at(self, index: int) -> tuple[tuple[str, str], ...]:
        return _metadata(self._data, int(self._ordinals[index]))

    def __len__(self) -> int:
        return len(self._centers)

    def __getitem__(
        self, index: int | slice
    ) -> GeneratedScene | tuple[GeneratedScene, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(len(self))[index])
        index = range(len(self))[index]
        changes = tuple(
            ScenePoseChange(uid, pose.tolist())
            for uid, pose in zip(self._uids, self._poses[index], strict=True)
        )
        variant = SceneVariant(
            self._data["source_scene"]["content_sha256"],
            self._parameters["seed"],
            index,
            changes,
            self._metadata_at(index),
        )
        # Replace pose fields on shallow shells; SceneFragment performs the
        # single deep ownership copy before returning any configuration.
        entities = [copy(entity) for entity in self._entities]
        poses = dict(zip(self._uids, self._poses[index], strict=True))
        for entity in entities:
            if entity.uid in poses:
                entity.init_local_pose = poses[entity.uid].copy()
                entity.init_pos = tuple(entity.init_local_pose[:3, 3])
        layout = LayoutSolution(
            "solved",
            {
                g["root_id"]: tuple(map(float, xy))
                for g, xy in zip(
                    self._data["groups"], self._centers[index], strict=True
                )
            },
            {"linear": 0.0, "fixed": 0.0, "collision": 0.0},
            "Accepted bounded batch proposal.",
        )
        return GeneratedScene(
            SceneFragment(entities),
            variant,
            layout,
            SceneCheckReport(
                (
                    SceneCheck(
                        "scene.layout", "passed", layout.message, layout.residuals
                    ),
                )
            ),
        )

    def _records(
        self, baseline: PreparedSceneBaseline, parameters: dict, algorithm: str
    ) -> list[dict]:
        """Serialize owned numeric evidence without materializing configurations."""
        if (
            parameters != self._parameters
            or algorithm != _BATCH_GENERATOR
            or baseline.data != self._data
        ):
            raise ValueError("Batch provenance differs from the generated snapshot.")
        _check_centers(self._data, self._centers, parameters["collision_margin_m"])
        records = []
        for index, (centers, poses, ordinal) in enumerate(
            zip(self._centers, self._poses, self._ordinals, strict=True)
        ):
            ordinal = int(ordinal)
            metadata = _metadata(self._data, ordinal)
            identity = _variant_id(
                self._data["source_scene"]["content_sha256"],
                parameters["seed"],
                index,
                tuple(zip(self._uids, poses.tolist(), strict=True)),
                metadata,
            )
            records.append(
                {
                    "candidate_id": f"candidate_{ordinal:06d}",
                    "proposal_ordinal": ordinal,
                    "variant_id": identity,
                    "variant_ordinal": index,
                    "variant_metadata": dict(metadata),
                    "groups": [
                        {
                            "root_id": g["root_id"],
                            "center_xy": xy.tolist(),
                            "yaw_delta_deg": 0.0,
                        }
                        for g, xy in zip(self._data["groups"], centers, strict=True)
                    ],
                }
            )
        return records


class _LoadedVariants(Sequence[SceneVariant]):
    """Own validated numeric poses and construct typed variants when consumed."""

    def __init__(
        self, data: dict, centers: np.ndarray, records: list[dict], seed: int
    ) -> None:
        self._source = data["source_scene"]["content_sha256"]
        self._seed = seed
        self._uids, self._poses = _group_poses(data, centers)
        self._poses.setflags(write=False)
        self._metadata = []
        for index, record in enumerate(records):
            # Validate metadata without allocating per-entity pose contracts.
            metadata = SceneVariant(
                self._source, seed, index, (), tuple(record["variant_metadata"].items())
            ).metadata
            identity = _variant_id(
                self._source,
                seed,
                index,
                tuple(zip(self._uids, self._poses[index].tolist(), strict=True)),
                metadata,
            )
            if identity != record["variant_id"]:
                raise ValueError(
                    "Reconstructed variant identity does not match the saved layout."
                )
            self._metadata.append(metadata)

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(
        self, index: int | slice
    ) -> SceneVariant | tuple[SceneVariant, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(len(self))[index])
        index = range(len(self))[index]
        return SceneVariant(
            self._source,
            self._seed,
            index,
            tuple(
                ScenePoseChange(uid, pose.tolist())
                for uid, pose in zip(self._uids, self._poses[index], strict=True)
            ),
            self._metadata[index],
        )


def _randomize_batched(
    cfg: SceneRandomizationCfg,
    baseline: PreparedSceneBaseline,
    parameters: dict,
) -> SceneRandomizationResult:
    """Consume deterministic proposal prefixes within the exact logical budget."""
    from .generation import SceneRandomizationResult, _build_recipe

    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA batch generation requested but CUDA is unavailable.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA batch device {device} is unavailable.")
    recipe = _build_recipe(baseline)
    data = baseline.data
    half = (
        torch.tensor(
            [g["aabb_2d"]["size_xy"] for g in data["groups"]],
            dtype=torch.float64,
            device=device,
        )
        / 2
    )
    center = np.asarray(data["table"]["center_xy"])
    table_half = np.asarray(data["table"]["size_xy"]) / 2
    region = torch.tensor(
        np.stack((center - table_half, center + table_half)),
        dtype=torch.float64,
        device=device,
    )
    centers, ordinals = [], []
    accepted, attempts = 0, 0
    while (
        attempts < parameters["max_attempts"]
        and accepted < parameters["candidate_count"]
    ):
        size = min(cfg.batch_size, parameters["max_attempts"] - attempts)
        xy, valid = _sample_aabb_layouts(
            half,
            region,
            seed=parameters["seed"],
            start=attempts,
            count=size,
            margin=parameters["collision_margin_m"],
        )
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        needed = parameters["candidate_count"] - accepted
        selected = indices[:needed]
        selected_ordinals = selected.cpu().numpy() + attempts
        centers.append(xy[selected].cpu().numpy())
        ordinals.append(selected_ordinals)
        accepted += len(selected_ordinals)
        attempts = (
            int(selected_ordinals[-1]) + 1
            if accepted == parameters["candidate_count"]
            else attempts + size
        )
    resolved = {
        **parameters,
        "backend": "batched",
        "device": str(device),
        "batch_size": cfg.batch_size,
    }
    pool = _CandidatePool(
        baseline,
        np.concatenate(centers),
        np.concatenate(ordinals),
        resolved,
        recipe.fragment.entities,
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
        _BATCH_GENERATOR,
    )
