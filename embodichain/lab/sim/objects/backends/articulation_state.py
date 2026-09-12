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
"""Backend-specific articulation state-layout adapters."""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "get_state_joint_names",
    "map_source_qpos_to_state_order",
    "read_state_mimic_info",
]


def get_state_joint_names(entity: object) -> list[str]:
    """Return active joint names in the backing state-buffer order."""
    try:
        layout = entity.joint_dof_layout
    except (AttributeError, RuntimeError):
        return entity.get_actived_joint_names()
    return [joint.name for joint in layout]


def map_source_qpos_to_state_order(
    entity: object,
    qpos: torch.Tensor,
    *,
    is_spawn_bound: bool,
) -> torch.Tensor:
    """Map source-ordered initial qpos values into runtime state order."""
    if not is_spawn_bound:
        return qpos

    source_joint_names = entity.get_actived_joint_names()
    state_joint_names = get_state_joint_names(entity)
    if source_joint_names == state_joint_names:
        return qpos

    source_indices = {name: index for index, name in enumerate(source_joint_names)}
    try:
        state_order = [source_indices[name] for name in state_joint_names]
    except KeyError as error:
        raise RuntimeError(
            "Spawn articulation state layout contains a joint absent from "
            "the source articulation layout."
        ) from error
    return qpos[..., state_order]


def read_state_mimic_info(
    entity: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read mimic metadata and map source joint ids into state-buffer ids."""
    source_info = entity.get_mimic_info()
    source_mimic_ids = np.asarray(source_info.mimic_id, dtype=np.int32).reshape(-1)
    source_parent_ids = np.asarray(
        source_info.mimic_parent,
        dtype=np.int32,
    ).reshape(-1)
    multipliers = np.asarray(
        source_info.mimic_multiplier,
        dtype=np.float32,
    ).reshape(-1)
    offsets = np.asarray(source_info.mimic_offset, dtype=np.float32).reshape(-1)
    relation_count = len(source_mimic_ids)
    if not all(
        len(values) == relation_count
        for values in (source_parent_ids, multipliers, offsets)
    ):
        raise RuntimeError("Articulation mimic metadata has inconsistent lengths.")
    if relation_count == 0:
        return source_mimic_ids, source_parent_ids, multipliers, offsets

    source_joint_names = entity.get_actived_joint_names()
    try:
        state_joint_ids = {
            joint.name: int(joint.dof_start) for joint in entity.joint_dof_layout
        }
    except (AttributeError, RuntimeError):
        state_joint_ids = {name: index for index, name in enumerate(source_joint_names)}

    try:
        mimic_ids = np.asarray(
            [
                state_joint_ids[source_joint_names[int(source_id)]]
                for source_id in source_mimic_ids
            ],
            dtype=np.int32,
        )
        parent_ids = np.asarray(
            [
                state_joint_ids[source_joint_names[int(source_id)]]
                for source_id in source_parent_ids
            ],
            dtype=np.int32,
        )
    except (IndexError, KeyError) as error:
        raise RuntimeError(
            "Articulation mimic metadata references a joint absent from "
            "the backing state layout."
        ) from error

    return mimic_ids, parent_ids, multipliers, offsets
