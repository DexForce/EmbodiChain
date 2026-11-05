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

"""Select original raw pose windows without another schema or acquisition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from embodichain.compute.geometry.rigid import validate_rigid_pose_batch
from embodichain.utils.math import xyz_quat_to_4x4_matrix

if TYPE_CHECKING:
    import torch
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager

__all__: list[str] = []


def _recorded_pose_window(
    observations: ObservationManager | None,
    uid: str,
    start_frames: Sequence[int],
    end_frames: Sequence[int],
    *,
    episode_ids: Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    if observations is None:
        raise RuntimeError("Observation history is unavailable.")
    fields = [
        name
        for name, descriptor in observations.schema.items()
        if descriptor.entity_uid == uid
        and descriptor.quantity == "pose"
        and descriptor.coordinate_frame == "arena"
        and descriptor.representation == "xyz_xyzw"
        and descriptor.shape == (7,)
        and descriptor.units == ("m", "unitless")
    ]
    if not fields:
        raise RuntimeError(f"Recorded raw arena pose for {uid!r} is unavailable.")
    field = fields[0]
    window = observations.read_history_window(
        start_frames, end_frames, fields=(field,), episode_ids=episode_ids
    )
    channel = window["fields"][field]
    if not all(bool(valid.all()) for valid in channel["valid"]):
        raise RuntimeError(
            f"Recorded raw pose acquisition for {uid!r} is unavailable: "
            f"{channel['acquisition_errors']}."
        )
    if not all(bool(valid.all()) for valid in window["timestamp_valid"]):
        raise RuntimeError("Recorded raw pose acquisition timestamp is unavailable.")
    rows = tuple(xyz_quat_to_4x4_matrix(values) for values in channel["values"])
    for start, end, poses in zip(
        window["start_frames"], window["end_frames"], rows, strict=True
    ):
        validate_rigid_pose_batch(poses, end - start + 1)
    metadata = {
        "field_id": observations.field_ids[field],
        "env_ids": list(window["env_ids"]),
        "episode_ids": list(window["episode_ids"]),
        "start_frames": list(window["start_frames"]),
        "end_frames": list(window["end_frames"]),
        "observation_revisions": [row.tolist() for row in window["revisions"]],
        "timestamps": [row.tolist() for row in window["timestamps"]],
        "clock_domain": window["clock_domain"],
        "valid": [row.tolist() for row in channel["valid"]],
        "acquisition_errors": [list(row) for row in channel["acquisition_errors"]],
    }
    return rows, metadata
