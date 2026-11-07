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

"""Adapters for live captures and qualified observation history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import EvaluationResult, EvaluationSession, NearTargetHoldSpec
from embodichain.compute.geometry._distance import _measure_near_target

if TYPE_CHECKING:
    import torch

    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.managers.observation_types import (
        ObservationDescriptor,
        ObservationFrame,
        ObservationSample,
    )

__all__ = ["advance_observation", "evaluate_recording"]


def _rigid_pose_field(uid: str) -> str:
    return f"rigid_objects/{uid}/pose"


def _check_pose(field: str, descriptor: ObservationDescriptor) -> None:
    if (
        descriptor.quantity != "pose"
        or descriptor.coordinate_frame != "arena"
        or descriptor.representation != "xyz_xyzw"
        or descriptor.shape != (7,)
    ):
        raise ValueError(f"Evaluation field {field!r} must be a raw arena pose.")


def _near_target_samples(
    spec: NearTargetHoldSpec, frame: ObservationFrame
) -> tuple[ObservationSample, ObservationSample]:
    position_field, target_field = spec.required_fields
    position = frame.fields[position_field]
    target = frame.fields[target_field]
    _check_pose(position_field, position.descriptor)
    _check_pose(target_field, target.descriptor)
    return position, target


def _measure_near_target_frame(
    spec: NearTargetHoldSpec, frame: ObservationFrame
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position, target = _near_target_samples(spec, frame)
    satisfied, finite, distance = _measure_near_target(
        position.values[:, :3],
        target.values[:, :3],
        spec.tolerance,
        spec.axes,
        spec.inclusive,
    )
    valid = position.valid & target.valid & finite
    return satisfied & valid, valid, distance


def advance_observation(
    session: EvaluationSession,
    frame: ObservationFrame,
    *,
    step: int,
    episode_owner: EpisodeContext,
) -> None:
    """Evaluate one live observation capture in the session's row order.

    Args:
        session: Evaluation window for the expected episode cohort.
        frame: Qualified raw capture from the host's ObservationManager.
        step: Actual control slot, also used to index the recorded capture.
        episode_owner: The host's exact episode context.

    The host supplies the same owner for the session lifetime. Policy features
    and transformed or noisy observations are excluded from this path.
    """
    import torch

    current_episodes = episode_owner.episode_ids
    if (
        frame.episode_owner is not episode_owner
        or (
            session._observation_owner is not None
            and session._observation_owner is not episode_owner
        )
        or len(frame.episode_ids) != len(current_episodes)
        or any(row >= len(current_episodes) for row in session.env_ids)
        or tuple(current_episodes[row] for row in session.env_ids)
        != session.episode_ids
        or tuple(frame.episode_ids[row] for row in session.env_ids)
        != session.episode_ids
    ):
        raise ValueError("Live observation does not belong to the evaluation episodes.")
    position, target = _near_target_samples(session.spec, frame)
    rows = torch.tensor(session.env_ids, dtype=torch.long, device=session.device)
    session.advance(
        step,
        position.values[:, :3].index_select(0, rows),
        target.values[:, :3].index_select(0, rows),
        valid=(position.valid & target.valid).index_select(0, rows),
        revisions=torch.full_like(rows, frame.revision),
    )
    session._observation_owner = episode_owner


def evaluate_recording(
    spec: NearTargetHoldSpec,
    history: ObservationManager,
    *,
    env_ids: Sequence[int],
    episode_ids: Sequence[int],
    recorded_end_step: int,
    start_step: int = 0,
    chunk_steps: int = 64,
    device: str | torch.device = "cpu",
) -> EvaluationResult:
    """Replay an aligned, inclusive interval from qualified raw history.

    Args:
        spec: Rule and bounds for the replayed window.
        history: ObservationManager owning the source history and its schema.
        env_ids: Selected native rows in result order.
        episode_ids: Expected source episodes in the same order.
        recorded_end_step: Last recorded slot to consume for the selected rows.
            Use one less than the first slot for an empty recording.
        start_step: First expected control slot.
        chunk_steps: Maximum number of slots read per history query.
        device: Device for history chunks and evaluation state.

    Returns:
        The final predicate and closed-window results. An incomplete recording
        can preserve a witnessed hold but cannot prove an unwitnessed one.
    """
    import torch

    session = EvaluationSession(
        spec,
        env_ids=env_ids,
        episode_ids=episode_ids,
        start_step=start_step,
        device=device,
    )
    if (
        type(recorded_end_step) is not int
        or not session.start_step - 1 <= recorded_end_step <= session.end_step
        or type(chunk_steps) is not int
        or chunk_steps < 1
    ):
        raise ValueError("Invalid recording interval or chunk size.")
    position_field, target_field = session.spec.required_fields
    for field in session.spec.required_fields:
        _check_pose(field, history.schema[field])

    size = len(session.env_ids)
    for start in range(session.start_step, recorded_end_step + 1, chunk_steps):
        end = min(start + chunk_steps - 1, recorded_end_step)
        chunk = history.read_history_window(
            (start,) * size,
            (end,) * size,
            fields=session.spec.required_fields,
            env_ids=session.env_ids,
            episode_ids=session.episode_ids,
            device=session.device,
        )
        position = chunk["fields"][position_field]
        target = chunk["fields"][target_field]
        positions = torch.stack(position["values"], dim=1)[..., :3]
        targets = torch.stack(target["values"], dim=1)[..., :3]
        valid = torch.stack(position["valid"], dim=1) & torch.stack(
            target["valid"], dim=1
        )
        revisions = torch.stack(chunk["revisions"], dim=1)
        for offset, step in enumerate(range(start, end + 1)):
            session.advance(
                step,
                positions[offset],
                targets[offset],
                valid=valid[offset],
                revisions=revisions[offset],
            )
    return session.snapshot(closed=True)
