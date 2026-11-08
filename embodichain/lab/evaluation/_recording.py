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

"""Stream the raw observation channels of a trajectory without a simulator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any

from ._graph import CompiledEvaluation
from ._session import EvaluationSnapshot, RuleSession

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


def evaluate_recorded_rules(
    compiled: CompiledEvaluation,
    recording: str | PathLike[str] | Mapping[str, Any],
    *,
    env_ids: Sequence[int],
    episode_ids: Sequence[int],
    scope: str,
    start_step: int = 0,
    end_step: int | None = None,
    segment_id: int | None = None,
    chunk_steps: int = 64,
    device: str | torch.device = "cpu",
) -> EvaluationSnapshot:
    """Evaluate named rules from a trusted trajectory path or its loaded payload.

    Read only required fields in bounded chunks. Original acquisition revisions
    and expected episode IDs qualify evidence; absent fields and rows stay
    unavailable. For an explicit segment occurrence, use its recorded start
    and end boundaries, including the post-action frame of its last control step.
    Selected rows must share the segment start. Rows may finish at different slots.

    Args:
        compiled: Rules bound to recorded observation field IDs.
        recording: Trusted `.pt` trajectory or equivalent loaded mapping.
        env_ids: Original environment rows in result order.
        episode_ids: Expected source episode IDs in that order.
        scope: Occurrence identity included in the result.
        start_step: First source state slot for an explicit interval.
        end_step: Inclusive last source slot, or the available recorded extent.
        segment_id: Optional recorded occurrence in per-episode segment metadata.
        chunk_steps: Upper bound on copied state slots per read.
        device: Evaluation device.
    """
    import torch

    if (
        type(chunk_steps) is not int
        or chunk_steps < 1
        or type(start_step) is not int
        or start_step < 0
        or (
            end_step is not None
            and (type(end_step) is not int or end_step < start_step - 1)
        )
        or (
            segment_id is not None
            and (
                type(segment_id) is not int
                or segment_id < 0
                or start_step != 0
                or end_step is not None
            )
        )
    ):
        raise ValueError("Invalid replay interval, segment selection or chunk size.")
    payload = (
        recording
        if isinstance(recording, Mapping)
        else torch.load(recording, map_location="cpu", mmap=True, weights_only=False)
    )
    ids, expected = tuple(env_ids), tuple(episode_ids)
    history = payload.get("raw_observations")
    segment_ends = None
    if segment_id is not None:
        bounds = _segment_bounds(payload, ids, expected, segment_id)
        if bounds is not None:
            start_step, segment_ends = bounds
            end_step = max(segment_ends)

    session = RuleSession(
        compiled,
        env_ids=ids,
        episode_ids=expected,
        scope=scope,
        start_step=start_step,
        device=device,
    )
    if history is None or (segment_id is not None and segment_ends is None):
        session.advance(
            start_step,
            {},
            revisions=torch.full((len(ids),), -1, device=session.device),
            episode_ids=expected,
        )
        return session.snapshot(closed=True)

    if not isinstance(history, Mapping) or history.get("schema_version") != 1:
        raise ValueError("Unsupported raw observation recording schema.")
    source_ids, source_episodes = tuple(history["env_ids"]), tuple(
        history["episode_ids"]
    )
    revisions, lengths = history["revisions"], history["lengths"]
    size = len(source_ids)
    if (
        not size
        or len(set(source_ids)) != size
        or len(source_episodes) != size
        or revisions.ndim != 2
        or revisions.shape[0] != size
        or revisions.dtype != torch.long
        or lengths.shape != (size,)
        or lengths.dtype != torch.long
        or bool(((lengths < 0) | (lengths >= revisions.shape[1])).any())
    ):
        raise ValueError("Invalid recorded row identities or history lengths.")
    row_map = {env_id: row for row, env_id in enumerate(source_ids)}
    rows = [row_map.get(env_id, -1) for env_id in ids]
    source_episode_ids = tuple(source_episodes[row] if row >= 0 else -1 for row in rows)
    index = torch.tensor([max(row, 0) for row in rows], device=revisions.device)
    row_lengths = lengths.index_select(0, index.to(lengths.device))
    row_lengths.masked_fill_(
        torch.tensor([row < 0 for row in rows], device=row_lengths.device), -1
    )
    if segment_ends is not None:
        row_lengths = torch.minimum(
            row_lengths, torch.tensor(segment_ends, device=row_lengths.device)
        )
    stop = int(row_lengths.max()) if end_step is None else end_step
    if stop < start_step or start_step >= revisions.shape[1]:
        session.advance(
            start_step,
            {},
            revisions=torch.full((len(ids),), -1, device=session.device),
            episode_ids=expected,
        )
        return session.snapshot(closed=True)
    row_lengths = row_lengths.to(session.device)

    fields = {}
    for field in compiled.required_fields:
        sample = history["fields"].get(field)
        if sample is None:
            continue
        values, valid = sample["values"], sample["valid"]
        descriptor = sample["descriptor"]
        if not _matches_field(
            field, descriptor.get("entity_uid"), descriptor["quantity"]
        ):
            continue
        if _is_joint_field(field, descriptor["quantity"]) and not descriptor.get(
            "joint_names"
        ):
            continue
        if (
            values.shape[:2] != revisions.shape
            or valid.shape != revisions.shape
            or valid.dtype != torch.bool
            or tuple(descriptor["shape"]) != tuple(values.shape[2:])
            or descriptor["dtype"] != str(values.dtype)
        ):
            raise ValueError(f"Invalid recorded observation field {field!r}.")
        fields[field] = (values, valid, descriptor)

    for start in range(start_step, min(stop + 1, revisions.shape[1]), chunk_steps):
        end = min(start + chunk_steps, stop + 1, revisions.shape[1])
        chunk = {
            field: (
                _project_field(
                    values[:, start:end]
                    .index_select(0, index.to(values.device))
                    .to(session.device),
                    descriptor,
                    field,
                ),
                valid[:, start:end]
                .index_select(0, index.to(valid.device))
                .to(session.device),
            )
            for field, (values, valid, descriptor) in fields.items()
        }
        revision_chunk = (
            revisions[:, start:end].index_select(0, index).to(session.device)
        )
        for offset, step in enumerate(range(start, end)):
            session.advance(
                step,
                {field: values[:, offset] for field, (values, _) in chunk.items()},
                valid={field: valid[:, offset] for field, (_, valid) in chunk.items()},
                revisions=revision_chunk[:, offset].masked_fill(step > row_lengths, -1),
                episode_ids=source_episode_ids,
            )
    return session.snapshot(closed=True)


def _matches_field(field, uid, quantity):
    if field.startswith("rigid_objects/"):
        return field == f"rigid_objects/{uid}/{quantity}"
    if field.startswith("articulations/"):
        source = f"articulations/{uid}/{quantity}"
        return field == source or (
            quantity == "qpos" and field.startswith(f"{source}/")
        )
    return True


def _project_field(values, descriptor, field):
    quantity = descriptor["quantity"]
    if quantity == "pose":
        if (
            descriptor["coordinate_frame"] != "arena"
            or descriptor["representation"] != "xyz_xyzw"
            or values.shape[-1] != 7
        ):
            raise ValueError(
                f"Evaluation field {field!r} must contain raw arena poses."
            )
        return values[..., :3]
    if _is_joint_field(field, quantity):
        return _select_joint(values, field, descriptor["joint_names"])
    return values


def _is_joint_field(field, quantity):
    return (
        quantity == "qpos" and field.startswith("articulations/") and "/qpos/" in field
    )


def _select_joint(values, field, joint_names):
    joint = field.rsplit("/", 1)[1]
    if joint not in joint_names or values.shape[-1] != len(joint_names):
        raise ValueError(f"Evaluation field {field!r} has no matching native joint.")
    return values[..., joint_names.index(joint)]


def _segment_bounds(payload, env_ids, episode_ids, segment_id):
    episodes = {
        item["env_id"]: item
        for item in payload.get("meta", {}).get("demo_episodes", ())
    }
    boundaries = []
    for env_id, episode_id in zip(env_ids, episode_ids, strict=True):
        episode = episodes.get(env_id, {})
        if episode.get("episode_context", {}).get("episode") != episode_id:
            boundaries.append(None)
            continue
        matches = [
            segment
            for segment in episode.get("segments", ())
            if segment["segment_id"] == segment_id
        ]
        if len(matches) != 1:
            boundaries.append(None)
            continue
        segment = matches[0]
        start, end = segment["start_step"], segment["end_step"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("Invalid recorded segment boundaries.")
        boundaries.append((start + 1, end))
    starts = {bound[0] for bound in boundaries if bound is not None}
    if not starts:
        return None
    if len(starts) != 1:
        raise ValueError("Selected segment occurrences must share one start slot.")
    return starts.pop(), tuple(
        -1 if bound is None else bound[1] for bound in boundaries
    )
