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

import json
import math
from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING

from . import EvaluationStatus
from ._graph import CompiledEvaluation
from ._recording import _field_projection, _foreign_layouts, _joint_layout_pairs

if TYPE_CHECKING:
    import torch

    from . import RuleSession
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_types import ObservationFrame

__all__ = ["advance_observation"]


def _raw_field_kind(field: str) -> str:
    owner, _, channel = field.partition("/")
    quantity = channel.rsplit("/", 1)[1]
    if owner == "deformables":
        from ._deformable import _centroid_selection

        _centroid_selection(field)
        return "position"
    if owner == "rigid_objects":
        return {
            "pose": "position",
            "frame": "object_frame",
            "segment_start_frame": "object_frame",
            "lin_vel": "linear_velocity",
            "ang_vel": "angular_velocity",
        }[quantity]
    if owner in ("links", "robot_links"):
        return f"link_{quantity}"
    if owner == "robot":
        return {"qpos": "joint_positions", "command": "joint_command"}[quantity]
    return "joint_position"


def _graph_frame_inputs(
    compiled: CompiledEvaluation,
    frame: ObservationFrame,
    rows: torch.Tensor | None = None,
):
    values, valid = {}, {}
    descriptors = {}
    expected_names = compiled._preset_joint_names()
    for field in compiled.required_fields:
        sample = frame.fields.get(field)
        if sample is None:
            continue
        projection = _field_projection(field, sample.descriptor, expected_names)
        if projection is None:
            continue
        value, mask = sample.values[..., projection], sample.valid
        if rows is not None:
            value = value.index_select(0, rows)
            mask = mask.index_select(0, rows)
        values[field], valid[field] = value, mask
        descriptors[field] = sample.descriptor
    for field in _foreign_layouts(descriptors):
        del values[field], valid[field]
    return values, valid


def _combine_selections(
    selections: Mapping[str, CompiledEvaluation],
) -> tuple[tuple[tuple[str, ...], CompiledEvaluation], ...]:
    """Share compiled dependencies without changing evidence qualification."""
    groups = []
    unconstrained = object()
    for ref, spec in selections.items():
        fields = set(spec.required_fields)
        expected_names = spec._preset_joint_names()
        for members, nodes, required, layouts in groups:
            if any(
                layouts.get(field, unconstrained)
                != expected_names.get(field, unconstrained)
                for field in required & fields
            ):
                continue
            # A union must not cause a channel to be rejected by a joint-order
            # check that did not apply to it in the original selection.
            if any(
                (command in fields) != (position in fields)
                or (command in required) != (position in required)
                for command, position in _joint_layout_pairs(required | fields)
            ):
                continue
            if all(nodes.get(name, node) is node for name, node in spec._nodes):
                members[ref] = spec
                nodes.update(spec._nodes)
                required.update(fields)
                layouts.update(expected_names)
                break
        else:
            # Independently compiled libraries may also reuse a local name.
            groups.append(({ref: spec}, dict(spec._nodes), fields, expected_names))
    combined = []
    for members, nodes, fields, _ in groups:
        refs = tuple(members)
        spec = members[refs[0]]
        if len(refs) > 1:
            specs = tuple(members.values())
            spec = CompiledEvaluation(
                tuple(dict.fromkeys(root for graph in specs for root in graph.roots)),
                tuple(sorted(fields)),
                sha256(
                    json.dumps([graph.identity for graph in specs]).encode()
                ).hexdigest(),
                tuple(nodes.items()),
            )
        combined.append((refs, spec))
    return tuple(combined)


def _json_metric(value):
    if isinstance(value, (list, tuple)):
        return [_json_metric(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_metric(item) for key, item in value.items()}
    return None if isinstance(value, float) and not math.isfinite(value) else value


_RULE_BOUNDARIES = (
    "witness_steps",
    "witness_revisions",
    "violation_steps",
    "violation_revisions",
)


def _rule_tensor_report(result, *, owned=False):
    def tensor(value):
        return value if owned else value.clone()

    return {
        "status": tensor(result.status),
        "reason_codes": {str(int(code)): code.reason for code in EvaluationStatus},
        "metrics": {key: tensor(value) for key, value in result.metrics.items()},
        "parameters": _json_metric(result.parameters),
        **{
            key: tensor(value)
            for key in _RULE_BOUNDARIES
            if (value := getattr(result, key)) is not None
        },
    }


def _rule_report(result):
    statuses = [EvaluationStatus(value) for value in result.status.tolist()]
    return {
        "status": [status.name.lower() for status in statuses],
        "reasons": [status.reason for status in statuses],
        "metrics": {
            key: _json_metric(metric.tolist()) for key, metric in result.metrics.items()
        },
        "parameters": _json_metric(result.parameters),
        **{
            key: value.tolist()
            for key in _RULE_BOUNDARIES
            if (value := getattr(result, key)) is not None
        },
    }


def advance_observation(
    session: RuleSession,
    frame: ObservationFrame,
    *,
    step: int,
    episode_owner: EpisodeContext,
) -> None:
    """Consume a qualified host capture in the session's environment order."""
    _advance_observation(session, frame, step=step, episode_owner=episode_owner)


def _advance_observation(session, frame, *, step, episode_owner, measurements=None):
    import torch

    rows = session.env_ids
    current_episodes = episode_owner.episode_ids
    if (
        frame.episode_owner is not episode_owner
        or (
            session._observation_owner is not None
            and session._observation_owner is not episode_owner
        )
        or len(frame.episode_ids) != len(current_episodes)
        or max(rows) >= len(current_episodes)
        or tuple(current_episodes[row] for row in rows) != session.episode_ids
        or tuple(frame.episode_ids[row] for row in rows) != session.episode_ids
    ):
        raise ValueError("Live observation does not belong to the evaluation episodes.")
    if session._observation_rows is None:
        session._observation_rows = torch.tensor(
            rows, device=session.device, dtype=torch.long
        )
    indices = session._observation_rows
    values, valid = _graph_frame_inputs(session.compiled, frame, indices)
    session._advance(
        step,
        values,
        valid=valid,
        revisions=torch.full_like(indices, frame.revision),
        episode_ids=session.episode_ids,
        measurements=measurements,
    )
    session._observation_owner = episode_owner
