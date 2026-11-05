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

"""Bounded source-order candidate IK diagnostics; no physical evidence."""

from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterator
import numpy as np
import torch

__all__: list[str] = []


def tensor_record(value: Any) -> dict[str, Any]:
    # Raw input values are owned read-only; avoid a writable Torch alias to them.
    tensor = torch.as_tensor(
        np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    )
    if not hasattr(value, "dtype") and tensor.is_floating_point():
        tensor = torch.as_tensor(value, dtype=torch.float64)
    tensor = tensor.detach().cpu()
    record = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
        return {**record, "available": False, "reason": "nonfinite_or_complex"}
    return {**record, "available": True, "values": tensor.tolist()}


@contextmanager
def candidate_group(state: Any, object_pose: torch.Tensor) -> Iterator[None]:
    if state is None or state.pour_inputs is None:
        yield
        return
    if not state.ik_audit:
        state.ik_audit = {
            "scope": "planning_inputs_not_physical",
            "group_limit": 8,
            "trace_complete": True,
            "truncated": False,
            "omitted_groups": 0,
            "omitted_calls": 0,
            "groups": [],
        }
    groups = state.ik_audit["groups"]
    previous = state.active_ik_audit
    if len(groups) < 8:
        group = {
            "calls": [],
            "trace_complete": True,
            "object_pose": tensor_record(object_pose),
        }
        groups.append(group)
    else:
        group = {"calls": [], "omitted": True}
        state.ik_audit["omitted_groups"] += 1
        state.ik_audit.update(trace_complete=False, truncated=True)
    state.active_ik_audit = group
    try:
        yield
    except Exception:
        group["trace_complete"] = False
        state.ik_audit["trace_complete"] = False
        raise
    finally:
        state.active_ik_audit = previous


def before_ik(state, poses, joint_seed, manipulator):
    group = None if state is None else state.active_ik_audit
    if group is not None and not group.get("omitted") and len(group["calls"]) < 8:
        seed = (
            joint_seed[:, None, None, :].expand(*poses.shape[:3], -1)
            if joint_seed.ndim == 2
            else joint_seed
        )
        devices = (
            []
            if poses.device.type != "cuda"
            else [
                (
                    poses.device.index
                    if poses.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        )
        record = {
            "role": (
                group["next_role"]
                if "next_role" in group
                else ("pre_grasp", "grasp", "lift")[len(group["calls"])]
            ),
            "candidate_shape": list(poses.shape[:3]),
            "control_part": manipulator.control_part,
            "joint_ids": list(manipulator.joint_ids),
            "targets": tensor_record(poses.reshape(poses.shape[0], -1, 4, 4)),
            "joint_seed": tensor_record(
                seed.reshape(poses.shape[0], -1, len(manipulator.joint_ids))
            ),
            "torch_cpu_rng_state": torch.get_rng_state().tolist(),
            "torch_cuda_rng_states": {
                str(i): torch.cuda.get_rng_state(i).tolist() for i in devices
            },
            "torch_intraop_threads": torch.get_num_threads(),
        }
        group["calls"].append(record)
        return record
    if group is not None:
        group["dropped_calls"] = group.get("dropped_calls", 0) + 1
        group["trace_complete"] = False
        state.ik_audit.update(trace_complete=False, truncated=True)
        state.ik_audit["omitted_calls"] += 1
    return None


def projected_targets(state, projection, transforms):
    targets, valid = projection.targets(transforms)
    group = None if state is None else state.active_ik_audit
    if group is None:
        yield from targets
        return
    if not group.get("omitted"):
        group["object_to_eef_variants"] = tensor_record(transforms)
        group["geometry_valid_mask"] = tensor_record(valid)
        if "operands" not in state.ik_audit:
            state.ik_audit["operands"] = {
                name: tensor_record(getattr(state.pour_inputs, name))
                for name in (
                    "source_vertices",
                    "receiver_vertices",
                    "staging",
                    "receiver_pose",
                    "pad_line_tcp",
                    "gravity_world",
                )
            }
            state.ik_audit["recipe"] = {
                name: (
                    value
                    if value is None or isinstance(value, str)
                    else tensor_record(value)
                )
                for name, value in state.pour_inputs.recipe().items()
            }
    for role, target in zip(
        ("staging", "peak", "restore", "return_transport", "return_place"),
        targets,
        strict=True,
    ):
        group["next_role"] = role
        yield target


def finish_audit(plan, state):
    if state is not None and state.ik_audit and not plan.plan_success.any():
        return replace(
            plan,
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **plan.diagnostics.metadata,
                    "pour_pick_ik_audit": deepcopy(state.ik_audit),
                },
            ),
        )
    return plan
