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

"""TaskSpec evidence collection inside the existing runner, before reset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from embodichain.gen_sim.task_spec import TaskSpec, evaluate_task_spec
from embodichain.utils.utility import load_config
from ..contracts import canonical_hash
from .stability import StabilityConstraint

__all__: list[str] = []


def _needs_universe(spec: TaskSpec, predicate: Any) -> bool:
    return any(
        r.quantifier == "all"
        and r.name
        in (predicate.arguments.get("subject"), predicate.arguments.get("reference"))
        for r in spec.roles
    )


def _pose(simulation: Any, binding: Any) -> torch.Tensor:
    entity = simulation.get_rigid_object(binding.simulation_uid)
    if entity is None:
        raise ValueError("Required native rigid object is not present.")
    pose = entity.get_local_pose(to_matrix=True).detach().cpu()
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4) or not torch.isfinite(pose).all():
        raise ValueError("Final pose observation is invalid.")
    return pose


def capture_final_goals(
    env: Any,
    deployment: Any,
    bundle: Path,
    graph: Mapping[str, Any],
    contract: Mapping[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Collect instantaneous final geometry, never reusing milestone success."""
    target = getattr(env, "unwrapped", env)
    simulation = target.sim
    registration = deployment.integration.registration
    bindings = {v.entity_id: v for v in registration.scene_binding.rigid_objects}
    constraints = load_config(bundle / "task_program/constraints.json")["presets"]
    groups = {v["id"]: v for v in graph["task_groups"]}
    spec = TaskSpec.from_dict(load_config(bundle / "task_spec.json"))
    declarations = {v.id: v for v in spec.goal}
    result: dict[str, Any] = {}
    for goal in contract["coverage"]["goals"]:
        predicate = declarations[goal["id"]]
        if _needs_universe(spec, predicate):
            result[predicate.id] = {
                "unavailable": "Complete selector-universe evidence was not provided."
            }
            continue
        observations = []
        errors = []
        for group_id in goal["groups"]:
            member = next(
                v
                for entries in contract["coverage"]["lineage"].values()
                for v in entries
                if v["group_id"] == group_id
            )
            uid = member["object_uid"]
            try:
                if predicate.name == "articulation_joint_near":
                    factory = next(
                        v
                        for v in registration.registered_semantic_lowerer_factories
                        if v.call_id == "gen_sim.articulation_slide"
                    )
                    part = next(
                        v
                        for v in factory.bindings
                        if v.object_id == uid
                        and (v.part_id or None) == member["object_part_id"]
                    )
                    native_binding = next(
                        v
                        for v in registration.scene_binding.articulations
                        if v.entity_id == uid
                    )
                    art = simulation.get_articulation(native_binding.simulation_uid)
                    values = (
                        art.get_qpos()
                        .detach()
                        .cpu()[:, art.joint_names.index(part.joint)]
                    )
                    expected = part.target(predicate.arguments["target_state"])
                    accepted = (values - expected).abs() <= part.tolerance(
                        predicate.arguments["target_state"]
                    )
                    metrics = {
                        "joint_position": values.tolist(),
                        "joint_target": expected,
                    }
                else:
                    cfg = next(
                        constraints[f"gen_sim.{node_id}.stable"]
                        for node_id in reversed(groups[group_id]["node_ids"])
                        if f"gen_sim.{node_id}.stable" in constraints
                        and constraints[f"gen_sim.{node_id}.stable"]["entity"] == uid
                    )
                    pose = _pose(simulation, bindings[uid])
                    if pose.shape[0] != num_envs:
                        raise ValueError("Final observation environment count differs.")
                    metrics = {"pose": pose.tolist()}
                    if predicate.name == "object_upright":
                        constraint = StabilityConstraint.decode(cfg)
                        axis = constraint.local_axis
                        if axis is None:
                            raise ValueError(
                                "No measured local upright axis is declared."
                            )
                        alignment = (
                            pose[:, :3, :3] @ torch.tensor(axis, dtype=pose.dtype)
                        )[:, 2]
                        accepted = alignment >= constraint.minimum_alignment
                        metrics["alignment"] = alignment.tolist()
                    elif predicate.name == "semantic_goal":
                        # Origins alone cannot prove containment, support or release.
                        raise ValueError(
                            "Complete final relation/release observer is not available."
                        )
                    else:
                        raise ValueError(
                            "This predicate needs an observer not provided by final geometry."
                        )
                if len(accepted) != num_envs:
                    raise ValueError("Final observation environment count differs.")
                observations.append(
                    {
                        "group_id": group_id,
                        "uid": uid,
                        "part_id": member["object_part_id"],
                        "accepted": accepted.tolist(),
                        "measurements": metrics,
                    }
                )
            except (
                KeyError,
                ValueError,
                TypeError,
                StopIteration,
                AttributeError,
            ) as error:
                errors.append(str(error) or type(error).__name__)
        result[predicate.id] = (
            {
                "members": observations,
                "values": [
                    all(v["accepted"][i] is True for v in observations)
                    for i in range(num_envs)
                ],
            }
            if observations and not errors
            else {"unavailable": errors or ["No final observation."]}
        )
    return result


def evaluate_runtime_evidence(
    spec: TaskSpec,
    contract: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    final_goals: Mapping[str, Any],
    *,
    num_envs: int,
    run_id: str,
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only newly collected final measurements and exact segment evidence."""
    if contract["task_spec_digest"] != canonical_hash(spec.to_dict()) or contract[
        "graph_digest"
    ] != canonical_hash(graph):
        raise ValueError("TaskSpec runtime evidence identity mismatch.")
    if canonical_hash(program) != contract["program_digest"]:
        raise ValueError("TaskSpec runtime program identity mismatch.")
    if canonical_hash(constraints) != contract["constraints_digest"]:
        raise ValueError("TaskSpec runtime constraints identity mismatch.")
    program_items = {v["name"]: v for v in program["program"]["items"]}
    members = {
        v["group_id"]: v
        for entries in contract["coverage"]["lineage"].values()
        for v in entries
    }
    groups = {v["id"]: v for v in graph["task_groups"]}
    segments = {}
    for segment in runtime.get("segments", []):
        if not isinstance(segment, Mapping):
            raise ValueError("Malformed segment evidence.")
        name = segment.get("name")
        if name in segments:
            raise ValueError("Duplicate segment evidence cannot certify a milestone.")
        segments[name] = segment
    rows = []
    for env_id in range(num_envs):
        goals = {}
        for predicate in spec.goal:
            record = final_goals.get(predicate.id, {})
            declared = next(
                v for v in contract["coverage"]["goals"] if v["id"] == predicate.id
            )
            observations = record.get("members", [])
            role = next(
                (r for r in spec.roles if r.name == predicate.arguments.get("subject")),
                None,
            )
            if role is None or _needs_universe(spec, predicate):
                continue
            expected = {
                (id, members[id]["object_uid"], members[id]["object_part_id"])
                for id in declared["groups"]
            }
            observed = {
                (v.get("group_id"), v.get("uid"), v.get("part_id"))
                for v in observations
            }
            if (
                expected
                and expected == observed
                and len(observations) == len(expected)
                and all(
                    isinstance(v.get("accepted"), list)
                    and len(v["accepted"]) == num_envs
                    and all(type(flag) is bool for flag in v["accepted"])
                    and isinstance(v.get("measurements"), Mapping)
                    and bool(v["measurements"])
                    for v in observations
                )
            ):
                goals[predicate.observation or predicate.id] = all(
                    v["accepted"][env_id] for v in observations
                )
        trajectory = []
        complete_groups: dict[str, set[str]] = {}
        measured_groups: dict[str, set[str]] = {}
        successful_nodes: set[str] = set()
        for node in graph["nodes"]:
            segment = segments.get(node["id"])
            if segment is None:
                continue
            group = groups[node["task_instance_id"]]
            success = segment.get("successes", [])
            identities = (
                segment.get("metadata", {}).get("runtime", {}).get("env_ids", [])
            )
            row = identities.index(env_id) if env_id in identities else -1
            active = segment.get("active", [])
            identities_valid = (
                isinstance(identities, list)
                and all(type(id) is int and 0 <= id < num_envs for id in identities)
                and len(set(identities)) == len(identities)
                and isinstance(success, list)
                and len(success) == len(identities)
                and all(type(v) is bool for v in success)
                and isinstance(active, list)
                and len(active) == len(identities)
                and all(type(v) is bool for v in active)
            )
            policies = segment.get("metadata", {}).get("post_policies", [])
            requested_presets = {
                p["preset"]
                for p in program_items[node["id"]].get("post", [])
                if p.get("kind") == "wait_stable"
                and p["preset"] in constraints["presets"]
                and constraints["presets"][p["preset"]]["entity"]
                == members[group["id"]]["object_uid"]
            }
            measured = {
                policy.get("result", {}).get("kind")
                for policy in policies
                if policy.get("result", {}).get("kind")
                in {"upright", "placement", "stack", "supported_placement"}
                and policy["result"].get("preset") in requested_presets
                and policy["result"]["kind"]
                == constraints["presets"][policy["result"]["preset"]]["kind"]
                and isinstance(policy["result"].get("measurements"), Mapping)
                and bool(policy["result"]["measurements"])
                and isinstance(policy["result"].get("accepted_mask"), list)
                and len(policy["result"]["accepted_mask"]) == num_envs
                and all(type(v) is bool for v in policy["result"]["accepted_mask"])
                and policy["result"]["accepted_mask"][env_id] is True
            }
            if (
                identities_valid
                and row >= 0
                and success[row] is True
                and active[row] is True
            ):
                successful_nodes.add(node["id"])
                measured_groups.setdefault(group["id"], set()).update(measured)
            if node["id"] != group["node_ids"][-1]:
                continue
            complete_groups[group["id"]] = (
                measured_groups.get(group["id"], set())
                if set(group["node_ids"]).issubset(successful_nodes)
                else set()
            )
            snapshot = {}
            for milestone in spec.milestones:
                record = next(
                    (
                        v
                        for v in contract["coverage"]["milestones"]
                        if v["id"] == milestone.id
                    ),
                    None,
                )
                if (
                    record
                    and record["groups"]
                    and all(complete_groups.get(id) for id in record["groups"])
                ):
                    for predicate in milestone.achieve:
                        if _needs_universe(spec, predicate):
                            continue
                        acceptable = (
                            {"upright"}
                            if predicate.name == "object_upright"
                            else (
                                {"placement", "stack", "supported_placement"}
                                if predicate.name == "semantic_goal"
                                else set()
                            )
                        )
                        if acceptable and all(
                            complete_groups[id].intersection(acceptable)
                            for id in record["groups"]
                        ):
                            snapshot[predicate.observation or predicate.id] = True
            trajectory.append(snapshot)
        # Sparse segment observations prove order, not continuous invariant/time windows.
        for condition in spec.temporal:
            for snapshot in trajectory:
                snapshot[condition.predicate.observation or condition.predicate.id] = {
                    "status": "unavailable",
                    "reason": "No complete control-step trajectory.",
                }
        evaluation = evaluate_task_spec(
            spec, {"init": {}, "goal": goals, "invariants": {}}, trajectory=trajectory
        )
        rows.append(
            {
                "env_id": env_id,
                **evaluation.to_dict(),
                "final_measurements": dict(final_goals),
                "evidence_scope": "final_geometry_and_measured_segment_order",
            }
        )
    return {
        "schema_version": "task_spec_evaluation/v1",
        "observer_revision": 1,
        "run_id": run_id,
        "task_spec_digest": contract["task_spec_digest"],
        "task_spec_semantic_hash": spec.semantic_hash,
        "graph_digest": contract["graph_digest"],
        "program_digest": contract["program_digest"],
        "constraints_digest": contract["constraints_digest"],
        "environments": rows,
    }
