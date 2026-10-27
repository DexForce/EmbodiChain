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
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch

from embodichain.gen_sim.task_spec import TaskSpec, evaluate_task_spec
from embodichain.utils.utility import load_config
from ..contracts import canonical_hash
from .stability import StabilityConstraint, _constraint_axis_alignment
from .placement_evidence import (
    _placement_predicate,
    _require_handover_receiver,
    measure_planar_placement,
    measured_release,
    placement_node,
)

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
    *,
    runtime: Mapping[str, Any] | None = None,
    pour_motions: Mapping[str, Any] | None = None,
    legacy_observer: bool = False,
) -> dict[str, Any]:
    """Collect final geometry and measured events, never generic milestone success."""
    target = getattr(env, "unwrapped", env)
    simulation = target.sim
    registration = deployment.integration.registration
    bindings = {v.entity_id: v for v in registration.scene_binding.rigid_objects}
    constraint_payload = load_config(bundle / "task_program/constraints.json")
    constraints = constraint_payload["presets"]
    groups = {v["id"]: v for v in graph["task_groups"]}
    handovers = {}
    if runtime is not None and any(g["task_type"] == "E4" for g in groups.values()):
        from .handover_evidence import capture_handover_operations

        handovers = capture_handover_operations(
            env,
            deployment,
            graph,
            runtime,
            load_config(bundle / "task_program/program.yaml"),
            constraint_payload,
            num_envs,
            final=True,
        )
    coordinated = {}
    if runtime is not None and any(g["task_type"] == "E5" for g in groups.values()):
        from .coordinated_evidence import capture_coordinated_holds

        coordinated = capture_coordinated_holds(
            env,
            deployment,
            graph,
            runtime,
            load_config(bundle / "task_program/program.yaml"),
            constraint_payload,
            num_envs,
            final=True,
        )
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
        diagnostics = []
        for group_id in goal["groups"]:
            member = next(
                v
                for entries in contract["coverage"]["lineage"].values()
                for v in entries
                if v["group_id"] == group_id
            )
            uid = member["object_uid"]
            try:
                if (
                    predicate.arguments.get("task_type") == "E5"
                    and predicate.name == "held_by_both_grippers"
                ):
                    record = coordinated.get(group_id, {})
                    if "accepted" not in record:
                        raise ValueError(
                            record.get(
                                "unavailable",
                                "No measured coordinated hold observation.",
                            )
                        )
                    observations.append(deepcopy(record))
                    continue
                if predicate.arguments.get("task_type") == "E4" and predicate.name in {
                    "held",
                    "orientation",
                    "handover_complete",
                }:
                    record = handovers.get(group_id, {}).get(predicate.name, {})
                    if "accepted" not in record:
                        raise ValueError(
                            record.get(
                                "unavailable", "No measured handover observation."
                            )
                        )
                    observations.append(deepcopy(record))
                    continue
                if predicate.arguments.get("task_type") in {"E8", "E9"}:
                    from .control_evidence import measure_control_operation

                    if runtime is None:
                        raise ValueError(
                            "Control goal requires measured operation receipts."
                        )
                    observations.append(
                        measure_control_operation(
                            deployment,
                            graph,
                            groups[group_id],
                            runtime,
                            load_config(bundle / "task_program/program.yaml"),
                            num_envs,
                        )
                    )
                    continue
                if predicate.name == "pour_motion" and not legacy_observer:
                    record = (pour_motions or {}).get(group_id, {})
                    if "accepted" not in record:
                        raise ValueError(
                            record.get("unavailable", "No measured pour-motion event.")
                        )
                    observations.append(deepcopy(record))
                    continue
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
                elif _placement_predicate(predicate) and not (
                    legacy_observer and predicate.arguments.get("task_type") == "E5"
                ):
                    if runtime is None:
                        raise ValueError(
                            "Placement requires measured execution release evidence."
                        )
                    segments = _segments(runtime)
                    if not segments:
                        raise ValueError(
                            "Placement has no execution boundary evidence."
                        )
                    record = _placement_record(
                        target,
                        deployment,
                        graph,
                        contract,
                        predicate,
                        member,
                        segments,
                        next(reversed(segments.values())),
                        num_envs,
                        legacy_observer=legacy_observer,
                        program=(
                            load_config(bundle / "task_program/program.yaml")
                            if predicate.arguments.get("task_type") == "E5"
                            else None
                        ),
                    )
                    observations.append(record)
                    continue
                elif predicate.name == "poured":
                    raise ValueError(
                        "Fluid transfer is unmodeled; tilt/restore pose evidence "
                        "does not certify the poured predicate."
                    )
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
                    direction_free = (
                        predicate.name == "orientation"
                        and predicate.arguments.get("orientation_goal") == "vertical"
                    )
                    if predicate.name == "object_upright" or direction_free:
                        constraint = StabilityConstraint.decode(cfg)
                        axis = constraint.local_axis
                        if axis is None:
                            raise ValueError(
                                "No measured local upright axis is declared."
                            )
                        if constraint.axis_direction_free is not direction_free:
                            raise ValueError(
                                "Orientation evidence uses a different axis-direction contract."
                            )
                        alignment = _constraint_axis_alignment(constraint, pose)
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
                if hasattr(error, "measurements"):
                    diagnostics.append(error.measurements)
        result[predicate.id] = (
            {
                "members": observations,
                "values": [
                    (
                        all(v["accepted"][i] is True for v in observations)
                        if all(
                            v.get("available") is None or v["available"][i]
                            for v in observations
                        )
                        else None
                    )
                    for i in range(num_envs)
                ],
            }
            if observations and not errors
            else {
                "unavailable": errors or ["No final observation."],
                **({"measurements": diagnostics} if diagnostics else {}),
            }
        )
    return result


def _segments(runtime: Mapping[str, Any]) -> dict[str, Any]:
    records = {}
    for segment in runtime.get("segments", []):
        if not isinstance(segment, Mapping) or segment.get("name") in records:
            raise ValueError("Malformed or duplicate execution boundary evidence.")
        records[segment["name"]] = segment
    return records


def _side_placement_record(
    target: Any,
    deployment: Any,
    graph: Mapping[str, Any],
    contract: Mapping[str, Any],
    predicate: Any,
    member: Mapping[str, Any],
    segments: Mapping[str, Any],
    boundary: Mapping[str, Any],
    num_envs: int,
    *,
    recorded: bool,
    legacy_observer: bool = False,
) -> dict[str, Any]:
    """Observe a bound rigid body/link, tabletop landing, and measured release."""
    from .placement_evidence import _native_mesh, _valid_pose
    from .spatial_reference import measure_relation, resolve_reference

    registration = deployment.integration.registration
    scene_binding = registration.scene_binding
    reference = contract["coverage"]["roles"].get(
        predicate.arguments.get("reference"), []
    )
    if len(reference) != 1 or member["object_part_id"] is not None:
        raise ValueError(
            "Side placement needs one bound reference and a rigid subject."
        )
    reference_uid, part = reference[0]["uid"], reference[0]["part_id"]
    group = next(g for g in graph["task_groups"] if g["id"] == member["group_id"])
    nodes = {n["id"]: n for n in graph["nodes"]}
    candidates = [
        nodes[key]
        for key in group["node_ids"]
        if nodes[key]["call"].get("call_id") == "simulation.place_relative"
        and nodes[key]["call"]["arguments"].get("object") == member["object_uid"]
        and nodes[key]["call"]["arguments"].get("reference") == reference_uid
        and nodes[key]["call"]["arguments"].get("reference_part") == part
        and nodes[key]["call"]["arguments"].get("relation")
        == predicate.arguments["relation"]
    ]
    if len(candidates) != 1:
        raise ValueError("Side predicate does not match one bound placement call.")
    node = candidates[0]
    stage = node["call"]["arguments"].get("stage_id", node["task_instance_id"])
    routes = [
        r
        for f in registration.registered_semantic_lowerer_factories
        if f.call_id == "simulation.place_relative"
        for r in f.routes
        if r.object_id == member["object_uid"]
        and r.reference_entity_id == reference_uid
        and r.reference_part == part
        and r.relation == predicate.arguments["relation"]
        and r.stage_id in {"", stage}
    ]
    if len(routes) != 1:
        raise ValueError("Side placement lacks one qualified reference route.")
    route = routes[0]
    if legacy_observer and route.reference_link_id is None:
        raise ValueError("Side placement lacks one qualified reference link route.")
    reference_id = route.reference_link_id or reference_uid
    preset = f"gen_sim.{node['id']}.stable"
    constraint = dict(deployment.integration.adapter_factory.constraints)[preset]
    expected_release = constraint.displacement
    if (
        node["task_type"] == "E1"
        and route.reference_link_id is None
        and route.reference_part is None
        and route.relation in {"left_of", "right_of", "front_of", "behind"}
        and expected_release is not None
    ):
        from ..task_program_bundle import (
            _LATERAL_RELEASE_CLEARANCE,
            _PLACEMENT_CLEARANCE,
        )

        # The generator deliberately separates the release command height from
        # its settled preset. Reconstruct that exact policy, not a Z tolerance.
        expected_release = (
            *expected_release[:2],
            expected_release[2] + (_LATERAL_RELEASE_CLEARANCE - _PLACEMENT_CLEARANCE),
        )
    if (
        constraint.entity != member["object_uid"]
        or constraint.reference != reference_id
        or expected_release != route.world_displacement
    ):
        raise ValueError("Side planning route and acceptance reference disagree.")
    bindings = {b.entity_id: b for b in scene_binding.rigid_objects}
    if route.reference_link_id is not None:
        link = next(b for b in scene_binding.links if b.entity_id == reference_id)
        parent = next(
            b for b in scene_binding.articulations if b.entity_id == reference_uid
        )
        if link.articulation_id != reference_uid:
            raise ValueError("Side reference link belongs to a different articulation.")
        art = target.sim.get_articulation(parent.simulation_uid)
        geometry = resolve_reference(
            {
                "uid": reference_uid,
                "fpath": art.cfg.fpath,
                "body_scale": art.cfg.body_scale,
                "fix_base": art.cfg.root_props.fixed_base,
                "init_local_pose": art.cfg.init_local_pose,
                "init_pos": art.cfg.init_pos,
                "init_rot": art.cfg.init_rot,
            },
            part,
        )
        if geometry.link != link.native_link_name:
            raise ValueError(
                "Side reference native link differs from its qualified part."
            )
        reference_vertices = geometry.vertices
    else:
        if part is not None or reference_uid not in bindings:
            raise ValueError("Side reference requires its exact bound rigid body.")
        reference_vertices, _ = _native_mesh(
            target.sim, bindings[reference_uid], num_envs
        )
    object_binding, table_binding = bindings[member["object_uid"]], bindings["table"]
    frames = None
    if recorded:
        frames = boundary.get("end_steps")
        counts = getattr(target, "_traj_steps", None)
        if (
            not isinstance(frames, (list, tuple))
            or len(frames) != num_envs
            or counts is None
            or any(
                type(v) is not int or not 0 < v <= int(counts[i])
                for i, v in enumerate(frames)
            )
        ):
            raise ValueError("Side milestone requires exact recorded boundary slots.")
        from embodichain.utils.math import xyz_quat_to_4x4_matrix

        def pose(binding: Any) -> torch.Tensor:
            values = target._traj_buffer["states"]["rigid_objects"][
                binding.simulation_uid
            ]["pose"]
            if (
                values.ndim != 3
                or values.shape[0] != num_envs
                or values.shape[-1] != 7
                or any(frame >= values.shape[1] for frame in frames)
            ):
                raise ValueError("Side milestone has no recorded native pose slot.")
            return xyz_quat_to_4x4_matrix(
                torch.stack([values[i, frame] for i, frame in enumerate(frames)])
                .detach()
                .cpu()
            )

        prefix = boundary["metadata"]["program_segment_source_path"]
        traces = [
            p
            for p in boundary["metadata"]["post_policies"]
            if p.get("kind") == "wait_stable"
            and type(p.get("policy_index")) is int
            and p.get("source_path") == [*prefix, "post", p["policy_index"]]
            and p.get("result", {}).get("measurements", {}).get("reference")
            == reference_id
            and p["result"]["measurements"].get("entity") == member["object_uid"]
            and p["result"].get("preset") == preset
            and p["result"].get("kind") == constraint.kind
            and p["result"].get("required_seconds") == constraint.duration
        ]
        if (
            len(traces) != 1
            or traces[0].get("result_mask") != [True] * num_envs
            or traces[0]["result"].get("accepted_mask") != [True] * num_envs
            or any(
                type(v) is not bool
                for v in traces[0]["result_mask"] + traces[0]["result"]["accepted_mask"]
            )
            or len(traces[0]["result"].get("stable_seconds", [])) != num_envs
            or any(
                type(v) not in (float, int)
                or not math.isfinite(v)
                or v < constraint.duration
                for v in traces[0]["result"]["stable_seconds"]
            )
        ):
            raise ValueError(
                "Side milestone lacks its exact accepted link observation."
            )
        measurement = traces[0]["result"]["measurements"]
        reference_pose = torch.tensor(measurement["reference_pose"])
        _valid_pose(reference_pose, num_envs)
        observed_object = torch.tensor(measurement["object_pose"])
        _valid_pose(observed_object, num_envs)
        if not torch.allclose(observed_object, pose(object_binding), atol=1e-4, rtol=0):
            raise ValueError(
                "Side milestone observation differs from its recorded boundary."
            )
        if route.reference_link_id is None and not torch.allclose(
            reference_pose, pose(bindings[reference_uid]), atol=1e-4, rtol=0
        ):
            raise ValueError("Side reference differs from its recorded boundary.")
    else:

        def pose(binding: Any) -> torch.Tensor:
            return _pose(target.sim, binding)

        reference_pose = (
            art.get_link_pose(link.native_link_name, to_matrix=True).detach().cpu()
            if route.reference_link_id is not None
            else pose(bindings[reference_uid])
        )
    object_pose = pose(object_binding)
    vertices, _ = _native_mesh(target.sim, object_binding, num_envs)
    accepted, metrics = measure_relation(
        vertices, object_pose, reference_vertices, reference_pose, route.relation
    )
    supported, support = measure_planar_placement(
        target.sim, object_binding, table_binding, object_pose, pose(table_binding)
    )
    released = measured_release(
        node, member["object_uid"], segments, boundary, num_envs
    )
    metrics.update(
        scope=(
            "bound_link_world_axes"
            if route.reference_link_id is not None
            else "bound_rigid_world_axes"
        ),
        reference_uid=reference_uid,
        reference_part=part,
        reference_link_id=route.reference_link_id,
        relation=route.relation,
        tabletop=support,
        released=released,
    )
    if recorded:
        metrics["boundary_frames"] = list(frames)
    return {
        "group_id": member["group_id"],
        "uid": member["object_uid"],
        "part_id": None,
        "accepted": [
            a and b and c for a, b, c in zip(accepted, supported, released, strict=True)
        ],
        "measurements": metrics,
    }


def _bare_table_node(node: Mapping[str, Any], predicate: Any) -> bool:
    call = node.get("call", {})
    arguments = call.get("arguments", {})
    return bool(
        predicate.name == "semantic_goal"
        and predicate.arguments.get("task_type") == "E1"
        and predicate.arguments.get("relation") == "on"
        and "orientation_goal" not in predicate.arguments
        and node.get("task_type") == "E1"
        and call.get("call_id") == "simulation.place_relative"
        and arguments.get("reference") == "table"
        and arguments.get("relation") == "on"
        and set(arguments) <= {"object", "reference", "relation", "stage_id"}
    )


def _bool_rows(values: Any, num_envs: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == num_envs
        and all(type(v) is bool for v in values)
    )


def _region_policy_matches(
    policy: Mapping[str, Any], cfg: Any, preset: str, num_envs: int
) -> bool:
    result = policy.get("result", {})
    duration = cfg.get("duration", 3.0) if isinstance(cfg, Mapping) else cfg.duration
    masks = (policy.get("result_mask"), result.get("accepted_mask"))
    seconds = result.get("stable_seconds", [])
    return bool(
        result.get("kind") == "supported_region"
        and result.get("preset") == preset
        and result.get("required_seconds") == duration
        and all(_bool_rows(mask, num_envs) and all(mask) for mask in masks)
        and isinstance(seconds, list)
        and len(seconds) == num_envs
        and all(
            type(v) in (int, float) and math.isfinite(v) and v >= duration
            for v in seconds
        )
    )


def _region_poses_match(
    measurements: Mapping[str, Any],
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor,
    num_envs: int,
) -> bool:
    from .placement_evidence import _valid_pose

    try:
        observed = torch.tensor(measurements["object_pose"]).to(object_pose)
        reference = torch.tensor(measurements["reference_pose"]).to(reference_pose)
        _valid_pose(observed, num_envs)
        _valid_pose(reference, num_envs)
        return bool(
            torch.allclose(observed, object_pose, atol=1e-4, rtol=0)
            and torch.allclose(reference, reference_pose, atol=1e-4, rtol=0)
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _release_window(
    node: Mapping[str, Any], segments: Mapping[str, Any]
) -> dict[str, Any]:
    """Add time bounds after ``measured_release`` verified the physical receipt."""
    calls = segments[node["id"]]["metadata"]["runtime"].get("calls", [])
    if len(calls) != 1 or calls[0].get("status") != "completed":
        raise ValueError("Region release requires one exact completed invocation.")
    call = calls[0]
    receipts = [
        receipt
        for receipt in call.get("effects", [])
        if receipt.get("boundary", {}).get("kind") == "terminal"
        and receipt.get("effect_spec", {}).get("effect_kind") == "release"
        and receipt["effect_spec"].get("invocation_id") == call["invocation_id"]
        and receipt["effect_spec"].get("invocation_revision")
        == call["invocation_revision"]
        and any(
            expectation.get("object_id") == node["call"]["arguments"]["object"]
            and expectation.get("resource_id") == node["call"]["resources"]["primary"]
            and expectation.get("relation") == "detached"
            for expectation in receipt["effect_spec"].get("state_expectations", [])
        )
    ]
    completions = [
        event["timestamp"]
        for event in call.get("events", [])
        if event.get("kind") == "action_completed"
        and event.get("invocation_id") == call.get("invocation_id")
        and event.get("invocation_revision") == call.get("invocation_revision")
    ]
    if not receipts or len(completions) != 1:
        raise ValueError("Region release lacks its terminal execution window.")
    timestamp = receipts[-1]["timestamp"]
    planned, completed = call["plan_attempts"][-1]["planned_at"], completions[0]
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (timestamp, planned, completed)
        )
        or not planned <= timestamp <= completed
    ):
        raise ValueError("Region release lies outside its actual execution window.")
    return {
        "invocation_id": call["invocation_id"],
        "invocation_revision": call["invocation_revision"],
        "timestamp": timestamp,
        "planned_at": planned,
        "completed_at": completed,
    }


def _strong_support_trace(
    boundary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    preset: str,
    uid: str,
    reference: str,
    num_envs: int,
) -> Mapping[str, Any]:
    """Resolve the declared terminal policy without borrowing another occurrence."""
    prefix = boundary["metadata"]["program_segment_source_path"]
    traces = [
        p
        for p in boundary["metadata"].get("post_policies", [])
        if p.get("kind") == "wait_stable"
        and type(p.get("policy_index")) is int
        and p.get("source_path") == [*prefix, "post", p["policy_index"]]
        and p.get("result", {}).get("preset") == preset
    ]
    if len(traces) != 1:
        raise ValueError("Strong support lacks its unique terminal policy.")
    trace, result = traces[0], traces[0]["result"]
    native = result.get("measurements", {}).get("native_support", {})
    duration = cfg.get("duration", 3.0)
    seconds = result.get("stable_seconds", [])
    start, end = result.get("start_step"), result.get("end_step")
    started, ended, dt = (
        result.get("clock_started_at"),
        result.get("clock_ended_at"),
        native.get("step_dt"),
    )
    if (
        result.get("kind") != cfg["kind"]
        or result.get("support_contract_revision") != 1
        or result.get("entity") != uid
        or result.get("reference") != reference
        or result.get("required_seconds") != duration
        or any(
            not _bool_rows(v, num_envs) or not all(v)
            for v in (trace.get("result_mask"), result.get("accepted_mask"))
        )
        or not isinstance(seconds, list)
        or len(seconds) != num_envs
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < duration
            for v in seconds
        )
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (started, ended, dt)
        )
        or dt == 0
        or not math.isclose(started, start * dt, rel_tol=0, abs_tol=1e-8)
        or not math.isclose(ended, end * dt, rel_tol=0, abs_tol=1e-8)
        or ended - started + 1e-8 < duration
        or any(v > ended - started + 1e-8 for v in seconds)
        or native.get("observed_step") != end
        or native.get("observed_at") != ended
        or boundary.get("end_steps") != [end] * num_envs
    ):
        raise ValueError("Strong support policy lacks its exact stable clock/boundary.")
    return result


def _strong_native_loads(native: Mapping[str, Any], num_envs: int) -> list[bool]:
    """Recheck the actual pair and load rather than trusting derived booleans."""
    import numpy as np
    from .support_evidence import support_from_contacts

    result = []
    for row, pair in zip(native["rows"], native["actor_pairs"], strict=True):
        raw = row.get("raw_contacts", {})
        pairs = np.asarray(raw.get("user_ids", []))
        points, normals = (
            np.asarray(raw.get(key, [])) for key in ("position", "normal")
        )
        impulses = np.asarray(raw.get("impulse", []))
        center = np.asarray(row.get("center_of_mass_world"))
        tolerance = row.get("com_tolerance_m")
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or pairs.dtype.kind not in "iu"
            or points.shape != (len(pairs), 3)
            or normals.shape != points.shape
            or impulses.shape != (len(pairs),)
            or center.shape != (3,)
            or not np.isfinite(center).all()
            or type(tolerance) not in (int, float)
            or not math.isfinite(tolerance)
            or not 0 < tolerance <= 0.01
        ):
            raise ValueError(
                "Strong native support lacks its original contact payload."
            )
        accepted, recomputed = support_from_contacts(
            pairs,
            points,
            normals,
            impulses,
            center,
            object_actor=pair["object_actor"],
            reference_actor=pair["reference_actor"],
            tolerance=tolerance,
        )
        if row.get("contacts_world") != recomputed["contacts_world"] or any(
            type(row.get(key)) not in (int, float)
            or type(recomputed[key]) not in (int, float)
            or not math.isclose(row[key], recomputed[key], rel_tol=1e-6, abs_tol=1e-12)
            for key in ("com_patch_distance_m", "upward_impulse")
        ):
            raise ValueError(
                "Strong native support derived patch differs from raw contacts."
            )
        result.append(accepted)
    if len(result) != num_envs:
        raise ValueError("Strong native support differs from the environment batch.")
    return result


def _strong_support_flags(
    native: Mapping[str, Any],
    landing: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
    window: Mapping[str, Any],
    trace: Mapping[str, Any],
    uid: str,
    reference: str,
    num_envs: int,
    contacts: list[Any] | None = None,
) -> list[bool]:
    from .support_contract import support_window_flags
    from .placement_evidence import _valid_pose
    from .stability import _contact_support_heights, _constraint_pose_flags

    pose, ref_pose = torch.tensor(native["object_pose"]), torch.tensor(
        native["reference_pose"]
    )
    _valid_pose(pose, num_envs)
    _valid_pose(ref_pose, num_envs)
    flags = support_window_flags(
        native,
        window,
        num_envs,
        entity=uid,
        reference=reference,
        native_entity=uid,
        native_reference=reference,
    )
    origin = trace["measurements"]["native_support"]
    original_pose, original_ref = torch.tensor(origin["object_pose"]), torch.tensor(
        origin["reference_pose"]
    )
    _valid_pose(original_pose, num_envs)
    _valid_pose(original_ref, num_envs)
    constraint = StabilityConstraint.decode(dict(cfg))
    if not _constraint_pose_flags(constraint, original_pose, original_ref).all():
        raise ValueError(
            "Strong history violates its original placement target or axes."
        )
    static_flags = _constraint_pose_flags(constraint, pose, ref_pose).tolist()
    flags = [a and b for a, b in zip(flags, static_flags, strict=True)]
    if (
        native.get("episode_generations") != origin.get("episode_generations")
        or native.get("actor_pairs") != origin.get("actor_pairs")
        or trace["clock_started_at"] + 1e-8 < window["completed_at"]
    ):
        raise ValueError("Strong support differs from the release episode or actors.")
    source_flags = support_window_flags(
        origin,
        window,
        num_envs,
        entity=uid,
        reference=reference,
        native_entity=uid,
        native_reference=reference,
    )
    source_supported = origin.get("supported", [])
    if (
        not _bool_rows(source_supported, num_envs)
        or not all(source_supported)
        or not all(source_flags)
        or not all(_strong_native_loads(origin, num_envs))
    ):
        raise ValueError("Strong stability lacks its own qualified native sample.")
    if not torch.isfinite(
        _contact_support_heights(
            torch.tensor(source_supported, dtype=torch.bool), origin, pose[:, 2, 3]
        )
    ).all():
        raise ValueError("Strong stability lacks its own qualified load patch.")
    supported = native.get("supported", [])
    if not _bool_rows(supported, num_envs):
        raise ValueError("Strong support lacks measured load flags.")
    loads = torch.isfinite(
        _contact_support_heights(
            torch.tensor(supported, dtype=torch.bool), native, pose[:, 2, 3]
        )
    ).tolist()
    raw_loads = _strong_native_loads(native, num_envs)
    for row, item in enumerate(native["rows"]):
        sample = item.get("sample")
        flags[row] &= bool(
            supported[row] and loads[row] and raw_loads[row] and sample is not None
        )
    if cfg["kind"] == "supported_region":
        if not isinstance(landing, Mapping):
            raise ValueError(
                "Strong region support lacks its complete landing geometry."
            )
        original_pose, original_ref = torch.tensor(origin["object_pose"]), torch.tensor(
            origin["reference_pose"]
        )
        if not all(
            _region_flags(
                origin,
                trace["measurements"].get("landing_region", {}),
                original_pose,
                original_ref,
                uid,
                trace["measurements"].get("support_gap", []),
                num_envs,
                reference_uid=reference,
            )
        ):
            raise ValueError("Strong stability lacks its own complete planar support.")
        # The collector computes this from the same native object mesh and patch.
        physical = _region_flags(
            native,
            landing,
            pose,
            ref_pose,
            uid,
            contacts or [],
            num_envs,
            reference_uid=reference,
        )
        flags = [a and b for a, b in zip(flags, physical, strict=True)]
    return flags


def _strong_placement_record(
    target: Any,
    node: Mapping[str, Any],
    group: Mapping[str, Any],
    configured: StabilityConstraint | None,
    object_binding: Any,
    reference_binding: Any,
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
    predicate: Any,
    member: Mapping[str, Any],
    segments: Mapping[str, Any],
    boundary: Mapping[str, Any],
    num_envs: int,
    *,
    recorded: bool,
    frames: list[int] | None,
) -> dict[str, Any]:
    """Collect the release occurrence and its independent historical/current support."""
    from dataclasses import asdict
    from .placement_evidence import _native_mesh
    from .stability import _contact_support_heights
    from .support_contract import release_window

    uid, reference = member["object_uid"], reference_binding.entity_id
    if (
        configured is None
        or configured.support_contract_revision != 1
        or configured.kind not in {"stack", "supported_placement", "supported_region"}
        or configured.entity != uid
        or configured.reference != reference
        or object_binding.simulation_uid != uid
        or reference_binding.simulation_uid != reference
        or predicate.arguments.get("relation") not in {"on", "inside"}
        or predicate.arguments["relation"] == "inside"
        and configured.kind != "supported_region"
    ):
        raise ValueError("Strong placement lacks its frozen supported rigid binding.")
    cfg, preset = asdict(configured), f"gen_sim.{node['id']}.stable"
    source = segments[group["node_ids"][-1]]
    trace = _strong_support_trace(source, cfg, preset, uid, reference, num_envs)
    released, window = release_window(node, segments, boundary, num_envs)
    if recorded:
        native = deepcopy(trace["measurements"]["native_support"])
    else:
        sensor = target.sim.get_sensor(f"gen_sim_support__{uid}__{reference}")
        reader = getattr(sensor, "support_reader", None)
        if (
            reader is None
            or reader.entity_id != uid
            or reader.reference_id != reference
        ):
            raise ValueError(
                "Strong final observation lacks its canonical native reader."
            )
        if reader.object is not target.sim.get_rigid_object(
            uid
        ) or reader.reference is not target.sim.get_rigid_object(reference):
            raise ValueError(
                "Strong reader differs from current frozen native bindings."
            )
        _, native = reader.observe(pose, reference_pose, support_contract_revision=1)
    if not _region_poses_match(native, pose, reference_pose, num_envs):
        raise ValueError("Strong support differs from exact observed boundary poses.")
    landing, contacts = None, None
    if configured.kind == "supported_region":
        _, landing = measure_planar_placement(
            target.sim, object_binding, reference_binding, pose, reference_pose
        )
        if recorded and not _region_poses_match(
            trace["measurements"].get("landing_region", {}),
            pose,
            reference_pose,
            num_envs,
        ):
            raise ValueError(
                "Strong landing region differs from recorded boundary poses."
            )
        vertices, _ = _native_mesh(target.sim, object_binding, num_envs)
        bottom = (vertices.to(pose) @ pose[:, 2, :3, None]).squeeze(-1).amin(-1) + pose[
            :, 2, 3
        ]
        heights = _contact_support_heights(
            torch.tensor(native["supported"], dtype=torch.bool), native, bottom
        )
        contacts = [float(v) if torch.isfinite(v) else None for v in bottom - heights]
    flags = _strong_support_flags(
        native, landing, cfg, window, trace, uid, reference, num_envs, contacts
    )
    metrics = {
        "scope": "native_supported_placement",
        "support_contract_revision": 1,
        "placement_node_id": node["id"],
        "support_boundary_node_id": source["name"],
        "boundary_node_id": boundary["name"],
        "stability_preset": preset,
        "observation_kind": "recorded" if recorded else "final",
        "native_support": native,
        "reference_uid": reference,
        "relation": predicate.arguments["relation"],
        "released": released,
        "release_window": window,
    }
    if landing is not None:
        metrics["landing_region"] = landing
        metrics["support_gap_m"] = contacts
    if recorded:
        metrics["boundary_frames"] = list(frames)
    return {
        "group_id": group["id"],
        "uid": uid,
        "part_id": member["object_part_id"],
        "accepted": [a and b for a, b in zip(flags, released, strict=True)],
        "measurements": metrics,
    }


def _strong_placement_matches(
    record: Mapping[str, Any],
    predicate: Any,
    roles: Mapping[str, Any],
    num_envs: int,
    *,
    graph: Mapping[str, Any],
    constraints: Mapping[str, Any],
    segments: Mapping[str, Any],
) -> bool:
    """Require the explicit strong scope; never downgrade to a legacy boolean."""
    from .support_contract import release_window

    try:
        metrics, uid = record["measurements"], record["uid"]
        references = roles.get(predicate.arguments.get("reference"), [])
        if len(references) != 1:
            return False
        group = next(g for g in graph["task_groups"] if g["id"] == record["group_id"])
        node = next(
            n for n in graph["nodes"] if n["id"] == metrics["placement_node_id"]
        )
        raw_call = node["call"]
        reference = references[0]["uid"]
        preset = f"gen_sim.{node['id']}.stable"
        cfg = constraints["presets"][preset]
        if raw_call.get("kind") == "place":
            from .drawer_binding import inside_parts

            relation = predicate.arguments.get("relation")
            if not (
                relation == "inside"
                and inside_parts(raw_call.get("inside", "")) == (reference, uid, None)
                or relation == "on"
                and raw_call.get("on") == f"on__{reference}__{uid}"
            ):
                return False
        source = segments[group["node_ids"][-1]]
        recorded = metrics["observation_kind"] == "recorded"
        boundary = source if recorded else next(reversed(segments.values()))
        if (
            metrics.get("scope") != "native_supported_placement"
            or metrics.get("support_contract_revision") != 1
            or not _placement_predicate(predicate)
            or cfg.get("support_contract_revision") != 1
            or cfg.get("kind")
            not in {"stack", "supported_placement", "supported_region"}
            or cfg.get("entity") != uid
            or cfg.get("reference") != reference
            or len(references) != 1
            or references[0].get("part_id") is not None
            or record.get("part_id") is not None
            or group["task_type"] != predicate.arguments.get("task_type")
            or node["id"] not in group["node_ids"]
            or node.get("task_type") != group["task_type"]
            or raw_call.get("object", raw_call.get("arguments", {}).get("object"))
            != uid
            or not any(
                v == {"uid": uid, "part_id": None}
                for v in roles.get(predicate.arguments.get("subject"), [])
            )
            or metrics.get("reference_uid") != reference
            or metrics.get("relation") != predicate.arguments.get("relation")
            or metrics["relation"] not in {"on", "inside"}
            or raw_call.get("kind") not in {"place", "registered"}
            or raw_call.get("kind") == "place"
            and not raw_call.get(metrics["relation"])
            or metrics["relation"] == "inside"
            and (
                raw_call.get("kind") != "place"
                or not raw_call.get("inside")
                or cfg["kind"] != "supported_region"
            )
            or raw_call.get("kind") == "registered"
            and raw_call.get("call_id")
            not in {
                "simulation.place_relative",
                "gen_sim.place_upright",
                "gen_sim.stack_place",
            }
            or raw_call.get("kind") == "registered"
            and (
                raw_call.get("arguments", {}).get("reference") != reference
                or raw_call["arguments"].get("relation") != metrics["relation"]
            )
            or metrics.get("stability_preset") != preset
            or metrics.get("boundary_node_id") != boundary["name"]
            or metrics.get("support_boundary_node_id") != source["name"]
            or metrics.get("observation_kind") not in {"recorded", "final"}
        ):
            return False
        trace = _strong_support_trace(source, cfg, preset, uid, reference, num_envs)
        released, window = release_window(node, segments, boundary, num_envs)
        native = metrics["native_support"]
        if (
            metrics.get("released") != released
            or metrics.get("release_window") != window
        ):
            return False
        if recorded:
            original = trace["measurements"]["native_support"]
            if (
                native != original
                or metrics.get("boundary_frames") != boundary.get("end_steps")
                or cfg["kind"] == "supported_region"
                and metrics.get("landing_region")
                != trace["measurements"].get("landing_region")
            ):
                return False
        physical = _strong_support_flags(
            native,
            metrics.get("landing_region"),
            cfg,
            window,
            trace,
            uid,
            reference,
            num_envs,
            metrics.get("support_gap_m"),
        )
        accepted = record.get("accepted")
        return _bool_rows(accepted, num_envs) and all(
            a == (p and r) for a, p, r in zip(accepted, physical, released, strict=True)
        )
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        StopIteration,
        AttributeError,
    ):
        return False


def _containment_available(metrics: Mapping[str, Any]) -> list[bool]:
    """Keep ambiguous 3D cavity evidence local to each released payload row."""
    from .placement_evidence import _support_gap_valid

    return [
        not (free and raised and overlap and not footprint and _support_gap_valid(gap))
        for free, raised, overlap, footprint, gap in zip(
            metrics["released"],
            metrics["reference_has_raised_geometry"],
            metrics["reference_aabb_overlap"],
            metrics["full_footprint_inside"],
            metrics["support_gap_m"],
            strict=True,
        )
    ]


def _placement_record(
    target: Any,
    deployment: Any,
    graph: Mapping[str, Any],
    contract: Mapping[str, Any],
    predicate: Any,
    member: Mapping[str, Any],
    segments: Mapping[str, Any],
    boundary: Mapping[str, Any],
    num_envs: int,
    *,
    recorded: bool = False,
    legacy_observer: bool = False,
    program: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if predicate.arguments.get("task_type") == "E5":
        if legacy_observer:
            raise ValueError(
                "Complete final relation/release observer is not available."
            )
        from .coordinated_placement_evidence import capture_transport_placement

        if program is None:
            raise ValueError("E5 placement requires its exact compiled program.")
        group = next(
            value for value in graph["task_groups"] if value["id"] == member["group_id"]
        )
        return capture_transport_placement(
            target,
            deployment,
            graph,
            {"segments": list(segments.values())},
            program,
            group,
            member,
            boundary,
            num_envs,
            recorded=recorded,
        )
    registration = deployment.integration.registration
    strong = (
        getattr(
            getattr(deployment.integration, "adapter_factory", None),
            "support_contract_revision",
            None,
        )
        == 1
    )
    from .spatial_reference import SIDE_RELATIONS

    if predicate.arguments.get("relation") in SIDE_RELATIONS:
        if strong:
            raise ValueError("Strong support does not qualify this placement relation.")
        return _side_placement_record(
            target,
            deployment,
            graph,
            contract,
            predicate,
            member,
            segments,
            boundary,
            num_envs,
            recorded=recorded,
            legacy_observer=legacy_observer,
        )
    scene_binding = registration.scene_binding
    bindings = {v.entity_id: v for v in scene_binding.rigid_objects}
    groups = {v["id"]: v for v in graph["task_groups"]}
    nodes = {v["id"]: v for v in graph["nodes"]}
    group_id = member["group_id"]
    node, reference_uid = placement_node(
        groups[group_id],
        nodes,
        predicate,
        member,
        contract["coverage"]["roles"],
        scene_binding,
        support_contract_revision=1 if strong else None,
    )
    uid = member["object_uid"]
    preset = f"gen_sim.{node['id']}.stable"
    configured = dict(
        getattr(deployment.integration.adapter_factory, "constraints", ())
        if hasattr(deployment.integration, "adapter_factory")
        else ()
    ).get(preset)
    region_scope = (
        _bare_table_node(node, predicate)
        and getattr(configured, "kind", None) == "supported_region"
    )
    if (
        region_scope
        and not strong
        and (
            configured.entity != uid
            or configured.reference != reference_uid
            or getattr(configured, "displacement", None) is not None
            or getattr(configured, "target_position", None) is not None
        )
    ):
        raise ValueError("Region placement preset differs from its exact graph call.")
    if (
        uid not in bindings
        or reference_uid not in bindings
        or member["object_part_id"] is not None
    ):
        raise ValueError(
            "Placement requires bound native rigid object/reference geometry."
        )
    frames = None
    if recorded:
        frames = boundary.get("end_steps")
        if not isinstance(frames, (list, tuple)) or len(frames) != num_envs:
            raise ValueError(
                "Placement milestone requires exact per-environment end steps."
            )
        counts = getattr(target, "_traj_steps", None)
        if (
            counts is None
            or len(counts) != num_envs
            or any(
                type(frame) is not int or not 0 < frame <= int(counts[row])
                for row, frame in enumerate(frames)
            )
        ):
            raise ValueError("Placement milestone boundary is not fully recorded.")
        states = target._traj_buffer["states"]["rigid_objects"]

        def pose(binding: Any) -> torch.Tensor:
            values = states[binding.simulation_uid]["pose"]
            if (
                values.ndim != 3
                or values.shape[0] != num_envs
                or values.shape[-1] != 7
                or any(frame >= values.shape[1] for frame in frames)
            ):
                raise ValueError("Placement milestone has no boundary state slot.")
            from embodichain.utils.math import xyz_quat_to_4x4_matrix

            # Recorder rows are pre-action; the seeded end_step slot is the
            # post-segment state. Never substitute end_step - 1 or the final pose.
            return xyz_quat_to_4x4_matrix(
                torch.stack([values[row, frame] for row, frame in enumerate(frames)])
                .detach()
                .cpu()
            )

    else:

        def pose(binding: Any) -> torch.Tensor:
            return _pose(target.sim, binding)

    object_pose, reference_pose = pose(bindings[uid]), pose(bindings[reference_uid])
    if object_pose.shape[0] != num_envs:
        raise ValueError("Placement observation environment count differs.")
    if strong:
        return _strong_placement_record(
            target,
            node,
            groups[group_id],
            configured,
            bindings[uid],
            bindings[reference_uid],
            object_pose,
            reference_pose,
            predicate,
            member,
            segments,
            boundary,
            num_envs,
            recorded=recorded,
            frames=frames,
        )
    if (
        not legacy_observer
        and not region_scope
        and predicate.arguments["relation"] == "on"
        and getattr(target, "_gen_sim_acceptance_contacts", None) is not None
    ):
        from .stable_support_evidence import capture_stable_support

        released, release_receipt, _ = measured_release(
            node, uid, segments, boundary, num_envs, with_receipt=True
        )
        observed = list(frames) if recorded else target._traj_steps.tolist()
        accepted, metrics = capture_stable_support(
            target,
            bindings,
            uid,
            reference_uid,
            observed[0],
            configured,
            release_receipt["timestamp"],
        )
        metrics.update(
            reference_uid=reference_uid,
            relation="on",
            released=released,
            placement_node_id=node["id"],
            boundary_frames=observed,
            observation_kind="recorded" if recorded else "final",
        )
        return {
            "group_id": group_id,
            "uid": uid,
            "part_id": member["object_part_id"],
            "accepted": [a and b for a, b in zip(accepted, released, strict=True)],
            "measurements": metrics,
        }
    if predicate.arguments["relation"] == "on":
        if recorded:
            prefix = boundary["metadata"]["program_segment_source_path"]
            traces = [
                p
                for p in boundary["metadata"].get("post_policies", [])
                if p.get("kind") == "wait_stable"
                and type(p.get("policy_index")) is int
                and p.get("source_path") == [*prefix, "post", p["policy_index"]]
                and p.get("result", {})
                .get("measurements", {})
                .get("native_support", {})
                .get("entity")
                == uid
                and p["result"]["measurements"]["native_support"].get("reference")
                == reference_uid
                and (
                    not region_scope
                    or _region_policy_matches(p, configured, preset, num_envs)
                )
            ]
            if len(traces) != 1:
                raise ValueError(
                    "Support milestone lacks its exact native observation."
                )
            native_record = traces[0]["result"]["measurements"]["native_support"]
            metrics = deepcopy(native_record) if region_scope else dict(native_record)
            poses_match = (
                _region_poses_match(metrics, object_pose, reference_pose, num_envs)
                if region_scope
                else torch.allclose(
                    torch.tensor(metrics["object_pose"]).to(object_pose),
                    object_pose,
                    atol=1e-4,
                    rtol=0,
                )
                and torch.allclose(
                    torch.tensor(metrics["reference_pose"]).to(reference_pose),
                    reference_pose,
                    atol=1e-4,
                    rtol=0,
                )
            )
            if not poses_match:
                raise ValueError(
                    "Support observation differs from recorded boundary poses."
                )
            accepted = metrics["supported"]
        else:
            from .support_evidence import NativeSupportObservation

            observer = NativeSupportObservation(
                target.sim,
                target.sim.get_rigid_object(bindings[uid].simulation_uid),
                target.sim.get_rigid_object(bindings[reference_uid].simulation_uid),
                entity_id=uid,
                reference_id=reference_uid,
            )
            supported, metrics = observer.observe(object_pose, reference_pose)
            accepted = supported.tolist()
    else:
        accepted, metrics = measure_planar_placement(
            target.sim,
            bindings[uid],
            bindings[reference_uid],
            object_pose,
            reference_pose,
            exact_silhouette=not legacy_observer,
        )
    released = measured_release(node, uid, segments, boundary, num_envs)
    if region_scope:
        from .placement_evidence import _native_mesh
        from .stability import _contact_support_heights

        native = metrics
        _, landing = measure_planar_placement(
            target.sim,
            bindings[uid],
            bindings[reference_uid],
            object_pose,
            reference_pose,
        )
        if recorded:
            observed = traces[0]["result"]["measurements"].get("landing_region", {})
            if not _region_poses_match(observed, object_pose, reference_pose, num_envs):
                raise ValueError("Landing region differs from recorded boundary poses.")
        vertices, _ = _native_mesh(target.sim, bindings[uid], num_envs)
        bottom = (vertices.to(object_pose) @ object_pose[:, 2, :3, None]).squeeze(
            -1
        ).amin(-1) + object_pose[:, 2, 3]
        heights = _contact_support_heights(
            torch.tensor(accepted, dtype=torch.bool), native, bottom
        )
        gaps = bottom - heights
        window = _release_window(node, segments)
        contact_gaps = [float(g) if torch.isfinite(g) else None for g in gaps]
        accepted = _region_flags(
            native, landing, object_pose, reference_pose, uid, contact_gaps, num_envs
        )
        metrics = {
            "scope": "native_supported_landing_region",
            "placement_node_id": node["id"],
            "boundary_node_id": boundary["name"],
            "observation_kind": "recorded" if recorded else "final",
            "native_support": native,
            "landing_region": landing,
            "support_gap_m": contact_gaps,
            "contact_tolerance_m": 0.01,
            "release_window": window,
        }
    metrics.update(
        reference_uid=reference_uid,
        relation=predicate.arguments["relation"],
        released=released,
    )
    if predicate.arguments.get("task_type") == "E4":
        metrics.update(
            placement_node_id=node["id"],
            observation_kind="recorded" if recorded else "final",
        )
    if recorded:
        metrics["boundary_frames"] = list(frames)
    return {
        "group_id": group_id,
        "uid": uid,
        "part_id": member["object_part_id"],
        "accepted": [a and b for a, b in zip(accepted, released, strict=True)],
        "measurements": metrics,
        **(
            {"available": _containment_available(metrics)}
            if not legacy_observer and predicate.arguments["relation"] == "inside"
            else {}
        ),
    }


def capture_placement_milestones(
    env: Any,
    deployment: Any,
    graph: Mapping[str, Any],
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    num_envs: int,
    *,
    spec: TaskSpec,
    legacy_observer: bool = False,
    program: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure placement at its own boundary, not at a later terminal state."""
    target = getattr(env, "unwrapped", env)
    segments = _segments(runtime)
    groups = {v["id"]: v for v in graph["task_groups"]}
    declarations = {m.id: m for m in spec.milestones}
    predicates = {}
    for coverage in contract["coverage"]["milestones"]:
        milestone = declarations[coverage["id"]]
        for predicate in milestone.achieve:
            if _placement_predicate(predicate):
                for group_id in coverage["groups"]:
                    predicates[group_id] = predicate
    result = {}
    for entries in contract["coverage"]["lineage"].values():
        for member in entries:
            group = groups[member["group_id"]]
            if group["task_type"] not in {"E1", "E4", "E5"} or (
                legacy_observer and group["task_type"] == "E5"
            ):
                continue
            terminal = segments.get(group["node_ids"][-1])
            if terminal is None:
                continue
            predicate = predicates.get(group["id"])
            if predicate is None or _needs_universe(spec, predicate):
                continue
            try:
                result[group["id"]] = _placement_record(
                    target,
                    deployment,
                    graph,
                    contract,
                    predicate,
                    member,
                    segments,
                    terminal,
                    num_envs,
                    recorded=True,
                    legacy_observer=legacy_observer,
                    program=program,
                )
            except (
                KeyError,
                ValueError,
                TypeError,
                AttributeError,
                StopIteration,
            ) as error:
                result[group["id"]] = {
                    "unavailable": str(error) or type(error).__name__,
                    **(
                        {"measurements": error.measurements}
                        if hasattr(error, "measurements")
                        else {}
                    ),
                }
    return result


def _region_flags(
    native: Mapping[str, Any],
    landing: Mapping[str, Any],
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
    uid: str,
    contacts: list[Any],
    num_envs: int,
    *,
    reference_uid: str = "table",
) -> list[bool]:
    """Conjoin exact native load and the complete measured planar footprint."""
    from .placement_evidence import (
        SUPPORT_TOLERANCE_M,
        PENETRATION_TOLERANCE_M,
        _support_gap_valid,
    )
    from .stability import _contact_support_heights

    supported = native.get("supported", [])
    footprints = landing.get("full_footprint_inside", [])
    gaps, areas = landing.get("support_gap_m", []), landing.get("footprint_area_m2", [])
    if (
        native.get("scope") != "native_contact_support"
        or native.get("entity") != uid
        or native.get("reference") != reference_uid
        or native.get("env_ids") != list(range(num_envs))
        or any(type(v) is not int for v in native.get("env_ids", []))
        or not _region_poses_match(native, pose, reference_pose, num_envs)
        or not _region_poses_match(landing, pose, reference_pose, num_envs)
        or landing.get("scope") != "measured_planar_landing_region"
        or landing.get("support_tolerance_m") != SUPPORT_TOLERANCE_M
        or landing.get("penetration_tolerance_m") != PENETRATION_TOLERANCE_M
        or not all(_bool_rows(v, num_envs) for v in (supported, footprints))
        or any(
            not isinstance(values, list)
            or len(values) != num_envs
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            for values in (gaps, areas)
        )
        or any(v <= 0 for v in areas)
        or not isinstance(contacts, list)
        or len(contacts) != num_envs
    ):
        raise ValueError(
            "Region placement lacks exact contact/planar geometry evidence."
        )
    qualified = torch.isfinite(
        _contact_support_heights(
            torch.tensor(supported, dtype=torch.bool), native, pose[:, 2, 3]
        )
    ).tolist()
    return [
        bool(
            s
            and q
            and f
            and _support_gap_valid(g)
            and type(c) in (int, float)
            and math.isfinite(c)
            and abs(c) <= 0.01
        )
        for s, q, f, g, c in zip(
            supported, qualified, footprints, gaps, contacts, strict=True
        )
    ]


def _placement_measurement_matches(
    record: Mapping[str, Any],
    predicate: Any,
    roles: Mapping[str, Any],
    num_envs: int,
    *,
    graph: Mapping[str, Any],
    constraints: Mapping[str, Any],
    segments: Mapping[str, Any],
    program: Mapping[str, Any] | None = None,
    contact_timeline: Mapping[str, Any] | None = None,
) -> bool:
    """Do not accept a generic success/alignment record as relation evidence."""
    if "available" in record and not _bool_rows(record["available"], num_envs):
        return False
    if predicate.arguments.get("task_type") == "E5":
        from .coordinated_placement_evidence import transport_record_matches

        if record.get("measurements", {}).get("scope") != "native_stable_support":
            return False
        if not transport_record_matches(
            record,
            predicate,
            roles,
            next(
                (
                    group
                    for group in graph["task_groups"]
                    if group["id"] == record.get("group_id")
                ),
                {},
            ),
            graph,
            {"segments": list(segments.values())},
            program,
            contact_timeline,
        ):
            return False
    if predicate.arguments.get("task_type") == "E4":
        try:
            metrics = record["measurements"]
            group = next(
                g for g in graph["task_groups"] if g["id"] == record["group_id"]
            )
            nodes = {n["id"]: n for n in graph["nodes"]}
            node = nodes[metrics["placement_node_id"]]
            _require_handover_receiver(group, nodes, predicate, record["uid"], node)
            if not _placement_predicate(predicate):
                return False
            references = roles.get(predicate.arguments.get("reference"), [])
            if len(references) != 1 or references[0].get("part_id") is not None:
                return False
            reference, relation = references[0]["uid"], predicate.arguments["relation"]
            raw_call = node["call"]
            if raw_call.get("kind") == "place":
                if (
                    raw_call.get(relation)
                    != f"{relation}__{reference}__{record['uid']}"
                ):
                    return False
            elif (
                raw_call.get("call_id")
                not in {
                    "simulation.place_relative",
                    "gen_sim.place_upright",
                    "gen_sim.stack_place",
                }
                or raw_call.get("arguments", {}).get("reference") != reference
                or raw_call.get("arguments", {}).get("relation") != relation
            ):
                return False
            if metrics.get("observation_kind") not in {"recorded", "final"}:
                return False
            boundary = (
                segments[group["node_ids"][-1]]
                if metrics["observation_kind"] == "recorded"
                else next(reversed(segments.values()))
            )
            if metrics.get("released") != measured_release(
                node, record["uid"], segments, boundary, num_envs
            ):
                return False
        except (KeyError, ValueError, TypeError, StopIteration, AttributeError):
            return False
    if record.get("measurements", {}).get("scope") == "native_stable_support":
        from .stable_support_evidence import stable_support_flags
        from .stability import StabilityConstraint
        from dataclasses import asdict

        try:
            metrics = record["measurements"]
            reference = roles[predicate.arguments["reference"]]
            configured = StabilityConstraint.decode(
                constraints["presets"][f"gen_sim.{metrics['placement_node_id']}.stable"]
            )
            if configured.kind == "supported_region":
                return False
            boundary = (
                segments[
                    next(
                        group["node_ids"][-1]
                        for group in graph["task_groups"]
                        if group["id"] == record["group_id"]
                    )
                ]
                if metrics["observation_kind"] == "recorded"
                else next(reversed(segments.values()))
            )
            if predicate.arguments.get("task_type") != "E5":
                source_node = next(
                    n for n in graph["nodes"] if n["id"] == metrics["placement_node_id"]
                )
                _, released_receipt, _ = measured_release(
                    source_node,
                    record["uid"],
                    segments,
                    boundary,
                    num_envs,
                    with_receipt=True,
                )
                if metrics["release_timestamp"] != released_receipt["timestamp"]:
                    return False
            return bool(
                num_envs == 1
                and predicate.arguments.get("relation") == "on"
                and reference == [{"uid": metrics["reference"], "part_id": None}]
                and metrics["entity"] == record["uid"]
                and metrics["reference_uid"] == metrics["reference"]
                and canonical_hash(metrics["constraint"])
                == canonical_hash(asdict(configured))
                and metrics["boundary_frames"] == boundary["end_steps"]
                and metrics["end_step"] == boundary["end_steps"][0]
                and record["accepted"]
                == [
                    a and b
                    for a, b in zip(
                        stable_support_flags(metrics, contact_timeline),
                        metrics["released"],
                        strict=True,
                    )
                ]
            )
        except (KeyError, ValueError, TypeError, StopIteration, AttributeError):
            return False
    if constraints.get("support_contract_revision") == 1 or any(
        cfg.get("support_contract_revision") == 1
        for cfg in constraints.get("presets", {}).values()
    ):
        return _strong_placement_matches(
            record,
            predicate,
            roles,
            num_envs,
            graph=graph,
            constraints=constraints,
            segments=segments,
        )
    metrics = record.get("measurements", {})
    references = roles.get(predicate.arguments.get("reference"), [])
    accepted = record.get("accepted", [])
    region_node = None
    region_cfg = None
    if graph and constraints:
        group = next(
            (g for g in graph["task_groups"] if g["id"] == record.get("group_id")),
            None,
        )
        candidates = (
            [
                node
                for node in graph["nodes"]
                if node["id"] in group["node_ids"]
                and _bare_table_node(node, predicate)
                and node["call"]["arguments"]["object"] == record.get("uid")
            ]
            if group is not None
            else []
        )
        if len(candidates) == 1:
            node = candidates[0]
            cfg = constraints.get("presets", {}).get(f"gen_sim.{node['id']}.stable", {})
            if cfg.get("kind") == "supported_region":
                region_node, region_cfg = node, cfg
    if metrics.get("scope") == "native_supported_landing_region":
        if region_node is None or segments is None:
            return False
        try:
            from .placement_evidence import _valid_pose

            uid = record["uid"]
            boundary_id = (
                group["node_ids"][-1]
                if metrics.get("observation_kind") == "recorded"
                else next(reversed(segments))
            )
            boundary = segments[boundary_id]
            native, landing = metrics["native_support"], metrics["landing_region"]
            pose = torch.tensor(native["object_pose"])
            reference_pose = torch.tensor(native["reference_pose"])
            _valid_pose(pose, num_envs)
            _valid_pose(reference_pose, num_envs)
            released = metrics["released"]
            physical = _region_flags(
                native,
                landing,
                pose,
                reference_pose,
                uid,
                metrics["support_gap_m"],
                num_envs,
            )
            if (
                len(references) != 1
                or references[0] != {"uid": "table", "part_id": None}
                or not any(
                    role.get("uid") == uid and role.get("part_id") is None
                    for role in roles.get(predicate.arguments.get("subject"), [])
                )
                or metrics.get("placement_node_id") != region_node["id"]
                or metrics.get("boundary_node_id") != boundary_id
                or metrics.get("observation_kind") not in {"recorded", "final"}
                or region_cfg.get("entity") != uid
                or region_cfg.get("reference") != "table"
                or region_cfg.get("displacement") is not None
                or region_cfg.get("target_position") is not None
                or metrics.get("reference_uid") != "table"
                or metrics.get("relation") != "on"
                or metrics.get("contact_tolerance_m") != 0.01
                or not all(_bool_rows(v, num_envs) for v in (accepted, released))
                or released
                != measured_release(region_node, uid, segments, boundary, num_envs)
                or metrics.get("release_window")
                != _release_window(region_node, segments)
            ):
                return False
            if metrics["observation_kind"] == "recorded":
                frames = metrics.get("boundary_frames")
                prefix = boundary["metadata"]["program_segment_source_path"]
                traces = [
                    policy
                    for policy in boundary["metadata"].get("post_policies", [])
                    if policy.get("kind") == "wait_stable"
                    and type(policy.get("policy_index")) is int
                    and policy.get("source_path")
                    == [*prefix, "post", policy["policy_index"]]
                    and _region_policy_matches(
                        policy,
                        region_cfg,
                        f"gen_sim.{region_node['id']}.stable",
                        num_envs,
                    )
                    and policy["result"].get("measurements", {}).get("native_support")
                    == native
                ]
                if (
                    frames != boundary.get("end_steps")
                    or not isinstance(frames, list)
                    or len(frames) != num_envs
                    or any(type(v) is not int or v <= 0 for v in frames)
                    or len(traces) != 1
                    or not _region_poses_match(
                        traces[0]["result"]["measurements"].get("landing_region", {}),
                        pose,
                        reference_pose,
                        num_envs,
                    )
                ):
                    return False
            return all(
                a == (p and r)
                for a, p, r in zip(accepted, physical, released, strict=True)
            )
        except (KeyError, ValueError, TypeError, RuntimeError, StopIteration):
            return False
    if metrics.get("scope") == "native_contact_support":
        if region_node is not None:
            return False
        supported, released = metrics.get("supported", []), metrics.get("released", [])
        return bool(
            predicate.arguments.get("relation") == "on"
            and len(references) == 1
            and metrics.get("reference_uid") == references[0]["uid"]
            and metrics.get("reference") == references[0]["uid"]
            and metrics.get("env_ids") == list(range(num_envs))
            and all(
                isinstance(values, list)
                and len(values) == num_envs
                and all(type(v) is bool for v in values)
                for values in (accepted, supported, released)
            )
            and all(
                a == (s and r)
                for a, s, r in zip(accepted, supported, released, strict=True)
            )
        )
    from .spatial_reference import SIDE_RELATIONS

    side_scope = metrics.get("scope") in {
        "bound_link_world_axes",
        "bound_rigid_world_axes",
    }
    support = metrics.get("tabletop", {}) if side_scope else metrics
    footprint = support.get("full_footprint_inside", [])
    released = metrics.get("released", [])
    gaps = support.get("support_gap_m", [])
    if (
        support.get("scope") != "measured_planar_landing_region"
        or len(references) != 1
        or metrics.get("reference_uid") != references[0]["uid"]
        or metrics.get("relation") != predicate.arguments.get("relation")
        or any(
            not isinstance(values, list) or len(values) != num_envs
            for values in (accepted, footprint, released, gaps)
        )
        or any(
            type(v) is not bool
            for values in (accepted, footprint, released)
            for v in values
        )
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in gaps)
    ):
        return False
    side = [True] * num_envs
    if side_scope:
        relation = predicate.arguments.get("relation")
        side_gaps = metrics.get("envelope_gaps_m", [])
        expected_axes = (
            2
            if relation
            in {"front_left_of", "front_right_of", "back_left_of", "back_right_of"}
            else 1
        )
        if (
            relation not in SIDE_RELATIONS
            or metrics.get("reference_part") != references[0]["part_id"]
            or (
                (
                    not isinstance(metrics.get("reference_link_id"), str)
                    or not metrics["reference_link_id"].startswith(
                        f"{references[0]['uid']}__"
                    )
                )
                if metrics["scope"] == "bound_link_world_axes"
                else (
                    references[0]["part_id"] is not None
                    or metrics.get("reference_link_id") is not None
                )
            )
            or not isinstance(side_gaps, list)
            or len(side_gaps) != expected_axes
            or any(
                not isinstance(values, list)
                or len(values) != num_envs
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) for v in values
                )
                for values in side_gaps
            )
        ):
            return False
        side = [all(axis[row] >= 0.0 for axis in side_gaps) for row in range(num_envs)]
    from .placement_evidence import (
        SUPPORT_TOLERANCE_M,
        PENETRATION_TOLERANCE_M,
        _support_gap_valid,
    )

    available = [True] * num_envs
    if (
        not side_scope
        and metrics.get("relation") == "inside"
        and metrics.get("footprint_kind") == "native_triangle_projection/v1"
    ):
        if not all(
            _bool_rows(metrics.get(key), num_envs)
            for key in ("reference_has_raised_geometry", "reference_aabb_overlap")
        ):
            return False
        available = _containment_available(metrics)
    return (
        support.get("support_tolerance_m") == SUPPORT_TOLERANCE_M
        and support.get("penetration_tolerance_m") == PENETRATION_TOLERANCE_M
        and record.get("available", [True] * num_envs) == available
        and all(
            a == (f and r and s and _support_gap_valid(g))
            for a, f, r, g, s in zip(
                accepted, footprint, released, gaps, side, strict=True
            )
        )
    )


def _articulation_milestone_matches(
    item: Mapping[str, Any],
    segment: Mapping[str, Any],
    *,
    uid: str,
    preset: str,
    item_index: int,
    env_id: int,
) -> bool:
    """Use the group's own measured joint/stability receipts, never final poses."""
    try:
        validators, posts = item.get("validators", []), item.get("post", [])
        if len(validators) != 1 or len(posts) != 1:
            return False
        declared, post = validators[0], posts[0]
        if (
            declared.get("kind") != "articulation_joint_position"
            or declared.get("articulation") != uid
            or post.get("kind") != "wait_stable"
            or post.get("entity") != uid
            or post.get("preset") != preset
        ):
            return False
        metadata = segment["metadata"]
        identities = metadata["runtime"]["env_ids"]
        row = identities.index(env_id)
        validation = metadata["validation"]
        if validation.get("env_ids") != identities:
            return False
        traces, policies = validation["validators"], metadata["post_policies"]
        if len(traces) != 1 or len(policies) != 1:
            return False
        trace, policy = traces[0], policies[0]
        result, stable = trace["result"], policy["result"]
        path = ["program", "items", item_index]
        if (
            trace.get("validator_index") != 0
            or trace.get("kind") != declared["kind"]
            or trace.get("source_path") != path + ["validators", 0]
            or result.get("source_path") != trace["source_path"]
            or result.get("kind") != declared["kind"]
            or result.get("articulation_id") != uid
            or result.get("joint") != declared["joint"]
            or result.get("env_ids") != identities
            or policy.get("policy_index") != 0
            or policy.get("kind") != "wait_stable"
            or policy.get("source_path") != path + ["post", 0]
            or stable.get("joint") != declared["joint"]
            or stable.get("evidence") != "joint_position_and_stability"
            or stable.get("stable_seconds") != 1.0
        ):
            return False

        def accepted(values: Any) -> bool:
            return (
                isinstance(values, list)
                and len(values) == len(identities)
                and all(type(value) is bool for value in values)
                and values[row] is True
            )

        if not all(
            accepted(values)
            for values in (
                validation.get("accepted_mask"),
                trace.get("result_mask"),
                result.get("accepted_mask"),
                policy.get("result_mask"),
                stable.get("accepted"),
            )
        ):
            return False
        positions, observed = result["joint_position"], stable["observed"]
        if len(positions) != len(identities) or len(observed) != len(identities):
            return False
        minimum, maximum = declared["minimum_position"], declared["maximum_position"]
        target, tolerance = stable["target"], stable["tolerance"]
        if not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in (
                minimum,
                maximum,
                target,
                tolerance,
                positions[row],
                observed[row],
            )
        ):
            return False
        return (
            tolerance >= 0
            and minimum == result.get("minimum_position") == target - tolerance
            and maximum == result.get("maximum_position") == target + tolerance
            and minimum <= positions[row] <= maximum
            and abs(observed[row] - target) <= tolerance
        )
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return False


def _transport_hold_matches(
    record, predicate, roles, group, graph, runtime, program, timeline
):
    from .coordinated_placement_evidence import transport_record_matches

    return transport_record_matches(
        record,
        predicate,
        roles,
        group,
        graph,
        runtime,
        program,
        timeline,
        acquisition=True,
    )


def _ordered_coordinated_placement(
    acquisition: Mapping[str, Any] | None,
    placement: Mapping[str, Any] | None,
    terminal: Mapping[str, Any],
    *,
    env_id: int,
) -> bool:
    """Place an E5 acquisition before its independently measured release/support."""
    if not isinstance(acquisition, Mapping) or not isinstance(placement, Mapping):
        return False
    acquired = acquisition.get("measurements")
    placed = placement.get("measurements")
    if not isinstance(acquired, Mapping) or not isinstance(placed, Mapping):
        return False
    shared = (
        "invocation_id",
        "invocation_revision",
        "acquired_step",
        "release_step",
        "contact_timeline_digest",
        "release_receipt_digest",
    )
    steps = (
        acquired.get("acquired_step"),
        acquired.get("release_step"),
        placed.get("start_step"),
        placed.get("end_step"),
    )
    ends = terminal.get("end_steps")
    return (
        acquired.get("scope") == "measured_coordinated_transport/v1"
        and placed.get("scope") == "native_stable_support"
        and all(acquired.get(key) == placed.get(key) for key in shared)
        and all(type(step) is int for step in steps)
        and 0 <= steps[0] < steps[1] <= steps[2] <= steps[3]
        and type(ends) is list
        and env_id < len(ends)
        and type(ends[env_id]) is int
        and placed.get("boundary_frames") == ends
        and steps[3] == ends[env_id]
    )


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
    placement_milestones: Mapping[str, Any] | None = None,
    control_milestones: Mapping[str, Any] | None = None,
    handover_milestones: Mapping[str, Any] | None = None,
    coordinated_milestones: Mapping[str, Any] | None = None,
    pour_milestones: Mapping[str, Any] | None = None,
    legacy_observer: bool = False,
    contact_timeline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate only newly collected final measurements and exact segment evidence."""
    from .pour_evidence import pour_record_matches

    if legacy_observer:
        pour_milestones = None

    if contract["task_spec_digest"] != canonical_hash(spec.to_dict()) or contract[
        "graph_digest"
    ] != canonical_hash(graph):
        raise ValueError("TaskSpec runtime evidence identity mismatch.")
    if canonical_hash(program) != contract["program_digest"]:
        raise ValueError("TaskSpec runtime program identity mismatch.")
    if canonical_hash(constraints) != contract["constraints_digest"]:
        raise ValueError("TaskSpec runtime constraints identity mismatch.")
    program_items = {v["name"]: v for v in program["program"]["items"]}
    program_indices = {
        v["name"]: index for index, v in enumerate(program["program"]["items"])
    }
    nodes = {v["id"]: v for v in graph["nodes"]}
    members = {
        v["group_id"]: v
        for entries in contract["coverage"]["lineage"].values()
        for v in entries
    }
    groups = {v["id"]: v for v in graph["task_groups"]}
    milestone_by_id = {milestone.id: milestone for milestone in spec.milestones}
    milestone_groups = {
        record["id"]: tuple(record["groups"])
        for record in contract["coverage"]["milestones"]
    }
    ordered_e5_acquisition: dict[tuple[str, str], tuple[str, ...]] = {}
    for milestone in spec.milestones:
        if len(milestone.after) != 1 or not any(
            _placement_predicate(predicate)
            and predicate.arguments.get("task_type") == "E5"
            for predicate in milestone.achieve
        ):
            continue
        predecessor = milestone_by_id.get(milestone.after[0])
        group_ids = milestone_groups.get(milestone.id, ())
        if (
            predecessor is not None
            and len(group_ids) == 1
            and group_ids == milestone_groups.get(predecessor.id, ())
            and any(
                predicate.name == "held_by_both_grippers"
                and predicate.arguments.get("task_type") == "E5"
                for predicate in predecessor.achieve
            )
        ):
            ordered_e5_acquisition[(group_ids[0], predecessor.id)] = tuple(
                predicate.observation or predicate.id for predicate in milestone.achieve
            )
    segments = {}
    e4_placement_milestones = {
        m.id
        for m in spec.milestones
        if any(
            _placement_predicate(p) and p.arguments.get("task_type") == "E4"
            for p in m.achieve
        )
    }
    e4_placement_groups = {
        group_id
        for coverage in contract["coverage"]["milestones"]
        if coverage["id"] in e4_placement_milestones
        for group_id in coverage["groups"]
    }
    transfer_prefixes = {}
    for group_id in e4_placement_groups:
        for index, key in enumerate(groups[group_id]["node_ids"]):
            if nodes[key]["call"].get("kind") == "hand_over":
                transfer_prefixes[group_id] = groups[group_id]["node_ids"][: index + 1]
                break
    handover_matches = None
    if any(g["task_type"] == "E4" for g in groups.values()):
        from .handover_evidence import handover_record_matches

        handover_matches = handover_record_matches
    coordinated_matches = None
    if any(g["task_type"] == "E5" for g in groups.values()):
        from .coordinated_evidence import coordinated_record_matches

        coordinated_matches = coordinated_record_matches
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
                    and (
                        predicate.arguments.get("task_type") != "E4"
                        or predicate.name
                        not in {"held", "orientation", "handover_complete"}
                        or (
                            handover_matches is not None
                            and handover_matches(
                                v,
                                predicate.name,
                                groups[v["group_id"]],
                                graph,
                                runtime,
                                program,
                                constraints,
                                num_envs,
                                final=True,
                            )
                        )
                    )
                    and (
                        predicate.arguments.get("task_type") != "E5"
                        or predicate.name != "held_by_both_grippers"
                        or (
                            coordinated_matches is not None
                            and coordinated_matches(
                                v,
                                groups[v["group_id"]],
                                graph,
                                runtime,
                                program,
                                constraints,
                                num_envs,
                                final=True,
                            )
                        )
                    )
                    and (
                        predicate.name != "pour_motion"
                        or pour_record_matches(
                            v,
                            predicate,
                            contract["coverage"]["roles"],
                            groups[v["group_id"]],
                            graph,
                            runtime,
                            program,
                            num_envs,
                        )
                    )
                    and (
                        not _placement_predicate(predicate)
                        or _placement_measurement_matches(
                            v,
                            predicate,
                            contract["coverage"]["roles"],
                            num_envs,
                            graph=graph,
                            constraints=constraints,
                            segments=segments,
                            program=program,
                            contact_timeline=contact_timeline,
                        )
                        and (v.get("available") is None or v["available"][env_id])
                    )
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
        measured_placements: dict[str, Any] = {}
        measured_controls: dict[str, Any] = {}
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
                in {
                    "upright",
                    "placement",
                    "stack",
                    "supported_placement",
                    "supported_region",
                }
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
                if (
                    group["task_type"] in {"E8", "E9"}
                    and node["id"] == group["node_ids"][-1]
                ):
                    for record in (control_milestones or {}).values():
                        metrics = record.get("measurements", {})
                        if (
                            record.get("group_id") == group["id"]
                            and metrics.get("scope") == "measured_control_operation"
                            and metrics.get("boundary_frames")
                            == segment.get("end_steps")
                        ):
                            measured_controls[group["id"]] = record
                if placement_milestones is not None:
                    record = placement_milestones.get(group["id"], {})
                    accepted = record.get("accepted", [])
                    metrics = record.get("measurements", {})
                    expected = members[group["id"]]
                    if (
                        node["id"] == group["node_ids"][-1]
                        and record.get("group_id") == group["id"]
                        and record.get("uid") == expected["object_uid"]
                        and record.get("part_id") == expected["object_part_id"]
                        and isinstance(accepted, list)
                        and len(accepted) == num_envs
                        and all(type(v) is bool for v in accepted)
                        and metrics.get("scope")
                        in {
                            "measured_planar_landing_region",
                            "bound_link_world_axes",
                            "bound_rigid_world_axes",
                            "native_contact_support",
                            "native_stable_support",
                            "measured_coordinated_transport/v1",
                            "native_supported_landing_region",
                            "native_supported_placement",
                        }
                        and metrics.get("boundary_frames") == segment.get("end_steps")
                    ):
                        measured_placements[group["id"]] = record
            terminal_boundary = node["id"] == group["node_ids"][-1]
            transfer_boundary = (
                group["id"] in e4_placement_groups
                and node["call"].get("kind") == "hand_over"
            )
            if not terminal_boundary and not transfer_boundary:
                continue
            if terminal_boundary:
                complete_groups[group["id"]] = (
                    measured_groups.get(group["id"], set())
                    if set(group["node_ids"]).issubset(successful_nodes)
                    else set()
                )
            snapshot = {}
            acquisition_snapshot = {}
            ordered_main_keys: set[str] = set()
            for milestone in spec.milestones:
                record = next(
                    (
                        v
                        for v in contract["coverage"]["milestones"]
                        if v["id"] == milestone.id
                    ),
                    None,
                )
                if record and record["groups"]:
                    for predicate in milestone.achieve:
                        if _needs_universe(spec, predicate):
                            continue
                        if not terminal_boundary and (
                            predicate.arguments.get("task_type") != "E4"
                            or predicate.name != "handover_complete"
                            or group["id"] not in record["groups"]
                        ):
                            continue
                        if predicate.arguments.get(
                            "task_type"
                        ) == "E4" and predicate.name in {
                            "handover_complete",
                            "held",
                            "orientation",
                        }:
                            observations = [
                                (handover_milestones or {})
                                .get(id, {})
                                .get(predicate.name, {})
                                for id in record["groups"]
                            ]
                            if observations and all(
                                set(
                                    transfer_prefixes[id]
                                    if predicate.name == "handover_complete"
                                    and id in transfer_prefixes
                                    else groups[id]["node_ids"]
                                ).issubset(successful_nodes)
                                and value.get("group_id") == id
                                and value.get("uid") == members[id]["object_uid"]
                                and value.get("part_id")
                                == members[id]["object_part_id"]
                                and _bool_rows(value.get("accepted"), num_envs)
                                and isinstance(value.get("measurements"), Mapping)
                                and bool(value["measurements"])
                                and handover_matches is not None
                                and handover_matches(
                                    value,
                                    predicate.name,
                                    groups[id],
                                    graph,
                                    runtime,
                                    program,
                                    constraints,
                                    num_envs,
                                )
                                for id, value in zip(
                                    record["groups"], observations, strict=True
                                )
                            ):
                                snapshot[predicate.observation or predicate.id] = all(
                                    value["accepted"][env_id] for value in observations
                                )
                            continue
                        if (
                            predicate.arguments.get("task_type") == "E5"
                            and predicate.name == "held_by_both_grippers"
                        ):
                            observations = [
                                (coordinated_milestones or {}).get(id, {})
                                for id in record["groups"]
                            ]
                            if observations and all(
                                set(groups[id]["node_ids"]).issubset(successful_nodes)
                                and value.get("group_id") == id
                                and value.get("uid") == members[id]["object_uid"]
                                and value.get("part_id")
                                == members[id]["object_part_id"]
                                and _bool_rows(value.get("accepted"), num_envs)
                                and isinstance(value.get("measurements"), Mapping)
                                and bool(value["measurements"])
                                and coordinated_matches is not None
                                and (
                                    _transport_hold_matches(
                                        value,
                                        predicate,
                                        contract["coverage"]["roles"],
                                        groups[id],
                                        graph,
                                        runtime,
                                        program,
                                        contact_timeline,
                                    )
                                    if value.get("measurements", {}).get("scope")
                                    == "measured_coordinated_transport/v1"
                                    else coordinated_matches(
                                        value,
                                        groups[id],
                                        graph,
                                        runtime,
                                        program,
                                        constraints,
                                        num_envs,
                                    )
                                )
                                for id, value in zip(
                                    record["groups"], observations, strict=True
                                )
                            ):
                                observation = predicate.observation or predicate.id
                                accepted = all(
                                    value["accepted"][env_id] for value in observations
                                )
                                ordered = ordered_e5_acquisition.get(
                                    (group["id"], milestone.id)
                                )
                                if ordered is None:
                                    snapshot[observation] = accepted
                                else:
                                    acquisition_snapshot[observation] = accepted
                                    ordered_main_keys.update(ordered)
                            continue
                        if (
                            predicate.name == "articulation_joint_near"
                            and predicate.arguments.get("task_type") == "E6"
                        ):
                            verified = []
                            for group_id in record["groups"]:
                                articulation_group = groups[group_id]
                                member = members[group_id]
                                call = nodes[articulation_group["node_ids"][0]]["call"]
                                arguments = call.get("arguments", {})
                                uid, part = (
                                    member["object_uid"],
                                    member["object_part_id"],
                                )
                                state = predicate.arguments["target_state"]
                                suffix = f".{part}" if part else ""
                                terminal = articulation_group["node_ids"][-1]
                                verified.append(
                                    articulation_group["task_type"] == "E6"
                                    and set(articulation_group["node_ids"]).issubset(
                                        successful_nodes
                                    )
                                    and call.get("call_id")
                                    == "gen_sim.articulation_slide"
                                    and arguments.get("object") == uid
                                    and arguments.get("part") == part
                                    and arguments.get("state") == state
                                    and _articulation_milestone_matches(
                                        program_items[terminal],
                                        segments[terminal],
                                        uid=uid,
                                        preset=f"gen_sim.articulation.{uid}{suffix}.{state}",
                                        item_index=program_indices[terminal],
                                        env_id=env_id,
                                    )
                                )
                            if verified and all(verified):
                                snapshot[predicate.observation or predicate.id] = True
                            continue
                        if predicate.arguments.get("task_type") in {"E8", "E9"}:
                            records = [
                                measured_controls.get(id, {}) for id in record["groups"]
                            ]
                            if all(
                                set(groups[id]["node_ids"]).issubset(successful_nodes)
                                and value.get("uid") == members[id]["object_uid"]
                                and value.get("accepted") == [True] * num_envs
                                for id, value in zip(
                                    record["groups"], records, strict=True
                                )
                            ):
                                snapshot[predicate.observation or predicate.id] = True
                            continue
                        if predicate.name == "pour_motion":
                            records = [
                                (pour_milestones or {}).get(id, {})
                                for id in record["groups"]
                            ]
                            if records and all(
                                set(groups[id]["node_ids"]).issubset(successful_nodes)
                                and pour_record_matches(
                                    value,
                                    predicate,
                                    contract["coverage"]["roles"],
                                    groups[id],
                                    graph,
                                    runtime,
                                    program,
                                    num_envs,
                                )
                                for id, value in zip(
                                    record["groups"], records, strict=True
                                )
                            ):
                                snapshot[predicate.observation or predicate.id] = all(
                                    value["accepted"][env_id] for value in records
                                )
                            continue
                        if _placement_predicate(predicate):
                            records = [
                                measured_placements.get(id, {})
                                for id in record["groups"]
                            ]
                            if all(
                                set(groups[id]["node_ids"]).issubset(successful_nodes)
                                and _placement_measurement_matches(
                                    value,
                                    predicate,
                                    contract["coverage"]["roles"],
                                    num_envs,
                                    graph=graph,
                                    constraints=constraints,
                                    segments=segments,
                                    program=program,
                                    contact_timeline=contact_timeline,
                                )
                                and (
                                    value.get("available") is None
                                    or value["available"][env_id]
                                )
                                for id, value in zip(
                                    record["groups"], records, strict=True
                                )
                            ):
                                snapshot[predicate.observation or predicate.id] = all(
                                    value["accepted"][env_id] for value in records
                                )
                            continue
                        acceptable = (
                            {"upright"}
                            if predicate.name == "object_upright"
                            else (
                                {
                                    "placement",
                                    "stack",
                                    "supported_placement",
                                }
                                if predicate.name == "semantic_goal"
                                else set()
                            )
                        )
                        if acceptable and all(
                            complete_groups.get(id, set()).intersection(acceptable)
                            for id in record["groups"]
                        ):
                            snapshot[predicate.observation or predicate.id] = True
            if acquisition_snapshot:
                if (
                    all(value is True for value in acquisition_snapshot.values())
                    and ordered_main_keys
                    and all(snapshot.get(key) is True for key in ordered_main_keys)
                    and _ordered_coordinated_placement(
                        (coordinated_milestones or {}).get(group["id"]),
                        measured_placements.get(group["id"]),
                        segment,
                        env_id=env_id,
                    )
                ):
                    trajectory.append(acquisition_snapshot)
                else:
                    snapshot.update(acquisition_snapshot)
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
        "observer_revision": (
            10
            if any(
                member.get("measurements", {}).get("footprint_kind")
                == "native_triangle_projection/v1"
                for goal in final_goals.values()
                for member in goal.get("members", [])
            )
            else (
                8
                if pour_milestones
                or any(
                    value.get("measurements", {}).get("scope")
                    == "bound_rigid_world_axes"
                    for value in (placement_milestones or {}).values()
                )
                else (
                    7
                    if e4_placement_groups
                    else (
                        6
                        if any(g["task_type"] == "E5" for g in groups.values())
                        else (
                            5
                            if any(g["task_type"] == "E4" for g in groups.values())
                            else 4
                        )
                    )
                )
            )
        ),
        "run_id": run_id,
        "task_spec_digest": contract["task_spec_digest"],
        "task_spec_semantic_hash": spec.semantic_hash,
        "graph_digest": contract["graph_digest"],
        "program_digest": contract["program_digest"],
        "constraints_digest": contract["constraints_digest"],
        "environments": rows,
        "placement_milestone_measurements": dict(placement_milestones or {}),
        **(
            {"pour_motion_measurements": dict(pour_milestones)}
            if pour_milestones
            else {}
        ),
        **(
            {"coordinated_milestone_measurements": dict(coordinated_milestones)}
            if coordinated_milestones is not None
            else {}
        ),
        **(
            {"handover_milestone_measurements": dict(handover_milestones)}
            if handover_milestones is not None
            else {}
        ),
        **(
            {"control_operation_measurements": dict(control_milestones)}
            if control_milestones is not None
            else {}
        ),
    }
