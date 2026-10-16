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

"""Call-local pouring geometry shared by Pick look-ahead and actual Pour."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.primitives.pour import PourOptions
from embodichain.lab.task_program.semantics import Pick, Place, RegisteredSemanticCall
from embodichain.utils.math import pose_inv

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class PourContinuation:
    pick_id: str
    stage_id: str
    pour_id: str
    object_id: str
    receiver_id: str
    resource: str
    return_pose: torch.Tensor
    clearance_m: float
    angle_rad: float
    sample_count: int


def supports_ordinary_pour(
    graph: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    """Select only independent, unconstrained E3 recipes before opt-in.

    Inputs are validated planner outputs. This capability check owns the recipe
    shape, not grounding: :func:`bind_calls` still validates exact typed fields.
    Already-held continuations stay on their existing legacy route.
    """
    pours = [step for step in candidate["draft"]["steps"] if step["task_type"] == "E3"]
    groups = [group for group in graph["task_groups"] if group["task_type"] == "E3"]
    if (
        not pours
        or not groups
        or any(
            step.get(field, "none") != "none"
            for step in pours
            for field in ("orientation_goal", "axis", "direction")
        )
    ):
        return False
    nodes = {node["id"]: node for node in graph["nodes"]}
    positions = {node["id"]: index for index, node in enumerate(graph["nodes"])}
    expected = (
        "pick",
        "simulation.move_held_object",
        "simulation.pour",
        "simulation.move_held_object",
        "place",
    )
    for group in groups:
        selected = [nodes[identifier] for identifier in group["node_ids"]]
        while selected and selected[-1]["call"].get("call_id") == "simulation.park":
            selected.pop()
        calls = [node["call"] for node in selected]
        if (
            len(calls) != 5
            or tuple(call.get("call_id", call["kind"]) for call in calls) != expected
            or positions[selected[-1]["id"]] - positions[selected[0]["id"]] != 4
        ):
            return False
        resources = calls[0].get("resources", {})
        subject = calls[0]["object"]
        if (
            set(resources) != {"primary"}
            or any(call.get("resources", {}) != resources for call in calls[1:])
            or any(call["arguments"].get("object") != subject for call in calls[1:4])
            or calls[4]["object"] != subject
            or not calls[1]["arguments"].get("reference")
            or "reference" in calls[3]["arguments"]
            or calls[4].get("at") is None
        ):
            return False
    return True


def bind_calls(
    program: Any, preset: Any, factory: Any, motion_samples: tuple[tuple[str, int], ...]
) -> tuple[PourContinuation, ...]:
    """Require the complete five-role recipe before replacing legacy targets."""
    from .services import _AbsolutePoseTarget

    segments = tuple(program.iter_segments())
    routes = {route.target_id: route for route in factory.routes}
    stages = {route.target_id: route for route in factory.staging_routes}
    budgets = dict(motion_samples)
    result = []
    for index, segment in enumerate(segments):
        if len(segment.calls) != 1:
            continue
        pick = segment.calls[0].call
        if type(pick) is not Pick or index + 4 >= len(segments):
            continue
        selected = segments[index : index + 5]
        if any(len(s.calls) != 1 or s.parallel_block is not None for s in selected):
            continue
        stage, pour, returned, place = (s.calls[0].call for s in selected[1:])
        if (
            type(stage) is not RegisteredSemanticCall
            or stage.call_id != "simulation.move_held_object"
            or stage.arguments.get("target") not in stages
        ):
            continue
        declaration = stages[stage.arguments["target"]]
        object_id = pick.object.entity_id
        if (
            stage.arguments
            != {
                "object": object_id,
                "target": declaration.target_id,
                "reference": declaration.receiver_id,
            }
            or declaration.object_id != object_id
            or type(pour) is not RegisteredSemanticCall
            or pour.call_id != "simulation.pour"
            or dict(pour.arguments) != {"object": object_id}
            or type(returned) is not RegisteredSemanticCall
            or returned.call_id != "simulation.move_held_object"
            or set(returned.arguments) != {"object", "target"}
            or returned.arguments["object"] != object_id
            or type(place) is not Place
            or place.object.entity_id != object_id
            or place.at is None
            or set(pick.resources) != {"primary"}
            or any(
                dict(c.resources) != dict(pick.resources)
                for c in (stage, pour, returned, place)
            )
        ):
            raise ValueError(
                "Canonical Pour requires its exact same-object five-call recipe."
            )
        route = routes.get(returned.arguments["target"])
        if (
            route is None
            or route.object_id != object_id
            or type(route.pose) is not _AbsolutePoseTarget
        ):
            raise ValueError(
                "Canonical Pour requires the declared absolute return pose."
            )
        from embodichain.lab.task_program.semantics import SemanticPose

        return_pose = SemanticPose(
            route.pose.position, route.pose.quaternion_xyzw
        ).to_matrix()
        # Declaration and compiler snapshots normalize through float64/float32.
        if not torch.allclose(
            return_pose,
            place.at.to_matrix(),
            rtol=0,
            atol=8 * torch.finfo(torch.float32).eps,
        ):
            raise ValueError(
                "Canonical Pour transport and Place return poses disagree."
            )

        def identity(s: Any) -> str:
            return (
                f"{program.program_id}/{s.segment_id}:{s.calls[0].segment_call_index}"
            )

        options = preset.action_option_template("simulation.pour")
        if type(options) is not PourOptions:
            raise TypeError("Canonical Pour requires exact PourOptions.")
        pour_id = identity(selected[2])
        result.append(
            PourContinuation(
                identity(segment),
                identity(selected[1]),
                pour_id,
                object_id,
                declaration.receiver_id,
                pick.resources["primary"],
                return_pose,
                declaration.clearance_m,
                float(options.rotate_angle),
                budgets[pour_id],
            )
        )
    if {r.stage_id for r in result} != {
        f"{program.program_id}/{s.segment_id}:{s.calls[0].segment_call_index}"
        for s in segments
        if len(s.calls) == 1
        and type(s.calls[0].call) is RegisteredSemanticCall
        and s.calls[0].call.call_id == "simulation.move_held_object"
        and s.calls[0].call.arguments.get("target") in stages
    }:
        raise ValueError(
            "A canonical Pour staging call lacks a bound Pick continuation."
        )
    return tuple(result)


def pad_line_world(robot: Any, motion: JointPositionTarget) -> torch.Tensor:
    """Use the same ordered native pad frames as the fixed GenSim embodiment."""
    prefix = motion.control_part.split("_arm")[0]
    pads = {
        "left": ("left_inner_finger_pad", "left_right_inner_finger_pad"),
        "right": ("right_left_inner_finger_pad", "right_inner_finger_pad"),
    }.get(prefix)
    if pads is None or not set(pads) <= set(robot.link_names):
        raise ValueError("Canonical Pour requires both declared embodiment pad frames.")
    poses = [robot.get_link_pose(name, to_matrix=True) for name in pads]
    return poses[1][:, :3, 3] - poses[0][:, :3, 3]


def observed_pose(context: Any, entity_id: str) -> torch.Tensor:
    entity = context.scene.entities.get(entity_id)
    if entity is None or entity.pose is None or context.batch_size != 1:
        raise ValueError("Canonical Pour requires an observed, exact B1 rigid entity.")
    pose = entity.pose
    return pose[None] if pose.ndim == 2 else pose


@dataclass
class CandidatePreview:
    route: PourContinuation
    source_vertices: np.ndarray
    receiver_vertices: np.ndarray
    staging: np.ndarray
    receiver_pose: np.ndarray
    pad_line_tcp: torch.Tensor
    gravity_world: tuple[float, float, float]

    def valid_mask(self, object_to_eef: torch.Tensor) -> torch.Tensor:
        matrices = object_to_eef.detach().cpu().double().numpy()
        jaw = matrices[..., :3, :3] @ self.pad_line_tcp.detach().cpu().double().numpy()
        norm = np.linalg.norm(jaw, axis=-1, keepdims=True)
        up = self.staging[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        up /= np.linalg.norm(up)
        unit = jaw / np.maximum(norm, 1e-9)
        projection = unit - (unit * up).sum(-1, keepdims=True) * up
        valid = (
            np.isfinite(projection).all(-1)
            & (norm[..., 0] > 1e-9)
            & (np.linalg.norm(projection, axis=-1) > 1e-9)
        )
        return torch.as_tensor(valid, device=object_to_eef.device)

    def targets(self, object_to_eef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from .pour_geometry import pour_frames

        valid = self.valid_mask(object_to_eef)
        matrices = object_to_eef.detach().cpu().numpy()
        targets = np.broadcast_to(self.staging, (5, *matrices.shape)).copy()
        line = self.pad_line_tcp.detach().cpu().double().numpy()
        for index in np.ndindex(matrices.shape[:-2]):
            if bool(valid[index]):
                frames = pour_frames(
                    self.source_vertices,
                    self.receiver_vertices,
                    self.staging,
                    self.receiver_pose,
                    matrices[index][:3, :3] @ line,
                    self.route.angle_rad,
                    self.route.return_pose.detach().cpu().numpy(),
                    sample_count=self.route.sample_count,
                    gravity_world=self.gravity_world,
                )
                targets[(slice(None), *index)] = frames["object_targets"]
        objects = torch.as_tensor(
            targets, device=object_to_eef.device, dtype=object_to_eef.dtype
        )
        return objects @ object_to_eef[None], valid


def prepare_preview(
    action: Any, request: Any, context: Any, route: PourContinuation
) -> CandidatePreview:
    from .pour_geometry import staging_pose

    if request.goal.semantics.entity_id != route.object_id:
        raise ValueError("Canonical Pour Pick identity differs from its continuation.")
    # Compiler retains both ordinary return targets; replace all five roles,
    # never just remove the candidate-dependent peak from qualification.
    if len(request.skill_options.downstream_object_target_poses) != 2:
        raise ValueError("Canonical Pour look-ahead has an unrecognized suffix.")
    vertices = action.pour_scene_vertices
    source = vertices[route.object_id][0].detach().cpu().numpy()
    receiver = vertices[route.receiver_id][0].detach().cpu().numpy()
    source_pose = observed_pose(context, route.object_id)[0].detach().cpu().numpy()
    receiver_pose = observed_pose(context, route.receiver_id)[0].detach().cpu().numpy()
    staging = staging_pose(
        source,
        receiver,
        source_pose,
        receiver_pose,
        clearance_m=route.clearance_m,
        gravity_world=action.pour_gravity_world,
    )
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    tcp = action.robot.compute_fk(
        qpos=context.robot.qpos[:, list(motion.joint_ids)],
        name=motion.control_part,
        to_matrix=True,
    )
    jaw = pad_line_world(action.robot, motion)
    local = (tcp[:, :3, :3].transpose(1, 2) @ jaw[..., None])[0, :, 0]
    return CandidatePreview(
        route,
        source,
        receiver,
        staging,
        receiver_pose,
        local,
        action.pour_gravity_world,
    )
