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

"""Explicit ordinary rigid manipulation using measured hand and occupied floor geometry."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
import math
import time
from types import SimpleNamespace
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

from embodichain.compute.trajectory import differentiate_positions
from embodichain.lab.sim.atomic_actions.control import JointPositionCommand
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose, resolve_pose_goal
from embodichain.lab.sim.atomic_actions.trajectory_ops import interpolate_hand_qpos
from embodichain.lab.task_program.semantics import Pick, Place, SceneAffordanceRef

from ._search_cost import PlanningBudgetExceeded, measure_search
from .alignment_continuation import _PREVIEW
from .motion import _joint_velocity_limits
from .pick_contact import _native_meshes, build_closing_contact_screen
from .pick_safety import _inspect_pick_motion, _inspect_pick_plan
from .planning_probe import _validate_final_plan_velocity, initial_plan_capture

__all__: list[str] = []

MEASURED_RIGID_REVISION = 1
_CAPTURE: ContextVar[str | None] = ContextVar("gen_sim_measured_rigid", default=None)
_MARGIN = 0.005


def declarations(value: object) -> tuple[str, ...]:
    """Require explicit, unique generated call names."""
    if (
        type(value) not in (list, tuple)
        or any(
            type(name) is not str or not name or name != name.strip() for name in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError("Measured rigid calls require unique exact segment names.")
    return tuple(value)


def bind_calls(value: object, program: Any, purposes: dict[str, str]) -> tuple:
    """Bind only ordinary rigid Pick or inside Place occurrences, never task IDs."""
    remaining = set(declarations(value))
    result = []
    for segment in program.iter_segments():
        if segment.name not in remaining:
            continue
        if len(segment.calls) != 1:
            raise ValueError("Measured rigid policy requires one exact semantic call.")
        analyzed = segment.calls[0]
        call = analyzed.call
        identity = (
            f"{program.program_id}/{segment.segment_id}:{analyzed.segment_call_index}"
        )
        if type(call) is Pick:
            if purposes.get(identity) != "ordinary":
                raise ValueError("Measured preshape requires an ordinary Pick.")
            container = ""
        elif type(call) is Place and type(call.inside) is SceneAffordanceRef:
            container = call.inside.entity_id
        else:
            raise ValueError(
                "Measured rigid calls require ordinary Pick or inside Place."
            )
        result.append((identity, call.object.entity_id, container))
        remaining.remove(segment.name)
    if remaining:
        raise ValueError("Measured rigid policy references missing calls.")
    return tuple(result)


def defer_initial_capture(identity: str) -> bool:
    return _CAPTURE.get() == identity


def _transform(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return vertices @ pose[:3, :3].T + pose[:3, 3]


class _Scene:
    """One immutable native arena snapshot; all pair tests retain actual rest offsets."""

    def __init__(self, simulation: Any, robot: Any) -> None:
        spawn = simulation.spawn_result
        handles = [spawn.get(path) for path in spawn.get_paths()]
        robots = [h for h in handles if h.get_name() == robot.uid]
        if len(robots) != 1:
            raise ValueError("Measured native robot identity is ambiguous.")
        self.native = robots[0]
        self.obstacles: list[dict[str, Any]] = []

        def add(name: str, body: Any, pose: np.ndarray) -> None:
            meshes = _native_meshes(body)
            if not meshes:
                return
            rest = float(body.get_rest_offset())
            if not math.isfinite(rest) or rest < 0 or not np.isfinite(pose).all():
                raise ValueError("Native rest offsets and body poses must be finite.")
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(meshes):
                manager.add_object(str(index), mesh, transform=pose)
            points = np.vstack([_transform(m.vertices, pose) for m in meshes])
            self.obstacles.append(
                dict(
                    name=name,
                    meshes=meshes,
                    rest=rest,
                    pose=pose,
                    manager=manager,
                    bounds=np.stack((points.min(0), points.max(0))),
                )
            )

        for uid in simulation.get_rigid_object_uid_list():
            matches = [
                h
                for h in handles
                if h.get_name() == uid and h.arena_name == self.native.arena_name
            ]
            if len(matches) != 1:
                raise ValueError("Native rigid identity is unavailable or ambiguous.")
            pose = simulation.get_rigid_object(uid).get_local_pose(to_matrix=True)[0]
            add(
                uid,
                matches[0].native().get_physical_body(),
                pose.detach().cpu().double().numpy(),
            )
        for uid in simulation.get_articulation_uid_list():
            matches = [
                h
                for h in handles
                if h.get_name() == uid and h.arena_name == self.native.arena_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Native articulation identity is unavailable or ambiguous."
                )
            native = matches[0]
            for link in native.get_link_names():
                add(
                    uid + ":" + link,
                    native.render_articulation.get_physical_body(link),
                    np.asarray(native.get_link_pose(link), dtype=np.float64),
                )

    def inspect(
        self,
        parts: Any,
        poses: np.ndarray,
        source: str,
        *,
        source_contact_from: int = 0,
    ) -> dict[str, Any]:
        managers, vertices = [], []
        for part in parts:
            if (
                not math.isfinite(part.rest_offset)
                or part.rest_offset < 0
                or not part.meshes
            ):
                raise ValueError(
                    "Native moving parts require finite rest offsets and enabled shapes."
                )
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            managers.append(manager)
            vertices.append(np.vstack([m.vertices for m in part.meshes]))
        if (
            not len(parts)
            or poses.shape != (len(poses), len(parts), 4, 4)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("Native clearance requires complete finite part poses.")
        queries = 0
        for frame in range(len(poses)):
            for index, part in enumerate(parts):
                pose = poses[frame, index]
                cloud = _transform(vertices[index], pose)
                low, high = cloud.min(0), cloud.max(0)
                manager = managers[index]
                for key in range(len(part.meshes)):
                    manager.set_transform(str(key), pose)
                for obstacle in self.obstacles:
                    if obstacle["name"] == source and frame >= source_contact_from:
                        continue
                    threshold = part.rest_offset + obstacle["rest"]
                    gap = np.maximum(
                        0,
                        np.maximum(
                            obstacle["bounds"][0] - high, low - obstacle["bounds"][1]
                        ),
                    )
                    if np.linalg.norm(gap) > threshold + 0.002:
                        continue
                    distance = float(manager.min_distance_other(obstacle["manager"]))
                    queries += 1
                    if not math.isfinite(distance) or distance <= threshold:
                        return dict(
                            valid=False,
                            frame=frame,
                            link=part.name,
                            obstacle=obstacle["name"],
                            distance_m=distance,
                            required_clearance_m=threshold,
                            queries=queries,
                        )
        return dict(
            valid=True,
            frames=len(poses),
            queries=queries,
            scope="discrete_native_pair_rest_not_continuous_contact_or_task_success",
        )

    def arm_parts(self, robot: Any, part: str, hand: Any) -> tuple:
        document = ET.parse(robot.cfg.fpath).getroot()
        descendants = {robot.get_solver(part).root_link_name}
        while True:
            children = {
                j.find("child").attrib["link"]
                for j in document.findall("joint")
                if j.find("parent").attrib["link"] in descendants
            }
            if children <= descendants:
                break
            descendants |= children
        names = descendants - {p.name for p in hand}
        required = {
            link.attrib["name"]
            for link in document.findall("link")
            if link.attrib["name"] in names and link.find("collision") is not None
        }
        if not required <= set(self.native.get_link_names()):
            raise ValueError("Native arm collision-link inventory is incomplete.")
        parts = []
        root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy()
        for name in sorted(required):
            body = self.native.render_articulation.get_physical_body(name)
            if body is None:
                raise ValueError("Native arm collision body is unavailable.")
            meshes = _native_meshes(body)
            if not meshes:
                continue
            actual = (
                robot.compute_fk(
                    qpos=robot.get_qpos(),
                    link_names=[name],
                    qpos_joint_names=robot.joint_names,
                )[0, 0]
                .detach()
                .cpu()
                .numpy()
            )
            if not np.allclose(
                root @ actual, self.native.get_link_pose(name), atol=2e-4, rtol=0
            ):
                raise ValueError("Native arm FK does not match observed frames.")
            parts.append(
                SimpleNamespace(
                    name=name, meshes=meshes, rest_offset=float(body.get_rest_offset())
                )
            )
        if not parts:
            raise ValueError("Native arm has no enabled collision geometry.")
        return tuple(parts)


class MeasuredRigidPolicy:
    """Opt-in planning transforms; the normal executor owns all physical acceptance."""

    def __init__(self, simulation: Any, binding: Any, calls: tuple) -> None:
        self.simulation, self.binding = simulation, binding
        self.calls = {
            identity: (subject, container) for identity, subject, container in calls
        }
        self.rigids = {b.entity_id: b for b in binding.rigid_objects}
        self.containers = {b.entity_id: b for b in binding.containers}
        if any(
            subject not in self.rigids
            or (
                container
                and (
                    container not in self.containers
                    or self.containers[container].parent_id not in self.rigids
                )
            )
            for subject, container in self.calls.values()
        ):
            raise ValueError(
                "Measured manipulation requires exact rigid scene bindings."
            )

    def plan(self, engine: Any, request: Any, context: Any, build: Callable) -> Any:
        if request.invocation_id not in self.calls or _PREVIEW.get():
            return build(request)
        capture = initial_plan_capture(request.invocation_id)
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or not self.simulation.is_default_backend
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
        ):
            failed = engine.actions[request.skill_id].failed_plan(
                request,
                context,
                message="Measured rigid policy requires its single-environment Default IK scope.",
                failure_code="measured_rigid_scope_unavailable",
                retryable=False,
            )
            return (
                failed
                if capture is None
                else capture.accept(request, failed, engine.robot)
            )
        token = _CAPTURE.set(request.invocation_id)
        selected = request
        started = time.perf_counter()
        try:
            with measure_search("measured_rigid_policy"):
                if request.skill_id == "pick_up":
                    selected, plan = self._pick(engine, request, context, build)
                elif request.skill_id == "place":
                    selected, plan = self._place(engine, request, context, build)
                else:
                    raise ValueError(
                        "Measured rigid policy received a mismatched skill."
                    )
                if time.perf_counter() - started > 60:
                    raise ValueError(
                        "Measured rigid planning exceeded its unchanged 60-second budget."
                    )
            return (
                plan
                if capture is None
                else capture.accept(selected, plan, engine.robot)
            )
        except PlanningBudgetExceeded as error:
            if capture is not None:
                capture.failed(error)
            raise
        except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as error:
            failed = engine.actions[request.skill_id].failed_plan(
                request,
                context,
                message=str(error),
                failure_code="measured_rigid_unqualified",
                retryable=False,
            )
            return (
                failed
                if capture is None
                else capture.accept(request, failed, engine.robot)
            )
        finally:
            _CAPTURE.reset(token)

    def _screen(self, engine: Any, request: Any, context: Any, semantics: Any) -> Any:
        geometry_request = SimpleNamespace(
            goal=SimpleNamespace(semantics=semantics), binding=request.binding
        )
        screen = build_closing_contact_screen(
            engine.robot,
            self.simulation,
            self.binding,
            geometry_request,
            revision=4,
            context=context,
        )
        if screen is None:
            raise ValueError("Measured rigid hand geometry is unavailable.")
        return screen

    @staticmethod
    def _poses(robot: Any, q: torch.Tensor, parts: Any) -> np.ndarray:
        fk = (
            robot.compute_fk(
                qpos=q,
                link_names=[p.name for p in parts],
                qpos_joint_names=robot.joint_names,
            )
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().double().numpy()
        return root[None, None] @ fk

    def _pick(self, engine: Any, request: Any, context: Any, build: Callable) -> tuple:
        from .articulation_collision import check_free_motion, robot_collision_scope

        subject, container = self.calls[request.invocation_id]
        if container or subject != request.goal.semantics.entity_id:
            raise ValueError("Measured Pick source differs from its bound occurrence.")
        if (
            request.goal.grasp_xpos is not None
            or request.goal.object_pose is not None
            or request.skill_options.fixed_object_to_eef is not None
            or request.skill_options.pick_object_part != "center"
        ):
            raise ValueError(
                "Measured preshape cannot replace an explicitly constrained Pick."
            )
        original = build(request)
        if not original.plan_success.all():
            return request, original
        endpoint = request.binding.endpoint("primary", "motion")
        held = original.expected_effects.held_object_updates[endpoint.task_state_key]
        screen = self._screen(engine, request, context, request.goal.semantics)
        scene = _Scene(self.simulation, engine.robot)
        uid = self.rigids[subject].simulation_uid
        target = next(o for o in scene.obstacles if o["name"] == uid)
        grasp = held.grasp_xpos[0].detach().cpu().double().numpy()
        points = np.vstack(
            [
                _transform(m.vertices, np.linalg.inv(grasp) @ target["pose"])
                for m in target["meshes"]
            ]
        )
        pads = [p for p in screen.parts if p.pad]
        if len(pads) != 2:
            raise ValueError("Measured preshape requires two native pads.")

        def pad_points(part: Any, index: int) -> np.ndarray:
            return np.vstack(
                [_transform(m.vertices, part.tcp_poses[index]) for m in part.meshes]
            )

        axis = pad_points(pads[1], 0).mean(0) - pad_points(pads[0], 0).mean(0)
        axis /= np.linalg.norm(axis)
        projected = points @ axis
        fitting = []
        for index in range(screen.steps):
            negative, positive = (
                pad_points(pads[0], index) @ axis,
                pad_points(pads[1], index) @ axis,
            )
            gaps = (projected.min() - negative.max(), positive.min() - projected.max())
            if all(
                gap >= p.rest_offset + screen.target_rest_offset + 0.002
                for gap, p in zip(gaps, pads, strict=True)
            ):
                fitting.append(index)
        if not fitting:
            raise ValueError(
                "No native preshape contains the complete target projection."
            )
        fraction = fitting[-1] / (screen.steps - 1)
        hand_endpoint = request.binding.endpoint("primary", "grasp")
        opened, closed = [
            hand_endpoint.joint_positions(
                name,
                num_envs=1,
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            for name in ("open", "grasp")
        ]
        narrowed = torch.lerp(opened, closed, fraction)
        selected = replace(
            request,
            binding=request.binding.with_command_overrides(
                {("primary", "grasp"): {"open": JointPositionCommand(narrowed)}}
            ),
        )
        positions = original.joint_trajectory.positions.clone()
        ids = list(hand_endpoint.target.joint_ids)
        segments = {s.name: s for s in original.segments}
        dt = context.require_control_dt()
        limits = _joint_velocity_limits(engine.robot, None).to(positions)[
            context.env_ids
        ][:, ids]
        actual = context.robot.qpos[:, ids]
        count = (
            max(1, math.ceil(float(((narrowed - actual).abs() / (limits * dt)).amax())))
            + 1
        )
        if count > segments["approach"].stop:
            raise ValueError(
                "Measured preshape does not fit the unchanged approach time."
            )
        positions[:, : segments["approach"].stop, ids] = narrowed[:, None]
        positions[:, :count, ids] = interpolate_hand_qpos(
            actual, narrowed, n_waypoints=count
        )
        close = segments["close"]
        n = request.skill_options.hand_interp_steps
        if close.stop - close.start != n + request.skill_options.grasp_settle_steps:
            raise ValueError("Measured preshape requires the canonical close grid.")
        positions[:, close.start : close.start + n, ids] = interpolate_hand_qpos(
            narrowed, closed, n_waypoints=n
        )
        velocities = original.joint_trajectory.velocities
        if velocities is not None:
            computed = (
                torch.zeros_like(positions)
                if request.motion_policy.velocity_targets == "zero"
                else differentiate_positions(positions, original.joint_trajectory.dt)
            )
            stationary = positions[:, 1:] == positions[:, :-1]
            computed[:, :-1] = computed[:, :-1].masked_fill(stationary, 0.0)
            computed[:, -1] = 0.0
            velocities = velocities.clone()
            velocities[:, :, ids] = computed[:, :, ids]
        trajectory = replace(
            original.joint_trajectory, positions=positions, velocities=velocities
        )
        action = engine.actions["pick_up"]
        commands = action._joint_command_sequence(
            selected, trajectory, active_mask=original.plan_success
        )
        candidate = replace(
            original,
            joint_trajectory=trajectory,
            commands=commands,
            tracking=action._tracking_sequence(selected, commands),
        )
        selected_screen = self._screen(
            engine, selected, context, request.goal.semantics
        )
        closure = action.inspect_closing_plan(
            selected, context, candidate, screen=selected_screen
        )
        with robot_collision_scope():
            software = _inspect_pick_plan(engine.robot, selected, context, candidate)
            free = check_free_motion(
                engine.robot,
                self.simulation,
                positions[0, : segments["approach"].stop],
                endpoint.target.control_part,
            )
        native = scene.inspect(
            selected_screen.parts,
            self._poses(engine.robot, positions[0], selected_screen.parts),
            uid,
            source_contact_from=close.start,
        )
        if not all(c["valid"] for c in (closure, software, free, native)):
            raise ValueError(
                "Measured preshape failed unchanged closure, motion or native qualification."
            )
        report = dict(
            accepted=True,
            preshape_fraction=fraction,
            target_width_m=float(np.ptp(projected)),
            extra_side_reserve_m=0.002,
            unchanged_arm_and_timing=True,
            closure=closure,
            command_qualification=software,
            approach=free,
            native=native,
        )
        return selected, replace(
            candidate,
            diagnostics=replace(
                candidate.diagnostics,
                metadata={
                    **candidate.diagnostics.metadata,
                    "gen_sim_measured_rigid": report,
                },
            ),
        )

    def _landing(self, request: Any, context: Any, held: Any, container: Any) -> tuple:
        from shapely.affinity import translate
        from shapely.geometry import MultiPoint
        from shapely.ops import polylabel
        from .placement_evidence import _prepare_placement_geometry

        subject = held.semantics.entity_id
        reference = (
            resolve_pose_goal(
                SceneEntityPose(container.parent_id), context, name="measured_parent"
            )[0]
            .cpu()
            .numpy()
        )
        geometry = _prepare_placement_geometry(
            self.simulation, self.rigids[subject], self.rigids[container.parent_id], 1
        )
        floor, height = geometry.floors[0]
        occupied = []
        subjects = set(self.rigids) - {subject, container.parent_id}
        # Occupancy is a fresh geometric observation, not a planning-history claim of release.
        for source in sorted(subjects):
            obj = self.simulation.get_rigid_object(self.rigids[source].simulation_uid)
            pose = (
                resolve_pose_goal(
                    SceneEntityPose(source), context, name="measured_content"
                )[0]
                .cpu()
                .numpy()
            )
            points = _transform(
                obj.get_vertices(scale=True)[0].cpu().numpy(),
                np.linalg.solve(reference, pose),
            )
            polygon = MultiPoint(points[:, :2]).convex_hull
            if floor.covers(polygon) and -0.01 <= points[:, 2].min() - height <= 0.05:
                occupied.append((source, polygon))
        if not occupied:
            return request, {"observed_contents": []}
        release = resolve_pose_goal(request.goal.xpos, context, name="measured_release")
        object_release = release @ torch.linalg.inv(held.object_to_eef.to(release))
        relative = np.linalg.solve(reference, object_release[0].cpu().numpy())
        vertices = geometry.vertices[0].numpy() @ relative[:3, :3].T
        footprint = MultiPoint(vertices[:, :2]).convex_hull
        safe = floor.buffer(-_MARGIN)
        allowed = safe
        for x, y in footprint.exterior.coords[:-1]:
            allowed = allowed.intersection(
                translate(safe, xoff=-float(x), yoff=-float(y))
            )
        for _, polygon in occupied:
            pairs = (
                np.asarray(polygon.exterior.coords[:-1])[:, None]
                - np.asarray(footprint.exterior.coords[:-1])[None]
            )
            forbidden = MultiPoint(pairs.reshape(-1, 2)).convex_hull.buffer(
                _MARGIN / math.cos(math.pi / 64), quad_segs=16
            )
            allowed = allowed.difference(forbidden)
        regions = (
            [allowed]
            if allowed.geom_type == "Polygon"
            else [g for g in getattr(allowed, "geoms", ()) if g.geom_type == "Polygon"]
        )
        regions = [g for g in regions if not g.is_empty and g.area > 0]
        if not regions:
            raise ValueError(
                "No measured inside landing retains both 5mm reservations."
            )
        choices = [polylabel(p, tolerance=0.001) for p in regions]
        point = max(choices, key=lambda p: allowed.boundary.distance(p))
        placed = translate(footprint, xoff=point.x, yoff=point.y)
        if not safe.covers(placed) or any(
            placed.distance(p) < _MARGIN - 1e-8 for _, p in occupied
        ):
            raise ValueError(
                "Measured inside proposal failed full-footprint reservations."
            )
        delta = release.new_tensor(
            reference[:3, :3]
            @ np.array([point.x - relative[0, 3], point.y - relative[1, 3], 0.0])
        )
        if request.goal.xpos.world_displacement is not None:
            delta += request.goal.xpos.world_displacement.to(delta)
        selected = replace(
            request,
            goal=replace(
                request.goal, xpos=replace(request.goal.xpos, world_displacement=delta)
            ),
        )
        return selected, dict(
            observed_contents=[uid for uid, _ in occupied],
            selected_root_xy=[point.x, point.y],
            floor_boundary_gap_m=floor.boundary.distance(placed),
            content_gaps_m={uid: placed.distance(p) for uid, p in occupied},
            required_reservation_m=_MARGIN,
            release_height_unchanged=True,
        )

    def _place_check(
        self,
        engine: Any,
        request: Any,
        context: Any,
        plan: Any,
        screen: Any,
        scene: _Scene,
        arm: Any,
        uid: str,
    ) -> dict:
        if not plan.plan_success.all():
            return {"valid": False, "planning_success": False}
        robot = engine.robot
        q = plan.joint_trajectory.positions[0]
        stop = next(s.stop for s in plan.segments if s.name == "approach")
        endpoint = request.binding.endpoint("primary", "motion").target
        hand_ids = list(request.binding.endpoint("primary", "grasp").target.joint_ids)
        held_q = q[:stop].clone()
        held_q[:, hand_ids] = context.robot.qpos[0, hand_ids]
        observed = scene.inspect(
            screen.parts, self._poses(robot, held_q, screen.parts), uid
        )
        emitted = scene.inspect(screen.parts, self._poses(robot, q, screen.parts), uid)
        arm_scene = scene.inspect(arm, self._poses(robot, q, arm), uid)
        tcp = (
            robot.compute_batch_fk(
                qpos=q[None, :stop, list(endpoint.joint_ids)],
                name=endpoint.control_part,
                to_matrix=True,
            )[0]
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        actual = (
            robot.compute_batch_fk(
                qpos=context.robot.qpos[:, None, list(endpoint.joint_ids)],
                name=endpoint.control_part,
                to_matrix=True,
            )[0, 0]
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        source = next(o for o in scene.obstacles if o["name"] == uid)
        relative = np.linalg.inv(actual) @ source["pose"]
        cargo = SimpleNamespace(
            name=uid, meshes=source["meshes"], rest_offset=source["rest"]
        )
        carried = scene.inspect((cargo,), (tcp @ relative)[:, None], uid)
        own = _inspect_pick_motion(robot, q, endpoint.control_part)
        velocity = bool(_validate_final_plan_velocity(plan, robot)["valid_mask"].all())
        return dict(
            valid=all(c["valid"] for c in (observed, emitted, arm_scene, carried, own))
            and velocity,
            observed_hold_hand=observed,
            emitted_hand=emitted,
            operated_arm_scene=arm_scene,
            rigid_cargo_approach=carried,
            robot_self_proxy=own,
            full_robot_velocity_valid=velocity,
        )

    def _place(self, engine: Any, request: Any, context: Any, build: Callable) -> tuple:
        subject, target = self.calls[request.invocation_id]
        if (
            not isinstance(request.goal.xpos, SceneEntityPose)
            or request.goal.xpos.entity_id != target
        ):
            raise ValueError("Measured Place target differs from its bound occurrence.")
        held = context.get_held_object(
            request.binding.endpoint("primary", "motion").task_state_key
        )
        if held is None or held.semantics.entity_id != subject:
            raise ValueError("Measured Place requires its verified input hold.")
        container = self.containers[target]
        if (
            container.release_clearance != 0.12
            or request.goal.xpos.world_orientation is not None
            or request.skill_options.max_approach_retract_z is not None
        ):
            raise ValueError(
                "Measured inside policy cannot change an explicit constrained release."
            )
        selected, landing = self._landing(request, context, held, container)
        screen = self._screen(engine, request, context, held.semantics)
        scene = _Scene(self.simulation, engine.robot)
        part = request.binding.endpoint("primary", "motion").target.control_part
        arm = scene.arm_parts(engine.robot, part, screen.parts)
        attempts = []
        for extra in (0.0, 0.05, 0.10, 0.15):
            candidate = replace(
                selected,
                skill_options=replace(
                    selected.skill_options,
                    lift_height=selected.skill_options.lift_height + extra,
                ),
            )
            plan = build(candidate)
            with measure_search("measured_inside_clearance"):
                qualification = self._place_check(
                    engine,
                    candidate,
                    context,
                    plan,
                    screen,
                    scene,
                    arm,
                    self.rigids[subject].simulation_uid,
                )
            attempts.append(
                dict(
                    lift_height_m=candidate.skill_options.lift_height,
                    qualification=qualification,
                )
            )
            if qualification["valid"]:
                return candidate, replace(
                    plan,
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_measured_rigid": dict(
                                accepted=True,
                                landing=landing,
                                candidates=attempts,
                                release_clearance_m_unchanged=container.release_clearance,
                                scope="discrete_observed_geometry_not_continuous_or_physical_success",
                            ),
                        },
                    ),
                )
        raise ValueError(
            "Bounded measured inside transport candidates are unqualified."
        )
