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

"""Conditional layout proposals with shared full task screening."""

from __future__ import annotations

from dataclasses import replace
import math
import time
import numpy as np

from ._pose_sampling import _AdaptivePoseSource
from ._pose_sampling_cfg import _ROBOT_LAYOUT_STRATEGIES
from ._robot_first_sampling import _anchor_targets, _prepare_anchor_pools
from embodichain.lab.sim.scene.expansion import (
    SceneLayoutCondition,
    scene_layout_input_from_sim,
    prepare_scene_randomization,
    generate_scene_variants,
    ScenePoseChange,
    SceneVariant,
)

__all__: list[str] = []


class _LayoutPoseSource(_AdaptivePoseSource):
    """Generate or certify complete layouts before checking task branches.

    The numerical layout source owns no World. The existing task backend owns
    workspace seed selection, full-state IK and collision-checked path screening.
    Runtime execution still replans and independently validates measured outcomes.
    """

    def __init__(self, env, options, *, seed, revision):
        before = self._world_clock(env.sim)
        self._collision = None
        super().__init__(env, options, seed=seed, revision=revision)
        for key, value in self.cfg["screening"].items():
            setattr(self.budget.cfg, key, value)
        self.budget.cfg.__post_init__()
        layout = self.cfg["layout"]
        source = scene_layout_input_from_sim(
            env.sim,
            support_uid=layout["support_uid"],
            parent_by_uid=layout["parent_by_uid"],
            movable_entity_ids=tuple(
                uid for uid in options["scene"] if uid in layout["parent_by_uid"]
            ),
            scene_id=revision,
        )
        self._prepared_layout = prepare_scene_randomization(source, layout["policy"])
        self._fixed_layout = {
            uid: ScenePoseChange(uid, source.poses[uid])
            for uid in self.configuration
            if uid in source.poses and uid not in self._prepared_layout.movable_uids
        }
        self._conditions = SceneLayoutCondition(layout.get("conditions", {}))
        # Compile/check condition syntax and permissions before any row is reset.
        from embodichain.lab.sim.scene.expansion.conditions import _condition_prepared

        _condition_prepared(self._prepared_layout, self._conditions)
        nodes = self._prepared_layout._geometry["nodes"]
        for n in nodes:
            if n["uid"] in options["scene"] and n["parent"] < 0:
                bounds = options["scene"][n["uid"]]
                if "follow" not in bounds and (
                    bounds["yaw_rad"] != [0, 0] or bounds["translation_m"][2] != [0, 0]
                ):
                    raise ValueError(
                        "Root layout sampling supports XY only; put member yaw in a declared intra-group domain"
                    )
        from embodichain.lab.sim.scene.feasibility import request_from_mapping

        entities = {
            uid: obj.get_local_pose(to_matrix=True)[0].detach().cpu().double().tolist()
            for uid in env.sim.get_rigid_object_uid_list()
            if (obj := env.sim.get_rigid_object(uid)) is not None
        }
        entities[env.robot.uid] = self.roots[env.robot.uid]
        if getattr(env.sim, "_default_plane", None) is not None:
            entities["default_plane"] = np.eye(4).tolist()
        request = request_from_mapping(
            dict(
                task_id=revision,
                robot_profile_id=env.robot.uid,
                policy_id=revision,
                validator_id=revision,
                parent_scene_id=revision,
                entity_poses=entities,
                variables=[],
                modes=[self.cfg["task_mode"]],
            )
        )
        self._mode = request.modes[0]
        for phase in self._mode.phases:
            for target in phase.targets:
                if (
                    target.control_part not in env.robot.control_parts
                    or env.robot.get_solver(name=target.control_part) is None
                ):
                    raise ValueError(
                        f"Task phase names unavailable control part {target.control_part!r}"
                    )
        self._entities = {k: np.array(v) for k, v in entities.items()}
        self._initial_qpos = None
        self._anchor_pools = ()
        self._workspace_seconds = self._screening_seconds = 0.0
        if self.cfg["strategy"] in _ROBOT_LAYOUT_STRATEGIES:
            started = time.perf_counter()
            selections = _anchor_targets(
                self._mode,
                layout,
                self._prepared_layout.policy,
                self.cfg["robot_anchors"],
            )
            self._anchor_pools = _prepare_anchor_pools(
                env.robot,
                self.budget,
                self._prepared_layout,
                self.cfg,
                selections,
                self.configuration,
                self.poses,
                seed=seed,
            )
            self._workspace_seconds = time.perf_counter() - started

        from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
            collision_bodies_from_simulation,
        )
        from .acceptance import _GeometryRow
        from embodichain.lab.sim.scene.feasibility import NewtonCollisionChecker

        bodies = collision_bodies_from_simulation(env)
        known = {b.frame_uid for b in bodies}
        ignored, _ = _GeometryRow.contact_permissions(env, options["validation"], known)
        try:
            self._collision = NewtonCollisionChecker(
                bodies, ignored_pairs=ignored, device="cpu", max_shapes=512
            )
            self._check_clock(env.sim, before)
        except BaseException:
            self.close()
            raise
        self._scene_identity = (
            (id(env.sim.spawn_result), env.sim.spawn_result.topology_revision)
            if hasattr(env.sim, "spawn_result")
            else None
        )
        self._task_checks = {}
        self._anchor_checks = {}

    def bind_program(self, adapter, program):
        """Keep the declared phase graph and live runtime independent authorities."""
        if not tuple(program.iter_segments()):
            raise ValueError("Layout sampling requires a nonempty task program")
        # The owner binds every group before rolling admission. Retain its
        # nominal planning start even while another row subsequently executes.
        self._initial_qpos = (
            self.env.robot.get_qpos()[0].detach().cpu().double().numpy().copy()
        )

    @staticmethod
    def _world_clock(sim):
        return getattr(sim, "physics_step_index", None), getattr(
            sim, "simulation_time", None
        )

    @classmethod
    def _check_clock(cls, sim, before):
        if before != cls._world_clock(sim):
            raise RuntimeError("Scene proposal screening advanced the owning World")

    def _offer_origin(self, raw):
        if self.cfg["strategy"] == "bidirectional":
            return "robot_first" if (raw + self.seed) % 2 else "scene_first"
        return super()._offer_origin(raw)

    def _offer(self, candidate, rng, raw):
        sim = self.env.sim
        if self._scene_identity is not None and self._scene_identity != (
            id(sim.spawn_result),
            sim.spawn_result.topology_revision,
        ):
            raise RuntimeError(
                "Prepared layout geometry is stale after a topology change"
            )
        before = self._world_clock(sim)
        started = time.perf_counter()
        try:
            return self._offer_layout(candidate, raw, self._offer_origin(raw))
        finally:
            self._screening_seconds += time.perf_counter() - started
            self._check_clock(sim, before)
            self._task_checks.setdefault(raw, []).append(
                dict(
                    stage="screening.clock",
                    status="passed",
                    reason="Owning World step/time unchanged",
                    phase_id=None,
                )
            )

    def _offer_layout(self, candidate, raw, origin):
        if origin != "bounded_pose":
            conditions = self._conditions
            if origin == "robot_first":
                ordinal = raw // 2 if self.cfg["strategy"] == "bidirectional" else raw
                visit, index = divmod(ordinal, len(self._anchor_pools))
                pool = self._anchor_pools[index]
                conditions = pool.condition(visit, conditions)
                self._anchor_checks[raw] = {
                    "root_uid": pool.selection.root_uid,
                    "phase_id": pool.selection.phase_id,
                    "control_part": pool.selection.target.control_part,
                    "pose": (
                        None
                        if conditions is None
                        else conditions.poses[pool.selection.root_uid]
                    ),
                }
                if conditions is None:
                    return candidate, origin, False, "no_workspace_anchor"
            batch = generate_scene_variants(
                self._prepared_layout,
                seed=self.seed,
                candidate_count=1,
                max_attempts=1,
                start_ordinal=raw,
                conditions=conditions,
            )
            if not batch.variants:
                return candidate, origin, False, "layout_budget_exhausted"
            generated = batch.variants[0]
            # References outside the support tree (e.g. a numerical goal)
            # retain their declared independent bounded sampling.
            changes = {
                c.entity_uid: self._fixed_layout.get(c.entity_uid, c)
                for c in candidate.scene.pose_changes
            }
            changes.update({c.entity_uid: c for c in generated.pose_changes})
            candidate = replace(
                candidate,
                scene=SceneVariant(
                    candidate.scene.parent_scene_id,
                    self.seed,
                    raw,
                    tuple(changes.values()),
                    generated.metadata,
                ),
            )
        else:
            # Equal downstream geometric/kinematic gates for the simple baseline.
            for change in candidate.scene.pose_changes:
                fixed = self._fixed_layout.get(change.entity_uid)
                if fixed is not None and not np.allclose(
                    change.arena_pose, fixed.arena_pose, rtol=0, atol=1e-9
                ):
                    return candidate, origin, False, "layout_moves_fixed_entity"
            anchors = {
                c.entity_uid: c.arena_pose
                for c in candidate.scene.pose_changes
                if c.entity_uid in self._prepared_layout.movable_uids
            }
            for uid, pose in self._conditions.poses.items():
                actual = anchors.get(uid, self._prepared_layout.source.poses[uid])
                if not np.allclose(actual, pose, rtol=0, atol=1e-9):
                    return candidate, "bounded_pose", False, "condition_mismatch"
            batch = generate_scene_variants(
                self._prepared_layout,
                seed=self.seed,
                candidate_count=1,
                max_attempts=1,
                start_ordinal=raw,
                conditions=SceneLayoutCondition(anchors),
            )
            if not batch.variants:
                return candidate, "bounded_pose", False, "layout_budget_exhausted"
        poses = {
            **self._entities,
            **{
                c.entity_uid: np.array(c.arena_pose)
                for c in candidate.scene.pose_changes
            },
            **{uid: np.array(pose) for uid, pose in candidate.deployment.poses.items()},
        }
        # Never expand explicit task-facing ranges while applying layout policy.
        for c in candidate.scene.pose_changes:
            opts = self.configuration[c.entity_uid]
            if "follow" in opts:
                continue
            delta = np.asarray(c.arena_pose)[:3, 3] - self.poses[c.entity_uid][:3, 3]
            bounds = np.asarray(opts["translation_m"])
            rotation = np.asarray(c.arena_pose)[:3, :3] @ np.linalg.inv(
                self.poses[c.entity_uid][:3, :3]
            )
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            low, high = opts["yaw_rad"]
            yaw += 2 * math.pi * round(((low + high) / 2 - yaw) / (2 * math.pi))
            if (
                np.any(delta < bounds[:, 0] - 1e-6)
                or np.any(delta > bounds[:, 1] + 1e-6)
                or yaw < low - 1e-6
                or yaw > high + 1e-6
            ):
                return (
                    candidate,
                    origin,
                    False,
                    "layout_outside_task_domain",
                )
        from embodichain.lab.sim.scene.feasibility import (
            RobotFeasibilityBackend,
            plan_task_mode,
        )

        backend = RobotFeasibilityBackend(
            self.env.robot,
            self._collision,
            self.budget.cfg,
            poses,
            seed=self.seed + raw,
            initial_qpos=self._initial_qpos,
        )
        plan = plan_task_mode(self._mode, backend, poses, self.budget)
        self._task_checks[raw] = [
            {
                "stage": c.stage,
                "status": c.status,
                "reason": c.reason,
                "phase_id": c.phase_id,
            }
            for c in plan.checks
        ]
        reason = (
            "passed"
            if plan.accepted
            else next(
                (c.reason for c in plan.checks if c.status != "passed"),
                "task_path_unverified",
            )
        )
        return candidate, origin, plan.accepted, reason

    def checkpoint(self):
        super().checkpoint()
        if self.output is not None and hasattr(self, "_task_checks"):
            from embodichain.data_pipeline.recording.media import _atomic_json

            _atomic_json(
                self.output / "scene-first-checks.json",
                dict(
                    phase_checks=self._task_checks,
                    layout_source=self._prepared_layout.source.fingerprint,
                    conditions=self._conditions.to_dict(),
                    workspace_prior=[pool.metrics() for pool in self._anchor_pools],
                    workspace_anchor_checks=self._anchor_checks,
                    workspace_prepare_wall_s=self._workspace_seconds,
                    proposal_screen_wall_s=self._screening_seconds,
                ),
            )

    def close(self):
        """Release only query resources, without modifying the owning World."""
        if self._collision is not None:
            self._collision.close()
            self._collision = None
