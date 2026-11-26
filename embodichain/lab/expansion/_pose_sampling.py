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

"""Selectable bounded and workspace-guided scene proposals with owned feedback."""

from __future__ import annotations
from collections.abc import Mapping
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import numpy as np
import torch

__all__: list[str] = []


class _BoundedPoseSource:
    """Sample bounded arena-frame changes with task-declared support followers."""

    def __init__(
        self, env: Any, configuration: Mapping[str, Any], *, seed: int, revision: str
    ) -> None:
        self.configuration, self.seed, self.request_id = (
            {uid: deepcopy(dict(options)) for uid, options in configuration.items()},
            seed,
            revision,
        )
        self.poses = {
            uid: env.sim.get_rigid_object(uid)
            .get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
            for uid in configuration
        }
        self.roots = {
            env.robot.uid: env.robot.get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .tolist()
        }

    def candidate(self, ordinal: int) -> JointSceneCandidate:
        import numpy as np
        from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
        from embodichain.lab.sim.scene.feasibility import (
            JointSceneCandidate,
            RobotDeploymentCandidate,
        )

        offsets = {}
        for uid, options in self.configuration.items():
            if "follow" in options:
                continue
            seed = int.from_bytes(
                hashlib.sha256(f"{self.seed}/{ordinal}/{uid}".encode()).digest()[:8],
                "little",
            )
            generator = np.random.default_rng(seed)
            offsets[uid] = [
                generator.uniform(*bounds)
                for bounds in (*options["translation_m"], options["yaw_rad"])
            ]
        changes, parameters = [], []
        for uid, options in self.configuration.items():
            dx, dy, dz, yaw = offsets[options.get("follow", uid)]
            pose = self.poses[uid].copy()
            c, s = np.cos(yaw), np.sin(yaw)
            pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
            pose[:3, 3] += (dx, dy, dz)
            changes.append(ScenePoseChange(uid, tuple(map(tuple, pose))))
            parameters.extend((dx, dy, dz, yaw))
        scene = SceneVariant(self.request_id, self.seed, ordinal, tuple(changes))
        return JointSceneCandidate(
            self.request_id,
            scene,
            RobotDeploymentCandidate(self.roots),
            "object_pose",
            tuple(parameters),
        )


from embodichain.lab.sim.scene.feasibility.budget import (
    FeasibilityBudget,
    FeasibilityBudgetExhausted,
)


class _PoseBudget(FeasibilityBudget):
    """Use whole-job limits, including raw proposals discarded by selection."""

    def charge(self, kind: str, amount: int = 1) -> None:
        self.check()
        self._reserve(kind, amount)

    def _reserve(self, kind: str, amount: int = 1) -> None:
        if type(amount) is not int or amount < 1:
            raise ValueError("Budget charges must be positive integers")
        limits = {
            "candidates": self.cfg.max_candidates,
            "ik": self.cfg.max_ik_calls,
            "planning": self.cfg.max_planning_calls,
            "rollouts": self.cfg.max_rollouts,
        }
        if self.counts[kind] + amount > limits[kind]:
            self._blocked = (self.validation, kind + "_budget_exhausted")
            raise FeasibilityBudgetExhausted(self._blocked[1])
        self.counts[kind] += amount


class _AdmittedPoseBudget:
    """Retain shared call limits/cancellation after the admission deadline.

    Already admitted trajectories still have the host's original max_steps and
    bridge safe-stop rules. The source's raw budget alone admits new work.
    """

    def __init__(self, budget: _PoseBudget) -> None:
        self.budget = budget

    def check(self) -> None:
        budget = self.budget
        if budget._blocked is not None and budget._blocked[0] == budget.validation:
            raise FeasibilityBudgetExhausted(budget._blocked[1])
        if budget.cancelled is not None and budget.cancelled.is_set():
            raise FeasibilityBudgetExhausted("cancelled")

    def charge(self, kind: str, amount: int = 1) -> None:
        self.check()
        self.budget._reserve(kind, amount)


class _AdaptivePoseSource(_BoundedPoseSource):
    """One bounded proposal stream; candidates never change after allocation."""

    def __init__(self, env, options, *, seed, revision):
        super().__init__(env, options["scene"], seed=seed, revision=revision)
        from .task_pose_coverage import TaskPoseCoverage
        from ._gym_utils import _hash
        from embodichain.lab.sim.scene.feasibility import (
            FeasibilityBudget,
            SceneRobotFeasibilityCfg,
        )

        self.env, self.options = env, deepcopy(options)
        self.cfg, self.schedule = (
            self.options["scene_sampling"],
            self.options["scheduling"],
        )
        if self.cfg["control_part"] not in env.robot.control_parts:
            raise ValueError("scene_sampling.control_part is absent from the robot")
        self.coverage_cfg = self.options.get("pose_coverage")
        invariant = {
            "robot": env.robot.cfg.to_dict(),
            "physics": env.sim.sim_config.physics_cfg.to_dict(),
            "poses": self.poses,
            "roots": self.roots,
            "domain": self.configuration,
            "coverage": self.coverage_cfg,
            "validation": options["validation"],
            "objects": {
                uid: obj.cfg.to_dict()
                for uid in getattr(env.sim, "get_rigid_object_uid_list", lambda: ())()
                if (obj := env.sim.get_rigid_object(uid)) is not None
            },
        }
        # _hash handles JSON data only; pose arrays are owned host arrays here.
        invariant.pop("poses")
        self.domain_id = _hash(invariant)
        self.coverage = (
            None
            if self.coverage_cfg is None
            else TaskPoseCoverage(
                self.domain_id,
                self.coverage_cfg.get("entity")
                or next(iter(self.coverage_cfg["measurements"].values()))["entity"],
                self.coverage_cfg["grids"],
                measurements=self.coverage_cfg.get("measurements"),
            )
        )
        self.budget = _PoseBudget(
            SceneRobotFeasibilityCfg(
                max_candidates=self.cfg["max_candidates"],
                max_ik_calls=self.cfg["max_ik_calls"],
                max_planning_calls=self.cfg["max_planning_calls"],
                max_rollouts=self.cfg["max_rollouts"],
                max_wall_time_s=options["engine"]["max_wall_time_s"],
            )
        )
        self.budget.validation = (
            True  # Use the full host allocation, not search/held-out splitting.
        )
        self.planning_budget = _AdmittedPoseBudget(self.budget)
        self._candidates = {}
        self._checks = {}
        self._predicted = {}
        self._outcomes = {}
        self._stats = {}
        self._events = []
        self._next_checkpoint_events = 1
        self._raw_ordinal = 0
        self._root_pose = env.robot.get_local_pose(to_matrix=True)[0].detach().clone()
        self._joint_seed = (
            env.robot.get_qpos(name=self.cfg["control_part"])[0:1].detach().clone()
            if self.cfg["strategy"] == "workspace_joint"
            else None
        )
        self._local = None
        self.output = None
        if self.cfg["strategy"] == "workspace_joint":
            self._prepare_workspace()

    def bind_program(self, adapter, program):
        """Reject mismatched authored grasp branches before any candidate mutates a row."""
        if self.cfg["strategy"] != "workspace_joint":
            return
        profile = adapter._factory.create_robot_skill_profile()
        preset = profile.presets[self.env.cfg.task_program.integration.runtime_preset]
        pick = preset.skill_option_templates.get(
            "pick_up", preset.skill_option_templates.get("pick")
        )
        fixed = None if pick is None else pick.fixed_object_to_eef
        if fixed is None:
            raise ValueError(
                "workspace_joint requires a fixed_object_to_eef pick policy; sampled grasp replacement is unsupported"
            )
        from embodichain.lab.task_program.semantics.calls import Pick

        picked = {
            entry.call.object.entity_id
            for segment in program.iter_segments()
            for entry in segment.calls
            if type(entry.call) is Pick
        }
        if not picked:
            raise ValueError(
                "workspace_joint requires an explicit Pick in its task program"
            )
        for entity in picked:
            transform = self.cfg["targets"].get(entity)
            if transform is None or not np.allclose(
                np.array(transform), fixed.detach().cpu().numpy(), rtol=0, atol=1e-6
            ):
                raise ValueError(
                    "Workspace grasp target must match the executed fixed_object_to_eef branch"
                )

    def _prepare_workspace(self):
        robot = self.env.robot
        part = self.cfg["control_part"]
        n = self.cfg["sample_count"]
        generator = torch.Generator(device=robot.device).manual_seed(self.seed)
        if self.cfg["workspace_source"] == "cache":
            cache = robot.get_workspace(part)
            settings = robot.cfg.workspace_cfg[part]
            ids = cache.sample_indices(
                n,
                strategy=settings.strategy,
                min_score=settings.min_score,
                generator=generator,
            )
            if bool((ids < 0).any()):
                raise ValueError("Workspace cache returned invalid indices")
            q = cache.qpos[ids].to(robot.device)
        else:
            limits = robot.get_qpos_limits(name=part, env_ids=[0])[0]
            if not bool(torch.isfinite(limits).all()):
                raise ValueError("Workspace FK requires finite joint limits")
            q = limits[:, 0] + torch.rand(
                (n, len(limits)), device=robot.device, generator=generator
            ) * (limits[:, 1] - limits[:, 0])
        fk = robot.compute_batch_fk(q[None], name=part, env_ids=[0], to_matrix=True)[0]
        if not bool(torch.isfinite(fk).all()):
            raise ValueError("Workspace FK produced nonfinite poses")
        self._local = (
            (torch.linalg.inv(self._root_pose)[None] @ fk).detach().cpu().numpy().copy()
        )

    def _guided(self, candidate, rng):
        from dataclasses import replace
        from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant

        poses = {
            c.entity_uid: np.array(c.arena_pose) for c in candidate.scene.pose_changes
        }
        origin = "uniform_exploration"
        if rng.random() >= self.cfg["uniform_fraction"]:
            origin = "workspace_joint"
            root = np.array(candidate.deployment.poses[self.env.robot.uid])
            for uid, transform in self.cfg["targets"].items():
                proposed = (
                    root[None] @ self._local @ np.linalg.inv(np.array(transform))[None]
                )
                base = self.poses[uid]
                options = self.configuration[uid]
                yaw = np.arctan2(proposed[:, 1, 0], proposed[:, 0, 0]) - math.atan2(
                    base[1, 0], base[0, 0]
                )
                bounds = np.array(options["yaw_rad"])
                yaw += (2 * math.pi) * np.round((bounds.mean() - yaw) / (2 * math.pi))
                xy = proposed[:, :2, 3] - base[:2, 3]
                valid = np.flatnonzero(
                    np.all(
                        (xy >= np.array(options["translation_m"])[:2, 0])
                        & (xy <= np.array(options["translation_m"])[:2, 1]),
                        axis=1,
                    )
                    & (yaw >= bounds[0])
                    & (yaw <= bounds[1])
                )
                if not len(valid):
                    return candidate, "workspace_no_projected_sample"
                index = int(rng.choice(valid))
                angle = float(yaw[index])
                c, s = math.cos(angle), math.sin(angle)
                poses[uid][:3, :3] = (
                    np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ base[:3, :3]
                )
                poses[uid][:2, 3] = proposed[index, :2, 3]
            for uid, opts in self.configuration.items():
                if "follow" in opts:
                    parent = opts["follow"]
                    poses[uid] = self._follower_pose(uid, parent, poses[parent])
        changes = tuple(
            ScenePoseChange(uid, tuple(map(tuple, pose))) for uid, pose in poses.items()
        )
        return (
            replace(
                candidate,
                scene=SceneVariant(
                    self.request_id, self.seed, candidate.scene.ordinal, changes
                ),
            ),
            origin,
        )

    def _check(self, candidate):
        from embodichain.lab.sim.motion._budget import _planning_budget_scope

        robot = self.env.robot
        part = self.cfg["control_part"]
        poses = {
            c.entity_uid: np.array(c.arena_pose) for c in candidate.scene.pose_changes
        }
        for uid, transform in self.cfg["targets"].items():
            target = torch.tensor(
                poses[uid] @ np.array(transform),
                device=robot.device,
                dtype=torch.float32,
            )[None]
            with _planning_budget_scope(self.budget):
                ok, q = robot.compute_ik(
                    pose=target, joint_seed=self._joint_seed, name=part, env_ids=[0]
                )
            limits = robot.get_qpos_limits(name=part, env_ids=[0])
            if (
                not bool(ok.all())
                or not bool(torch.isfinite(q).all())
                or not bool(((q >= limits[..., 0]) & (q <= limits[..., 1])).all())
            ):
                return False, "pose_ik_or_joint_limits"
            fk = robot.compute_fk(qpos=q, name=part, env_ids=[0], to_matrix=True)
            angle = torch.acos(
                (
                    (
                        torch.einsum(
                            "bii->b", fk[:, :3, :3].transpose(1, 2) @ target[:, :3, :3]
                        )
                        - 1
                    )
                    / 2
                ).clamp(-1, 1)
            )
            if (
                not bool(torch.isfinite(fk).all())
                or float((fk[:, :3, 3] - target[:, :3, 3]).norm()) > 1e-3
                or float(angle.max()) > 1e-2
            ):
                return False, "full_pose_fk_residual"
        return True, "passed"

    def _follower_pose(
        self, uid: str, parent: str, parent_pose: np.ndarray
    ) -> np.ndarray:
        """Apply the parent's arena displacement and rotation to the nominal follower."""
        nominal_parent = self.poses[parent]
        pose = self.poses[uid].astype(parent_pose.dtype, copy=True)
        pose[:3, 3] = pose[:3, 3] + parent_pose[:3, 3] - nominal_parent[:3, 3]
        pose[:3, :3] = parent_pose[:3, :3] @ nominal_parent[:3, :3].T @ pose[:3, :3]
        return pose

    def _offer(self, candidate, rng, raw):
        if self.cfg["strategy"] == "workspace_joint":
            candidate, origin = self._guided(candidate, rng)
            ok, reason = self._check(candidate)
            return candidate, origin, ok, reason
        return candidate, "bounded_pose", True, "passed"

    def _offer_origin(self, raw):
        return self.cfg["strategy"]

    def candidate(self, ordinal):
        from dataclasses import replace
        from embodichain.lab.sim.scene.feasibility import FeasibilityBudgetExhausted

        if ordinal in self._candidates:
            return self._candidates[ordinal]
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < self.options["engine"]["max_attempts"]
        ):
            raise ValueError("ordinal must be within the configured attempt budget")
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}/{ordinal}/pose-selection".encode()
                ).digest()[:8],
                "little",
            )
        )
        remaining = max(
            1,
            self.cfg["max_rollouts"]
            - self.budget.counts["rollouts"]
            - len(self._predicted),
        )
        quota = max(1, (self.cfg["max_candidates"] - self._raw_ordinal) // remaining)
        count = (
            1
            if self.schedule["policy"] == "fifo"
            and self.cfg["strategy"] == "bounded_pose"
            else min(self.schedule["pool_size"], quota)
        )
        pool = []
        last = None
        origin = "bounded_pose"
        reason = "no_candidate"
        for _ in range(count):
            raw = self._raw_ordinal
            # Keep the historical simple FIFO sequence exactly intact.
            candidate = super().candidate(
                ordinal
                if count == 1 and self.cfg["strategy"] == "bounded_pose"
                else raw
            )
            last = candidate
            origin = self._offer_origin(raw)
            try:
                self.budget.charge("candidates")
                self._raw_ordinal += 1
                ok = True
                candidate, origin, ok, reason = self._offer(candidate, rng, raw)
            except FeasibilityBudgetExhausted as e:
                ok = False
                reason = str(e)
            poses = {c.entity_uid: c.arena_pose for c in candidate.scene.pose_changes}
            cells = (
                ()
                if self.coverage is None
                else (
                    self.coverage.proposal_cells(poses)
                    if self.coverage_cfg.get("measurements") is not None
                    else self.coverage.cells(
                        poses[self.coverage_cfg["entity"]],
                        poses[self.coverage_cfg["place_reference"]],
                    )
                )
            )
            self._events.append(
                {
                    "event": "proposal",
                    "ordinal": ordinal,
                    "raw_ordinal": raw,
                    "origin": origin,
                    "screened": ok,
                    "reason": reason,
                    "pose_changes": {
                        c.entity_uid: c.arena_pose for c in candidate.scene.pose_changes
                    },
                    "installation": dict(candidate.deployment.poses),
                    "cells": cells,
                }
            )
            if ok:
                pool.append((candidate, cells, origin, raw))
            if self.schedule["policy"] == "fifo" and pool:
                break
            if self.budget._blocked is not None:
                break
        valid = bool(pool)
        if pool:
            if (
                self.schedule["policy"] == "fifo"
                or rng.random() < self.schedule["exploration_fraction"]
            ):
                chosen = pool[int(rng.integers(len(pool)))]
                selection = "exploration"
            else:
                predicted = Counter(
                    c for values in self._predicted.values() for c in values
                )

                def score(entry):
                    cells = entry[1]
                    success, total = self._stats.get(self._region(cells), (0, 0))
                    return (
                        (success + 1)
                        / (total + 2)
                        * self.coverage.gain(cells, predicted)
                    )

                scores = np.array([score(c) for c in pool])
                ids = np.flatnonzero(scores == scores.max())
                chosen = pool[int(rng.choice(ids))]
                selection = "coverage"
            candidate, cells, origin, raw = chosen
            self._predicted[ordinal] = cells
            reason = "passed"
        else:
            candidate = last
            cells = ()
            selection = "rejected"
        # Candidate identity includes the complete correlated vector, not just
        # the raw draw before workspace conditioning.
        parameters = tuple(
            v
            for change in candidate.scene.pose_changes
            for row in change.arena_pose
            for v in row
        )
        candidate = replace(
            candidate,
            scene=replace(candidate.scene, ordinal=ordinal),
            parameters=parameters,
        )
        self._candidates[ordinal] = candidate
        self._checks[ordinal] = (valid, reason)
        self._events.append(
            {
                "event": "selection",
                "ordinal": ordinal,
                "raw_ordinal": raw,
                "candidate_id": candidate.candidate_id,
                "selection": selection,
                "origin": origin,
                "cells": cells,
                "screened": valid,
            }
        )
        self._checkpoint_progress()
        return candidate

    def validate(self, ordinal):
        from .contracts import ValidationCheck, ValidationResult

        ok, reason = self._checks[ordinal]
        return ValidationResult(
            (
                ValidationCheck(
                    "scene_sampling.proposal", "passed" if ok else "failed", reason
                ),
            )
        )

    @staticmethod
    def _region(cells):
        return tuple((cell[0], *(axis // 2 for axis in cell[1:])) for cell in cells)

    def finish(self, ordinal, episode, receipt, *, executed):
        if ordinal in self._outcomes:
            return
        cells = self._predicted.pop(ordinal, ())
        success = episode is not None and episode.validation.accepted
        if executed:
            s, n = self._stats.get(self._region(cells), (0, 0))
            self._stats[self._region(cells)] = (s + int(success), n + 1)
        result = {
            "elapsed_s": self.budget.clock() - self.budget.started,
            "proposals": self.budget.counts["candidates"],
            "rollouts": self.budget.counts["rollouts"],
            "event": "outcome",
            "ordinal": ordinal,
            "executed": executed,
            "physical_accepted": success,
            "committed": receipt is not None and receipt.confirmed,
            "coverage": ({} if self.coverage is None else self.coverage.phase_counts()),
        }
        if episode is not None and "full_joint_positions" in episode.observations:
            states = episode.observations["full_joint_positions"]
            segments = episode.metadata.get("program_segments", ())
            pick_end = (
                min(len(states) - 1, int(segments[0]["end_step"])) if segments else 0
            )
            ids = self.env.robot.control_parts[self.cfg["control_part"]]
            if ids and isinstance(ids[0], str):
                ids = [self.env.robot.joint_names.index(name) for name in ids]
            row = int(episode.metadata["source_env_id"])
            values = states[[pick_end, len(states) - 1]][:, ids].to(
                self.env.robot.device
            )[None]
            tcp = self.env.robot.compute_batch_fk(
                values, name=self.cfg["control_part"], env_ids=[row], to_matrix=True
            )[0]
            result["measured_tcp_poses"] = tcp.detach().cpu().tolist()
            result["measured_tcp_indices"] = [pick_end, len(states) - 1]
        self._outcomes[ordinal] = result
        self._events.append(result)
        self._checkpoint_progress()

    def state_dict(self):
        from dataclasses import asdict

        return deepcopy(
            {
                "version": 1,
                "seed": self.seed,
                "revision": self.request_id,
                "options": self.options,
                "coverage": (
                    None if self.coverage is None else self.coverage.state_dict()
                ),
                "raw_ordinal": self._raw_ordinal,
                "candidates": {
                    str(k): {
                        "request_id": v.request_id,
                        "scene": asdict(v.scene),
                        "deployment": {"poses": dict(v.deployment.poses)},
                        "mode_id": v.mode_id,
                        "parameters": v.parameters,
                    }
                    for k, v in self._candidates.items()
                },
                "checks": self._checks,
                "predicted": self._predicted,
                "outcomes": self._outcomes,
                "events": self._events,
                "statistics": [
                    [cells, list(counts)] for cells, counts in self._stats.items()
                ],
                "budget": dict(self.budget.counts),
                "elapsed_s": self.budget.clock() - self.budget.started,
                "blocked": self.budget._blocked,
            }
        )

    def load_state_dict(self, state):
        """Restore deterministic proposal history; never promote pending writes."""
        from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
        from embodichain.lab.sim.scene.feasibility import (
            JointSceneCandidate,
            RobotDeploymentCandidate,
        )

        data = deepcopy(state)
        if (
            self._candidates
            or data.get("version") != 1
            or data.get("seed") != self.seed
            or data.get("revision") != self.request_id
            or data.get("options") != self.options
        ):
            raise ValueError("Sampling restore requires an empty, matching source")
        candidates = {}
        for key, value in data["candidates"].items():
            scene = dict(value["scene"])
            scene["pose_changes"] = tuple(
                ScenePoseChange(**change) for change in scene["pose_changes"]
            )
            candidates[int(key)] = JointSceneCandidate(
                value["request_id"],
                SceneVariant(**scene),
                RobotDeploymentCandidate(**value["deployment"]),
                value["mode_id"],
                tuple(value["parameters"]),
            )
        if (
            any(not 0 <= k < self.options["engine"]["max_attempts"] for k in candidates)
            or not 0 <= data["raw_ordinal"] <= self.cfg["max_candidates"]
        ):
            raise ValueError("Sampling checkpoint exceeds declared bounds")
        budget = data["budget"]
        limits = {
            "candidates": self.cfg["max_candidates"],
            "ik": self.cfg["max_ik_calls"],
            "planning": self.cfg["max_planning_calls"],
            "rollouts": self.cfg["max_rollouts"],
        }
        if (
            set(budget) != set(limits)
            or any(
                type(v) is not int or not 0 <= v <= limits[k] for k, v in budget.items()
            )
            or not math.isfinite(data["elapsed_s"])
            or data["elapsed_s"] < 0
        ):
            raise ValueError("Malformed sampling budget checkpoint")
        if self.coverage is not None:
            self.coverage.load_state_dict(data["coverage"])
        elif data["coverage"] is not None:
            raise ValueError("Checkpoint unexpectedly contains pose coverage")
        self._candidates = candidates
        self._checks = {int(k): tuple(v) for k, v in data["checks"].items()}
        self._predicted = {
            int(k): tuple(tuple(c) for c in v) for k, v in data["predicted"].items()
        }
        self._outcomes = {int(k): v for k, v in data["outcomes"].items()}
        self._events = data["events"]
        self._next_checkpoint_events = max(1, 2 * len(self._events))
        self._stats = {
            tuple(tuple(c) for c in cells): tuple(counts)
            for cells, counts in data["statistics"]
        }
        self._raw_ordinal = data["raw_ordinal"]
        self.budget.counts = budget
        self.budget.started = self.budget.clock() - data["elapsed_s"]
        self.budget.deadline = self.budget.started + self.budget.cfg.max_wall_time_s
        self.budget._blocked = (
            None if data["blocked"] is None else tuple(data["blocked"])
        )

    def _checkpoint_progress(self) -> None:
        # Whole-history diagnostic checkpoints grow geometrically. Explicit
        # checkpoints at workload completion still write the complete history.
        count = len(self._events)
        if count >= self._next_checkpoint_events:
            self.checkpoint()
            self._next_checkpoint_events = 2 * count

    def checkpoint(self):
        if self.output is None:
            return
        from embodichain.data_pipeline.recording.media import _atomic_json

        _atomic_json(
            self.output / "pose-sampling.json",
            {
                "strategy": self.cfg["strategy"],
                "policy": self.schedule["policy"],
                "domain_id": self.domain_id,
                "budget": dict(self.budget.counts),
                "events": self._events,
                "coverage": (
                    None if self.coverage is None else self.coverage.state_dict()
                ),
                "checkpoint": self.state_dict(),
            },
        )
