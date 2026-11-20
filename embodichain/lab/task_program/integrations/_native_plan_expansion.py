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

"""Native measured-plan branch of the shared same-grid expansion factory."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

__all__: list[str] = []


def _fit_joint_residual(
    reference, candidate, *, active, joint_limits, minimum_delta, checks
):
    """Bound a sampled residual without changing time, endpoints or validators.

    Try the original draw first, then at most eight halvings. No new random
    draws or smaller-than-declared variants are admitted. Each materialized
    candidate is checked in its actual storage dtype before it can leave here.
    """
    delta = candidate.positions - reference.positions
    for refinement in range(9):
        scale = 0.5**refinement
        fitted = (
            replace(candidate)
            if refinement == 0
            else replace(candidate, positions=reference.positions + scale * delta)
        )
        magnitude = float((fitted.positions - reference.positions).abs().max())
        if magnitude < minimum_delta:
            break
        values = fitted.positions[:, active]
        bounds = joint_limits[active]
        if bool(((values < bounds[:, 0]) | (values > bounds[:, 1])).any()):
            continue
        result = checks(fitted)
        if all(check.accepted for check in result):
            return fitted, result, scale
    return None


@dataclass(frozen=True)
class TaskProgramNativeExpansionRecord:
    """Owned measured-plan expansion provenance and hashed numeric attachment."""

    env_id: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        from embodichain.utils._owned_json import freeze_metadata

        object.__setattr__(self, "payload", freeze_metadata(self.payload))

    def to_metadata(self) -> dict[str, Any]:
        from embodichain.lab.gym.envs._json import json_safe_copy

        return {
            "env_id": self.env_id,
            **json_safe_copy(self.payload, field_name="expansion record"),
        }


class _NativePlanExpansion:
    """Select native grasp branches or adapt explicitly editable free phases."""

    def __init__(
        self,
        operator: str,
        options: dict,
        *,
        seed: int,
        scene_case,
        revision: str,
        output: Path | None = None,
        episode_id: int = 0,
        affordance_branch: int | None = None,
        preferred_local_poses: tuple[tuple[str, tuple[float, ...]], ...] = (),
    ):
        if operator not in (
            "object_pose",
            "affordance",
            "joint_residual",
            "via_points",
            "nullspace_residual",
        ):
            raise ValueError("unsupported native-plan expansion operator")
        self.operator = operator
        from embodichain.lab.gym.envs._json import json_safe_copy

        self.options = json_safe_copy(options, field_name="same-grid options")
        self.seed = seed
        self.scene_case, self.revision = scene_case, revision
        self.output = output
        self.episode_id = episode_id
        if affordance_branch is not None and (
            operator != "affordance"
            or type(affordance_branch) is not int
            or not 1 <= affordance_branch < self.options["affordance"]["branches"]
        ):
            raise ValueError(
                "affordance_branch must select a non-nominal configured branch"
            )
        self.affordance_branch = affordance_branch
        if preferred_local_poses and operator != "affordance":
            raise ValueError("preferred_local_poses requires the affordance operator")
        self.preferred_local_poses = preferred_local_poses
        self._records: list[TaskProgramNativeExpansionRecord] = []

    @property
    def records(self) -> tuple[TaskProgramNativeExpansionRecord, ...]:
        return tuple(self._records)

    def create_plan_transform(self, request, *, engine):
        if self.operator == "object_pose":
            return None
        if self.operator == "affordance":
            if (
                request.analysis_call_index
                not in self.options["affordance"]["call_indices"]
            ):
                return None

            def record(resolved, context, plan):
                return self._record_affordance(request, context, plan)

            return record

        calls = self.options.get("call_indices")
        if calls is not None and request.analysis_call_index not in calls:
            return None

        def transform(resolved, context, plan):
            return self._transform(request, engine, resolved, context, plan)

        return transform

    def prepare_planning_context(self, request, context, *, engine):
        """Select a non-nominal native Affordance branch before planning."""
        if self.operator != "affordance":
            return context
        settings = self.options["affordance"]
        if request.analysis_call_index not in settings["call_indices"]:
            return context
        if context.batch_size != 1:
            raise ValueError("Native affordance expansion requires batch_size=1")
        from embodichain.lab.sim.atomic_actions.affordance_sampling import (
            AffordanceSamplingContext,
        )

        return replace(
            context,
            affordance_sampling=AffordanceSamplingContext(
                count=settings["branches"],
                seed=self.seed,
                episode_id=self.episode_id,
                preferred_local_poses=self.preferred_local_poses,
                branch_override=(
                    self.affordance_branch
                    if self.affordance_branch is not None
                    else 1 + self.episode_id % (settings["branches"] - 1)
                ),
            ),
        )

    def _record_affordance(self, call, context, plan):
        """Require distinct native grasp geometry, retaining physical validation."""
        if not bool(plan.plan_success.all()):
            return plan
        samples = plan.diagnostics.metadata.get("affordance_sample", {})
        sampling = context.affordance_sampling
        branch = sampling.branch_override
        distinct = any(
            sample.get("success") == [True]
            and sample.get("unique_candidate_counts", [0])[0] > branch
            and sample.get("reused") == [False]
            for sample in samples.values()
        )
        if not distinct:
            raise ValueError("affordance produced no distinct feasible grasp branch")
        evidence = {
            "operator": "affordance",
            "skill": plan.skill_id,
            "workflow_call_index": call.analysis_call_index,
            "sampling": sampling.metadata(),
            "samples": samples,
            "mode": "native_affordance_replanning",
        }
        if self.output is not None:
            import numpy as np
            from embodichain.lab.expansion.persistence import _digest as file_digest

            self.output.mkdir(parents=True, exist_ok=True)
            path = (
                self.output
                / f"call-{call.analysis_call_index}-{len(self._records)}.npz"
            )
            with path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    candidate=plan.joint_trajectory.positions[0].detach().cpu().numpy(),
                    dt=plan.joint_trajectory.dt[0].detach().cpu().numpy(),
                )
            evidence["plan_artifact"] = {"path": str(path), "sha256": file_digest(path)}
        metadata = dict(plan.diagnostics.metadata, expansion=evidence)
        self._records.append(
            TaskProgramNativeExpansionRecord(int(context.env_ids[0]), evidence)
        )
        return replace(plan, diagnostics=replace(plan.diagnostics, metadata=metadata))

    def _transform(self, call, engine, request, context, plan):
        import torch
        from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory
        from embodichain.lab.sim.atomic_actions.trajectory_adapter import (
            ActionPlanTemplateAdapter,
        )
        from embodichain.lab.sim.motion.expansion import SourceContext
        from embodichain.lab.sim.motion.expansion.operators import (
            ProposalRejected,
            joint_residual,
            nullspace_residual,
            validate_motion_samples,
            via_points,
        )

        free = set(self.options["free_phases"].get(plan.skill_id, ()))
        if (
            not free
            or plan.joint_trajectory is None
            or not bool(plan.plan_success.all())
        ):
            return plan
        if context.batch_size != 1:
            raise ValueError("Native same-grid transforms require batch_size=1")
        endpoints = tuple(
            e for e in request.binding.endpoints if e.endpoint_id == "motion"
        )
        motion_ids = tuple(dict.fromkeys(i for e in endpoints for i in e.joint_ids))
        permissions = {
            segment.name: (self.operator,) if segment.name in free else ()
            for segment in plan.segments
        }
        adapter = ActionPlanTemplateAdapter(
            tuple(engine.robot.joint_names),
            permissions,
            {
                name: "free" if value else "contact"
                for name, value in permissions.items()
            },
            controlled_joint_indices=motion_ids,
        )
        source = SourceContext(
            self.scene_case.task_id,
            self.revision,
            f"{call.workflow_id}:{call.analysis_call_index}",
            self.scene_case,
            context.require_control_dt(),
        )
        template = adapter.export_template(plan, context=source)
        reference_phases = template.phases
        # Keep the first/last two samples, including discrete derivative stencils.
        phases = tuple(
            (
                replace(p, start_index=p.start_index + 1, stop_index=p.stop_index - 1)
                if p.allowed_operators and p.stop_index - p.start_index >= 5
                else replace(p, allowed_operators=())
            )
            for p in template.phases
        )
        bridge = plan.diagnostics.metadata.get("reference_bridge")
        if bridge:
            from embodichain.lab.sim.atomic_skills._expansion_phases import (
                _canonical_samples,
                _free_joint_indices,
            )

            trajectory = plan.joint_trajectory
            _, _, _, canonical = _canonical_samples(
                trajectory.positions[0], trajectory.dt[0]
            )
            # Engine protects raw [0, prefix+3). Operators edit strictly inside
            # phase endpoints, so retain the corresponding last canonical index.
            last_raw = min(
                int(bridge["prefix_samples"]) + 2, trajectory.waypoint_count - 1
            )
            protected_end = int(canonical[last_raw])
            phases = tuple(
                (
                    (
                        replace(p, start_index=max(p.start_index, protected_end))
                        if p.stop_index - max(p.start_index, protected_end) >= 3
                        else replace(p, allowed_operators=())
                    )
                    if p.allowed_operators
                    else p
                )
                for p in phases
            )
            active_ids = _free_joint_indices(
                template.positions, phases, template.controlled_joint_indices
            )
            if not active_ids:
                return plan
            template = replace(template, controlled_joint_indices=active_ids)
        if not any(p.allowed_operators for p in phases):
            return plan
        template = replace(
            template,
            phases=phases,
            allowed_operators=tuple(
                dict.fromkeys(op for p in phases for op in p.allowed_operators)
            ),
        )
        limits = engine.robot.get_qpos_limits().detach().cpu()
        if limits.ndim == 3:
            limits = limits[int(context.env_ids[0])]
        active = list(template.controlled_joint_indices)
        velocity = engine.robot.get_qvel_limits().detach().cpu()
        if velocity.ndim == 2:
            velocity = velocity[int(context.env_ids[0])]
        velocity = velocity[active].clamp(max=self.options["velocity_limit"])
        acceleration = torch.full_like(velocity, self.options["acceleration_limit"])
        seed = int.from_bytes(
            hashlib.sha256(
                f"{self.seed}/{source.unit_id}/{self.operator}".encode()
            ).digest()[:8],
            "little",
        ) % (2**63)
        generator = torch.Generator().manual_seed(seed)
        jacobians, kinematics = self._kinematics(engine.robot, endpoints, template)

        def motion_checks(trajectory):
            checks = []
            for phase in phases:
                if not phase.allowed_operators:
                    continue
                # Operators keep their endpoints exact. The narrowed phase
                # leaves two original samples at each boundary, covering every
                # second difference affected by an edit.
                start, stop = max(0, phase.start_index - 1), min(
                    len(trajectory.dt), phase.stop_index + 1
                )
                intervals = trajectory.dt[start:stop].clone()
                intervals[0] = 0
                checks.append(
                    validate_motion_samples(
                        trajectory.positions[start:stop, active],
                        intervals,
                        velocity_limits=velocity,
                        acceleration_limits=acceleration,
                    )
                )
            return checks

        reference_checks = motion_checks(template)
        if any(not check.accepted for check in reference_checks):
            raise ValueError(
                f"Reference free phases already exceed sampled motion bounds: "
                f"{reference_checks}"
            )
        reason = "no distinct candidate"
        best_rejected = None
        can_scale_residual = self.operator in ("joint_residual", "via_points")
        budget = self.options["max_proposals"]
        for iteration in range(budget + 1):
            residual_scale = 1.0
            if iteration == budget:
                # Preserve the existing random draws and full-amplitude preference.
                # Only budget exhaustion activates this bounded deterministic fallback.
                if not can_scale_residual or best_rejected is None:
                    break
                _, attempt, sampled, raw_checks = best_rejected
                fitted = _fit_joint_residual(
                    template,
                    sampled,
                    active=active,
                    joint_limits=limits,
                    minimum_delta=self.options["minimum_joint_delta"],
                    checks=motion_checks,
                )
                if fitted is None:
                    reason = f"best motion rejection at proposal {attempt}: {raw_checks}; bounded refinement cannot meet unchanged limits and minimum delta"
                    break
                candidate, checks, residual_scale = fitted
                sampled_positions = sampled.positions
                delta = float((candidate.positions - template.positions).abs().max())
            else:
                attempt = iteration
                arguments = dict(
                    joint_limits=limits,
                    normalized_scale=self.options["joint_offset_scale"],
                    generator=generator,
                )
                try:
                    if self.operator == "joint_residual":
                        candidate = joint_residual(template, **arguments)
                    elif self.operator == "via_points":
                        candidate = via_points(
                            template, via_count=self.options["via_count"], **arguments
                        )
                    elif self.operator == "nullspace_residual":
                        candidate = nullspace_residual(
                            template, task_jacobians=jacobians, **arguments
                        )
                    else:
                        raise ValueError(f"Unsupported operator: {self.operator}")
                except ProposalRejected as exc:
                    reason = str(exc)
                    continue
                delta = float((candidate.positions - template.positions).abs().max())
                if delta < self.options["minimum_joint_delta"]:
                    continue
                sampled_positions = candidate.positions
                checks = motion_checks(candidate)
                raw_checks = checks
                if any(not check.accepted for check in checks):
                    reason = f"edited-phase motion limits: {checks}"
                    if can_scale_residual:
                        score = max(
                            value
                            for report in checks
                            for check in report.checks
                            for value in check.metrics.values()
                        )
                        if best_rejected is None or score < best_rejected[0]:
                            best_rejected = (score, attempt, candidate, checks)
                    continue
            fk = self._check_fk(template, candidate, kinematics)
            if self.operator == "nullspace_residual" and (
                fk["max_translation_m"]
                > self.options["nullspace_translation_tolerance_m"]
                or fk["max_rotation_rad"]
                > self.options["nullspace_rotation_tolerance_rad"]
            ):
                reason = f"nullspace FK error: {fk}"
                continue
            positions = adapter.restore_positions(plan, candidate.positions).unsqueeze(
                0
            )
            metadata = dict(plan.diagnostics.metadata)
            evidence = {
                "operator": self.operator,
                "skill": plan.skill_id,
                "workflow_call_index": call.analysis_call_index,
                "phases": [p.phase_id for p in phases if p.allowed_operators],
                "reference_phases": [
                    {"name": p.phase_id, "start": p.start_index, "stop": p.stop_index}
                    for p in reference_phases
                ],
                "controlled_joint_names": [template.joint_names[i] for i in active],
                "task_rows": self.options["nullspace_task_rows"],
                "max_joint_delta": delta,
                "proposal": attempt,
                "residual_scale": residual_scale,
                "raw_proposal_count": min(iteration + 1, budget),
                "raw_motion_checks": [
                    dict(c.metrics) for r in raw_checks for c in r.checks
                ],
                "same_time_grid": True,
                "motion_validation_scope": "edited_phases_and_boundary_stencils",
                "motion": {
                    "reference": [
                        dict(c.metrics) for r in reference_checks for c in r.checks
                    ],
                    "candidate": [dict(c.metrics) for r in checks for c in r.checks],
                },
                "fk": fk,
            }
            if self.output is not None:
                import numpy as np
                from embodichain.lab.expansion.persistence import _digest as file_digest

                self.output.mkdir(parents=True, exist_ok=True)
                path = (
                    self.output
                    / f"call-{call.analysis_call_index}-{len(self._records)}.npz"
                )
                editable = torch.zeros_like(template.positions, dtype=torch.bool)
                for phase in phases:
                    if phase.allowed_operators:
                        editable[phase.start_index : phase.stop_index, active] = True
                with path.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        reference=template.positions.numpy(),
                        candidate=candidate.positions.numpy(),
                        dt=template.dt.numpy(),
                        editable=editable.numpy(),
                        **(
                            {"raw_candidate": sampled_positions.numpy()}
                            if residual_scale < 1
                            else {}
                        ),
                    )
                evidence["plan_artifact"] = {
                    "path": str(path),
                    "sha256": file_digest(path),
                }
            metadata.update(
                expansion=evidence,
                collision_evidence="unverified",
                collision_scope="not_checked",
            )
            rebuilt = engine.rebuild_plan_from_trajectory(
                request,
                context,
                replace(plan, diagnostics=replace(plan.diagnostics, metadata=metadata)),
                TimedTrajectory.from_positions(
                    positions,
                    env_ids=context.env_ids,
                    dt=plan.joint_trajectory.dt.clone(),
                ),
            )
            self._records.append(
                TaskProgramNativeExpansionRecord(int(context.env_ids[0]), evidence)
            )
            return rebuilt
        raise ValueError(
            f"{self.operator} produced no qualified {plan.skill_id} variant: {reason}"
        )

    def _kinematics(self, robot, endpoints, template):
        import torch

        active = list(template.controlled_joint_indices)
        blocks, models = [], []
        for endpoint in endpoints:
            indices = list(endpoint.joint_ids)
            if not set(indices) & set(active):
                continue
            solver = robot.get_solver(endpoint.target.control_part)
            if solver is None or solver.pk_serial_chain is None:
                raise ValueError(
                    "Trajectory augmentation requires the actual robot kinematic chain"
                )
            chain_ids = [
                robot.joint_names.index(n)
                for n in solver.pk_serial_chain.get_joint_parameter_names()
            ]
            qpos = template.positions[:, chain_ids]
            rows = self.options["nullspace_task_rows"]
            jacobian = (
                solver.get_jacobian(
                    qpos,
                    locations=torch.as_tensor(solver.tcp_xpos)[:3, 3],
                )
                .detach()
                .cpu()[:, rows, :]
            )
            block = torch.zeros(len(qpos), len(rows), len(active), dtype=jacobian.dtype)
            for column, joint in enumerate(chain_ids):
                if joint in active:
                    block[:, :, active.index(joint)] = jacobian[:, :, column]
            blocks.append(block)
            models.append((solver, chain_ids))
        if not blocks:
            raise ValueError("No kinematic chain covers the edited motion joints")
        return torch.cat(blocks, dim=1), models

    @staticmethod
    def _check_fk(reference, candidate, models):
        import torch

        translation, rotation = 0.0, 0.0
        for solver, indices in models:
            a = solver.get_fk(reference.positions[:, indices]).detach().cpu().double()
            b = solver.get_fk(candidate.positions[:, indices]).detach().cpu().double()
            translation = max(
                translation, float((a[:, :3, 3] - b[:, :3, 3]).norm(dim=-1).max())
            )
            relative = a[:, :3, :3].transpose(-1, -2) @ b[:, :3, :3]
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
                -1, 1
            )
            rotation = max(rotation, float(torch.acos(cosine).max()))
        return {"max_translation_m": translation, "max_rotation_rad": rotation}
