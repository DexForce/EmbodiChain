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

"""NMG ONNX benchmark adapter."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import torch

from embodichain.extensions.models.policy._nmg_observation import (
    _OnnxObservationDimensions,
)
from embodichain.extensions.models.policy.nmg import _create_ort_session
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    MoveType,
    NeuralPlanOptions,
    NeuralPlannerCfg,
    PlanResult,
    PlanState,
)

from ..config import PlannerSpecCfg
from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter, PlannerContext

__all__ = ["NmgOnnxAdapter"]


class NmgOnnxAdapter(PlannerAdapter):
    """Run a standalone NMG ONNX policy through :class:`NeuralPlanner`."""

    configuration_features = frozenset(
        {
            "policy_frame_from_world_by_control_part",
            "batch_control_parts",
            "onnx_cuda_use_tf32",
            "preserve_waypoint_arrivals",
        }
    )

    capabilities = frozenset(
        {
            "eef_waypoint",
            "masked_constraints",
            "joint_waypoint",
            "batched",
            "empty_world",
            "atomic_skill",
        }
    )

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None

    @property
    def retime(self) -> bool:
        """Whether this row re-parameterizes the rollout under dynamic limits."""
        return bool(self.spec.config.get("retime", False))

    @property
    def timing_is_solved(self) -> bool:
        """Whether reported timing is solved rather than nominal.

        Without retiming the policy integrates a joint delta per step and never
        solves a duration, so its ``dt`` carries no executable timing and the
        row must be reported as N/A. Retiming replaces that constant with a
        solved parameterization, which makes the duration comparable.
        """
        return self.retime

    def _retiming_constraints(self) -> dict[str, object]:
        """Use the shared suite budgets and strictest effective Robot caps."""
        limits = self.resolve_motion_limits(require_timing=True)
        velocity = limits["velocity"]
        dimensions, _ = self._onnx_preflight
        if dimensions is not None and len(velocity) != dimensions.num_controlled_joints:
            raise ValueError(
                "NMG ONNX policy uses "
                f"{dimensions.num_controlled_joints} joints but control part "
                f"{self.context.control_part!r} has {len(velocity)}."
            )
        return {"velocity": velocity, "acceleration": limits["acceleration"]}

    def _resolved_model_revision(self) -> str:
        """Use an explicit revision or derive one from the runtime ONNX path."""
        configured = self.spec.config.get("model_revision")
        if configured:
            return str(configured)
        model_path = self.spec.config.get("onnx_model_path")
        if model_path:
            return Path(str(model_path)).expanduser().stem
        return super()._resolved_model_revision()

    @cached_property
    def _onnx_preflight(self) -> tuple[_OnnxObservationDimensions | None, str | None]:
        """Read model dimensions once before case selection and timed construction."""
        model_path = self.spec.config.get("onnx_model_path")
        if not model_path:
            return None, "NMG requires config.onnx_model_path or --nmg-onnx-path."
        path = Path(str(model_path)).expanduser()
        if not path.is_file():
            return None, f"NMG ONNX policy does not exist: {path}."
        if path.suffix.lower() != ".onnx":
            return None, f"NMG requires a standalone .onnx policy, got {path}."
        try:
            session = _create_ort_session(
                str(path),
                device=self.context.device,
                providers=self.spec.config.get("onnx_providers"),
                cuda_use_tf32=self.spec.config.get("onnx_cuda_use_tf32"),
                intra_op_threads=1,
            )
            metadata = session.get_modelmeta().custom_metadata_map
            return _OnnxObservationDimensions.from_metadata(metadata), None
        except Exception as exc:
            return None, f"Could not read NMG ONNX policy metadata from {path}: {exc}"

    def availability(self) -> tuple[bool, str | None]:
        """Require a readable policy and validate any legacy dimension assertions."""
        dimensions, reason = self._onnx_preflight
        if dimensions is None:
            return False, reason
        joints = self.spec.config.get("num_arm_joints")
        waypoints = self.spec.config.get("num_waypoints")
        try:
            dimensions.check_expectations(
                num_arm_joints=int(joints) if joints is not None else None,
                num_waypoints=int(waypoints) if waypoints is not None else None,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return False, str(exc)
        return True, None

    def supports_case(self, case: BenchmarkCase) -> tuple[bool, str | None]:
        """Reject constraints incompatible with policy capacity or TCP mapping."""
        supported, reason = super().supports_case(case)
        if not supported:
            return supported, reason
        dimensions, _ = self._onnx_preflight
        capacity = (
            dimensions.num_waypoints
            if dimensions is not None
            else self.spec.config.get("num_waypoints")
        )
        if capacity is not None and case.num_waypoints > int(capacity):
            return (
                False,
                f"NMG policy supports at most {capacity} waypoints; case has "
                + f"{case.num_waypoints} and was not truncated or split.",
            )
        modalities = case.case_parameters.get("constraint_modalities", ())
        tcp_transform = self.spec.config.get("runtime_tcp_from_policy_tcp")
        if "position" in modalities and tcp_transform is not None:
            offset = torch.as_tensor(tcp_transform, dtype=torch.float32)[:3, 3]
            if bool(torch.any(offset.abs() > 1.0e-8)):
                return (
                    False,
                    "NMG position-only constraints require the runtime and policy "
                    "TCP origins to coincide.",
                )
        return True, None

    def _case_max_steps(self, case: BenchmarkCase) -> int:
        """Match NMG's training/evaluation budget of 30 steps per waypoint."""
        steps_per_waypoint = int(self.spec.config.get("steps_per_waypoint", 30))
        configured_max = int(self.spec.config.get("max_steps", 150))
        return min(configured_max, int(case.num_waypoints) * steps_per_waypoint)

    @property
    def motion_policy_planner(self) -> str:
        """Select EmbodiChain's neural MotionGenerator backend."""
        return "neural"

    def build(self) -> None:
        """Construct the ONNX-backed MotionGenerator."""
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason)
        self.resolve_motion_limits(require_timing=self.retime)
        values = self.spec.config
        model_path = str(Path(str(values["onnx_model_path"])).expanduser())
        providers = values.get("onnx_providers")
        planner_cfg = NeuralPlannerCfg(
            robot_uid=self.context.robot.uid,
            onnx_model_path=model_path,
            control_part=self.context.control_part,
            batch_control_parts=bool(values.get("batch_control_parts", False)),
            max_steps=int(values.get("max_steps", 150)),
            action_scale=float(values.get("action_scale", 0.2)),
            joint_periodic_features=bool(values.get("joint_periodic_features", False)),
            use_relative_obs=bool(values.get("use_relative_obs", True)),
            canonicalize_quat_obs=bool(values.get("canonicalize_quat_obs", True)),
            intermediate_orientation=bool(values.get("intermediate_orientation", True)),
            pos_eps=float(values.get("pos_eps", 0.01)),
            rot_eps=float(values.get("rot_eps", 0.1)),
            joint_eps=float(values.get("joint_eps", 0.01)),
            onnx_providers=list(providers) if providers is not None else None,
            onnx_cuda_use_tf32=values.get("onnx_cuda_use_tf32"),
            policy_frame_from_world=values.get("policy_frame_from_world"),
            policy_frame_from_world_by_control_part=values.get(
                "policy_frame_from_world_by_control_part"
            ),
            runtime_tcp_from_policy_tcp=values.get("runtime_tcp_from_policy_tcp"),
            dt=float(values.get("dt", 0.01)),
            constraints=self._retiming_constraints() if self.retime else None,
            preserve_waypoint_arrivals=values.get("preserve_waypoint_arrivals", False),
        )
        self.motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
        self.fixed_batch_size = self.motion_generator.planner._policy.fixed_batch_size

    def atomic_plan_options(
        self, *, control_dt: float | None = None
    ) -> NeuralPlanOptions | None:
        """Provide the suite's timing limits for retimed Atomic Skill trials.

        Args:
            control_dt: Host control period, if known. NMG uses its configured
                maximum output sample interval.

        Returns:
            Native options with the resolved suite budgets, or ``None`` for a
            nominal NMG rollout.
        """
        if self.retime:
            return NeuralPlanOptions(constraints=self._retiming_constraints())
        return None

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan all ordered constraints in one closed-loop rollout."""
        if self.motion_generator is None:
            raise RuntimeError("NMG adapter must be built before plan().")
        motion_validity = case.case_parameters.get(
            "motion_validity", "ordered_cartesian_waypoints"
        )
        modalities = None
        if "constraint_modalities" in case.case_parameters:
            modalities = list(case.case_parameters["constraint_modalities"])
            targets = [
                (
                    PlanState.from_qpos(
                        case.reference_qpos[:, index], move_type=MoveType.JOINT_MOVE
                    )
                    if modality == "joint"
                    else PlanState.from_xpos(
                        case.target_waypoints[:, index], move_type=MoveType.EEF_MOVE
                    )
                )
                for index, modality in enumerate(modalities)
            ]
        elif motion_validity == "ordered_joint_waypoints":
            targets = [
                PlanState.from_qpos(
                    case.reference_qpos[:, index], move_type=MoveType.JOINT_MOVE
                )
                for index in range(case.num_waypoints)
            ]
        elif motion_validity == "ordered_cartesian_waypoints":
            targets = [
                PlanState.from_xpos(
                    case.target_waypoints[:, index], move_type=MoveType.EEF_MOVE
                )
                for index in range(case.num_waypoints)
            ]
        else:
            raise ValueError(f"Unsupported motion_validity mode {motion_validity!r}.")
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                plan_opts=NeuralPlanOptions(
                    max_steps=self._case_max_steps(case), waypoint_modalities=modalities
                ),
            ),
        )

    def close(self) -> None:
        """Release the adapter-owned MotionGenerator."""
        self._close_motion_generator()


register_planner_adapter("nmg_onnx", NmgOnnxAdapter)
