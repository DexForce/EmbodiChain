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

"""cuRobo primary-baseline adapter for an empty collision world."""

from __future__ import annotations

import importlib.util

import torch

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    CuroboAutoGenCfg,
    CuroboPlanOptions,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MoveType,
    PlanResult,
)

from ..config import PlannerSpecCfg
from ..models import BenchmarkCase
from ..limits import resolve_dynamic_limit, resolve_robot_velocity_limits
from ..registry import register_planner_adapter
from .base import PlannerAdapter, PlannerContext
from ._constraints import _case_modalities, _case_targets

__all__ = ["CuroboAdapter"]


class CuroboAdapter(PlannerAdapter):
    """Run cuRobo with a frozen, empty-world operational configuration."""

    configuration_features = frozenset({"masked_constraints"})

    capabilities = frozenset(
        {
            "eef_waypoint",
            "joint_waypoint",
            "batched",
            "empty_world",
            "atomic_action",
        }
    )
    model_revision = "curobo-v2"
    separate_prepare = True
    timing_is_solved = True
    """cuRobo optimizes the trajectory's timing under its own dynamic limits."""

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None
        if spec.config.get("masked_constraints", False):
            self.capabilities = self.capabilities | {"masked_constraints"}

    def availability(self) -> tuple[bool, str | None]:
        """Require both CUDA and the optional cuRobo runtime."""
        if not torch.cuda.is_available():
            return False, "cuRobo requires CUDA, but CUDA is unavailable."
        if importlib.util.find_spec("curobo") is None:
            return False, "cuRobo is not installed; install one of the cuRobo extras."
        return True, None

    def _backend_motion_limits(self) -> dict[str, list[float]]:
        """Map common control-part budgets into cuRobo's active-joint order."""
        limits = self.resolve_motion_limits(require_timing=True)
        robot = self.context.robot
        active_ids = list(robot.active_joint_ids)
        from embodichain.lab.sim.motion.planners.curobo.curobo_yaml import (
            _curobo_joint_limits,
        )

        defaults = _curobo_joint_limits(robot)
        # Explicit requests are positive; the production exporter intersects
        # these with Robot caps again, retaining zero caps on locked joints.
        velocity = [value if value > 0 else 0.2 for value in defaults["max_velocity"]]
        acceleration = defaults["max_acceleration"]
        # Atomic dual-arm calls share one generator but switch motion parts.
        # Apply the suite budget to every configured kinematic chain, retaining
        # each chain's own effective Robot speed caps.
        solver_parts = getattr(getattr(robot, "cfg", None), "solver_cfg", {}) or {}
        parts = list(dict.fromkeys([self.context.control_part, *solver_parts]))
        for part in parts:
            controlled_ids = list(robot.get_joint_ids(part))
            if any(index not in active_ids for index in controlled_ids):
                raise ValueError(
                    "cuRobo control parts must contain only active joints."
                )
            if part == self.context.control_part:
                part_limits = limits
            else:
                reference = torch.empty(0, dtype=torch.float64)
                speed = resolve_robot_velocity_limits(
                    robot,
                    part,
                    self.context.joint_velocity_limit_rad_s,
                    reference,
                    dof=len(controlled_ids),
                )
                accel = resolve_dynamic_limit(
                    self.context.joint_acceleration_limit_rad_s2,
                    len(controlled_ids),
                    reference,
                )
                if speed is None or accel is None:
                    raise ValueError(
                        f"Timed planning requires motion limits for {part!r}."
                    )
                part_limits = {
                    "velocity": speed.tolist(),
                    "acceleration": accel.tolist(),
                }
            for column, joint_id in enumerate(controlled_ids):
                speed = part_limits["velocity"][column]
                if speed <= 0:
                    raise ValueError(
                        "cuRobo cannot plan a joint with a zero velocity limit."
                    )
                index = active_ids.index(joint_id)
                velocity[index] = speed
                acceleration[index] = part_limits["acceleration"][column]
        for key, actual in (
            ("max_velocity", velocity),
            ("max_acceleration", acceleration),
        ):
            if key in self.spec.config:
                requested = torch.as_tensor(self.spec.config[key], dtype=torch.float64)
                if not torch.allclose(
                    requested, torch.tensor(actual, dtype=torch.float64)
                ):
                    raise ValueError(
                        f"Planner config.{key} differs from the suite motion budget."
                    )
        return {"max_velocity": velocity, "max_acceleration": acceleration}

    def build(self) -> None:
        """Construct MotionGenerator without materializing its lazy backend."""
        limits = self._backend_motion_limits()
        values = self.spec.config
        world_values = dict(values.get("world", {}))
        auto_values = dict(values.get("auto_gen", {}))
        if bool(world_values.get("multi_env", False)):
            raise ValueError(
                "The current cuRobo benchmark adapter requires one shared empty world "
                "with world.multi_env=false."
            )
        world = CuroboWorldCfg(
            rigid_objects=None,
            voxel_size=float(world_values.get("voxel_size", 0.01)),
            voxel_padding=float(world_values.get("voxel_padding", 0.1)),
            dynamic_obstacle_names=[],
            multi_env=False,
        )
        planner_cfg = CuroboPlannerCfg(
            robot_uid=self.context.robot.uid,
            max_velocity=limits["max_velocity"],
            max_acceleration=limits["max_acceleration"],
            world=world,
            auto_gen=CuroboAutoGenCfg(**auto_values),
            collision_activation_distance=float(
                values.get("collision_activation_distance", 0.01)
            ),
            max_attempts=int(values.get("max_attempts", 5)),
            max_planning_time=values.get("max_planning_time"),
            cuda_device=values.get("cuda_device"),
            use_cuda_graph=bool(values.get("use_cuda_graph", True)),
            cuda_graph_fallback=bool(values.get("cuda_graph_fallback", True)),
            interpolation_dt=float(values.get("interpolation_dt", 0.025)),
            preserve_plan_samples=bool(values.get("preserve_plan_samples", True)),
            warmup_iterations=int(values.get("warmup_iterations", 1)),
        )
        self.motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))

    def _case_move_types(self, case: BenchmarkCase) -> list[MoveType]:
        """Return every backend needed by an ordered, possibly mixed request."""
        if (
            case.case_parameters.get("motion_validity")
            == "ordered_constraint_waypoints"
            and "masked_constraints" not in self.capabilities
        ):
            raise ValueError("Enable config.masked_constraints for masked cases.")
        return list(
            dict.fromkeys(
                MoveType.JOINT_MOVE if value == "joint" else MoveType.EEF_MOVE
                for value in _case_modalities(case)
            )
        )

    def atomic_plan_options(
        self, *, control_dt: float | None = None
    ) -> CuroboPlanOptions:
        """Use the same configured suite limits for preparation and Atomic calls.

        Args:
            control_dt: Host command clock; cuRobo retains native solved timing.

        Returns:
            Options retaining the generator's full active-joint budget vectors.
        """
        return CuroboPlanOptions()

    def _case_control_parts(self, case: BenchmarkCase) -> list[str]:
        parts = case.case_parameters.get("control_parts", {})
        return list(
            dict.fromkeys(
                values["motion"]
                for values in parts.values()
                if isinstance(values, dict) and "motion" in values
            )
        ) or [self.context.control_part]

    def prepare(self, case: BenchmarkCase) -> dict[str, object]:
        """Materialize and warm the backend required by the benchmark case."""
        if self.motion_generator is None:
            raise RuntimeError("cuRobo adapter must be built before prepare().")
        planner = self.motion_generator.planner
        prepared = [
            planner.prepare_backend(
                control_part=part,
                batch_size=case.batch_size,
                move_type=move_type,
            )
            for part in self._case_control_parts(case)
            for move_type in self._case_move_types(case)
        ]
        return prepared[0] if len(prepared) == 1 else {"backends": prepared}

    def prepare_cases(self, cases: list[BenchmarkCase]) -> dict[str, object]:
        """Prepare every distinct batch-size and planning-mode backend."""
        if not cases:
            raise ValueError("prepare_cases requires at least one supported case.")
        if self.motion_generator is None:
            raise RuntimeError("cuRobo adapter must be built before prepare_cases().")
        prepared = []
        seen = set()
        for case in cases:
            for part in self._case_control_parts(case):
                for move_type in self._case_move_types(case):
                    key = (case.batch_size, part, move_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    prepared.append(
                        self.motion_generator.planner.prepare_backend(
                            control_part=part,
                            batch_size=case.batch_size,
                            move_type=move_type,
                        )
                    )
        return {"backends": prepared}

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan all ordered Cartesian or joint waypoints in one call."""
        if self.motion_generator is None:
            raise RuntimeError("cuRobo adapter must be built before plan().")
        self._case_move_types(case)
        targets = _case_targets(case)
        modalities = _case_modalities(case)
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                plan_opts=CuroboPlanOptions(waypoint_modalities=modalities),
            ),
        )

    def close(self) -> None:
        """Destroy cached cuRobo graph and planner resources."""
        self._close_motion_generator()


register_planner_adapter("curobo", CuroboAdapter)
