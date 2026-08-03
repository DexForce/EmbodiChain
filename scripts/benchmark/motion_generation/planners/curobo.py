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

from embodichain.lab.sim.planners import (
    CuroboAutoGenCfg,
    CuroboPlanOptions,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    PlanResult,
    PlanState,
)

from ..config import PlannerSpecCfg
from ..models import BenchmarkCase
from ..registry import register_planner_adapter
from .base import PlannerAdapter, PlannerContext

__all__ = ["CuroboAdapter"]


class CuroboAdapter(PlannerAdapter):
    """Run cuRobo with a frozen, empty-world operational configuration."""

    capabilities = frozenset({"eef_waypoint", "batched", "empty_world"})
    model_revision = "curobo-v2"
    separate_prepare = True

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None

    def availability(self) -> tuple[bool, str | None]:
        """Require both CUDA and the optional cuRobo runtime."""
        if not torch.cuda.is_available():
            return False, "cuRobo requires CUDA, but CUDA is unavailable."
        if importlib.util.find_spec("curobo") is None:
            return False, "cuRobo is not installed; install one of the cuRobo extras."
        return True, None

    def build(self) -> None:
        """Construct MotionGenerator without materializing its lazy backend."""
        values = self.spec.config
        world_values = dict(values.get("world", {}))
        auto_values = dict(values.get("auto_gen", {}))
        if bool(world_values.get("multi_env", False)):
            raise ValueError(
                "free-space-common requires one shared empty cuRobo world "
                "with world.multi_env=false."
            )
        world = CuroboWorldCfg(
            rigid_objects=None,
            obstacle_representation=str(
                world_values.get("obstacle_representation", "sphere")
            ),
            collision_cache=dict(
                world_values.get("collision_cache", {"cuboid": 8, "mesh": 2})
            ),
            dynamic_obstacle_names=[],
            multi_env=False,
        )
        planner_cfg = CuroboPlannerCfg(
            robot_uid=self.context.robot.uid,
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

    def prepare(self, case: BenchmarkCase) -> dict[str, object]:
        """Materialize and warm the EEF backend without consuming a real case."""
        if self.motion_generator is None:
            raise RuntimeError("cuRobo adapter must be built before prepare().")
        planner = self.motion_generator.planner
        return planner.prepare_backend(
            control_part=self.context.control_part,
            batch_size=case.batch_size,
            move_type=MoveType.EEF_MOVE,
        )

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan all ordered EEF waypoints in one MotionGenerator call."""
        if self.motion_generator is None:
            raise RuntimeError("cuRobo adapter must be built before plan().")
        targets = [
            PlanState.from_xpos(case.target_waypoints[:, index])
            for index in range(case.num_waypoints)
        ]
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                plan_opts=CuroboPlanOptions(
                    start_qpos=case.start_qpos,
                    control_part=self.context.control_part,
                ),
            ),
        )

    def close(self) -> None:
        """Destroy cached cuRobo graph and planner resources."""
        self._close_motion_generator()


register_planner_adapter("curobo", CuroboAdapter)
