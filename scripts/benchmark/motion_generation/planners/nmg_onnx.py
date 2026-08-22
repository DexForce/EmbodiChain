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

import importlib.util
from pathlib import Path

from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
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

    capabilities = frozenset({"eef_waypoint", "empty_world", "atomic_action"})
    model_revision = "nmg-onnx-v1"

    def __init__(self, spec: PlannerSpecCfg, context: PlannerContext) -> None:
        super().__init__(spec, context)
        self.motion_generator: MotionGenerator | None = None

    def availability(self) -> tuple[bool, str | None]:
        """Require ONNX Runtime and a resolved standalone policy path."""
        if importlib.util.find_spec("onnxruntime") is None:
            return False, "onnxruntime is not installed."
        model_path = self.spec.config.get("onnx_model_path")
        if not model_path:
            return False, "NMG requires config.onnx_model_path or --nmg-onnx-path."
        path = Path(str(model_path)).expanduser()
        if not path.is_file():
            return False, f"NMG ONNX policy does not exist: {path}."
        if path.suffix.lower() != ".onnx":
            return False, f"NMG requires a standalone .onnx policy, got {path}."
        return True, None

    @property
    def motion_policy_planner(self) -> str:
        """Select EmbodiChain's neural MotionGenerator backend."""
        return "neural"

    def build(self) -> None:
        """Construct the ONNX-backed MotionGenerator."""
        values = self.spec.config
        model_path = str(Path(str(values["onnx_model_path"])).expanduser())
        providers = values.get("onnx_providers")
        planner_cfg = NeuralPlannerCfg(
            robot_uid=self.context.robot.uid,
            onnx_model_path=model_path,
            control_part=self.context.control_part,
            max_steps=int(values.get("max_steps", 240)),
            action_scale=float(values.get("action_scale", 0.2)),
            num_arm_joints=int(values.get("num_arm_joints", 7)),
            num_waypoints=int(values.get("num_waypoints", 8)),
            use_relative_obs=bool(values.get("use_relative_obs", True)),
            intermediate_orientation=bool(values.get("intermediate_orientation", True)),
            pos_eps=float(values.get("pos_eps", 0.01)),
            rot_eps=float(values.get("rot_eps", 0.1)),
            onnx_providers=list(providers) if providers is not None else None,
            policy_frame_from_world=values.get("policy_frame_from_world"),
            runtime_tcp_from_policy_tcp=values.get("runtime_tcp_from_policy_tcp"),
            dt=float(values.get("dt", 0.01)),
        )
        self.motion_generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))

    def plan(self, case: BenchmarkCase) -> PlanResult:
        """Plan all ordered Cartesian constraints in one closed-loop rollout."""
        if self.motion_generator is None:
            raise RuntimeError("NMG adapter must be built before plan().")
        targets = [
            PlanState.from_xpos(case.target_waypoints[:, index])
            for index in range(case.num_waypoints)
        ]
        return self.motion_generator.generate(
            targets,
            MotionGenOptions(
                start_qpos=case.start_qpos,
                control_part=self.context.control_part,
                plan_opts=NeuralPlanOptions(),
            ),
        )

    def close(self) -> None:
        """Release the adapter-owned MotionGenerator."""
        self._close_motion_generator()


register_planner_adapter("nmg_onnx", NmgOnnxAdapter)
