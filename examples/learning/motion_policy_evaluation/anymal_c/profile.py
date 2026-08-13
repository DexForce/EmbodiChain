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

"""ANYmal-C velocity Profile for Newton's public TorchScript policy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from dexsim.kit.motion_policy import (
    AdapterRequest,
    EvaluationFrame,
    JointMap,
    PolicyContext,
    PolicyOutput,
    require_finite,
)

from embodichain.learning.rl.motion_policy_evaluation import (
    MotionProfile,
    MotionProfileRequest,
)

__all__ = ["AnymalCVelocityAdapter", "PROFILE_ID", "build_profile"]

PROFILE_ID = "newton-anymal-c-velocity"

_SOURCE_REVISION = "7249270ab41be1c2d4c809aa87536bab3a1a26f4"
_ASSET_REVISION = "261cd1f429619d8ef4f546bd788ab9dea906b5e1"
_ROBOT_PATH = Path("anybotics_anymal_c/urdf/anymal.urdf")
_JOINT_NAMES = (
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
)
_DEFAULT_POSITION = np.asarray(
    (0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8),
    dtype=np.float32,
)


def build_profile(request: MotionProfileRequest) -> MotionProfile:
    """Build the Policy Spec for the public ANYmal-C checkpoint.

    Args:
        request: Checkpoint, resource checkout, device, and renderer.

    Returns:
        A Motion Profile ready for DexSim Motion Policy Kit.
    """
    if request.checkpoint.suffix.lower() != ".pt":
        raise ValueError("The ANYmal-C example requires a TorchScript .pt file")
    if request.resource_root is None:
        raise ValueError(
            "The ANYmal-C example requires --resource-root from prepare_resources.py"
        )
    robot_asset = request.resource_root / _ROBOT_PATH
    if not robot_asset.is_file():
        raise FileNotFoundError(
            f"ANYmal-C robot asset does not exist: {robot_asset}. "
            "Run prepare_resources.py first."
        )

    return MotionProfile(
        profile_id=PROFILE_ID,
        checkpoint=request.checkpoint,
        policy_spec={
            "schema_version": 1,
            "kind": "policy",
            "id": PROFILE_ID,
            "metadata": {
                "title": "Newton ANYmal-C velocity policy",
                "description": "Public 48-D command locomotion TorchScript policy.",
                "status": "example",
                "tags": ["external", "quadruped", "velocity", "torchscript"],
            },
            "robot": {
                "asset": {"path": str(robot_asset)},
                "use_urdf_material": True,
                "initial": {
                    "root_height": 0.76,
                    "joint_positions": {
                        "default": 0.0,
                        "overrides": dict(
                            zip(
                                _JOINT_NAMES,
                                _DEFAULT_POSITION.tolist(),
                                strict=True,
                            )
                        ),
                    },
                },
                "control": {
                    "defaults": {
                        "stiffness": 300.0,
                        "damping": 10.0,
                        "effort_limit": 80.0,
                        "armature": 0.06,
                    },
                },
            },
            "policy": {
                "models": {"actor": {"path": str(request.checkpoint)}},
                "adapter": {
                    "type": "python",
                    "entrypoint": ("anymal_c.profile:AnymalCVelocityAdapter"),
                    "config": {
                        "inference_device": str(request.device),
                        "joint_names": list(_JOINT_NAMES),
                    },
                },
            },
            "evaluation": {
                "initial_command": [0.0, 0.0, 0.0],
                "termination": {"behavior": "pause"},
            },
            "runtime": {
                "physics_dt": 0.005,
                "sim_steps_per_control": 4,
                "physics_backend": "default",
                "simulation_device": "cpu",
                "inference_provider": (
                    "cuda" if request.device.type == "cuda" else "cpu"
                ),
                "renderer": request.renderer,
            },
        },
        provenance={
            "source": "Newton ANYmal-C keyboard policy example",
            "source_revision": _SOURCE_REVISION,
            "source_example": "newton/examples/robot/example_robot_policy.py",
            "asset_revision": _ASSET_REVISION,
            "model_format": "torchscript",
            "observation_size": 48,
            "action_size": 12,
        },
    )


class AnymalCVelocityAdapter:
    """Reproduce the upstream command locomotion observation and action path."""

    command_enabled = True
    command_step = (0.1, 0.05, 0.1)
    command_limits = (1.0, 0.5, 1.0)

    def __init__(self, request: AdapterRequest) -> None:
        config = dict(request.config)
        self.device = torch.device(str(config["inference_device"]))
        self.joint_names = tuple(config["joint_names"])
        if self.joint_names != _JOINT_NAMES:
            raise ValueError("ANYmal-C joint order is incompatible")
        self.checkpoint = request.models["actor"]
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.joints: JointMap | None = None
        self.model: torch.jit.ScriptModule | None = None

    def setup(self, context: PolicyContext) -> None:
        """Load the model and bind the runtime joint order."""
        if context.robot is None:
            raise RuntimeError("ANYmal-C robot description is required")
        self.joints = JointMap.from_joint_names(
            context.robot.joint_names,
            self.joint_names,
        )
        self.model = torch.jit.load(
            str(self.checkpoint),
            map_location=self.device,
        ).eval()
        with torch.inference_mode():
            output = self.model(torch.zeros((1, 48), device=self.device))
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 12):
            shape = (
                None if not isinstance(output, torch.Tensor) else tuple(output.shape)
            )
            raise ValueError(f"ANYmal-C policy output must be (1, 12), got {shape}")

    def reset(self, frame: EvaluationFrame) -> None:
        """Reset the previous action used by the policy observation."""
        self.previous_action.fill(0.0)

    def infer(self, frame: EvaluationFrame) -> PolicyOutput:
        """Build one 48-D observation and return 12 joint targets."""
        if frame.robot_state is None:
            raise RuntimeError("ANYmal-C robot state is required")
        observation = self._build_observation(frame)
        tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            output = self._model()(tensor)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 12):
            shape = (
                None if not isinstance(output, torch.Tensor) else tuple(output.shape)
            )
            raise ValueError(f"ANYmal-C policy output must be (1, 12), got {shape}")
        action = require_finite(
            "ANYmal-C action",
            output[0].detach().cpu().numpy(),
        )
        self.previous_action = action.copy()
        return PolicyOutput(
            action=self._joints().command(
                position=_DEFAULT_POSITION + 0.5 * action,
            ),
            termination_reason=_fall_reason(frame.robot_state.root_pose),
        )

    def metrics(self) -> dict[str, float]:
        """Return the metrics produced by this velocity example."""
        return {}

    def close(self) -> None:
        """Release the loaded TorchScript model."""
        self.model = None

    def _build_observation(self, frame: EvaluationFrame) -> np.ndarray:
        state = frame.robot_state
        if state is None:
            raise RuntimeError("ANYmal-C robot state is required")
        pose = np.asarray(state.root_pose, dtype=np.float32)
        velocity = np.asarray(state.root_velocity, dtype=np.float32)
        rotation = pose[:3, :3]
        qpos = self._joints().to_model(state.qpos)
        qvel = self._joints().to_model(state.qvel)
        command = require_finite(
            "ANYmal-C command",
            frame.controls["command"],
        )
        if command.shape != (3,):
            raise ValueError("ANYmal-C command must contain vx, vy, and yaw rate")
        observation = np.concatenate(
            (
                rotation.T @ velocity[:3],
                rotation.T @ velocity[3:],
                rotation.T @ np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
                command,
                qpos - _DEFAULT_POSITION,
                qvel,
                self.previous_action,
            ),
            dtype=np.float32,
        )
        if observation.shape != (48,):
            raise ValueError(
                f"ANYmal-C observation must have shape (48,), got {observation.shape}"
            )
        return require_finite("ANYmal-C observation", observation)

    def _joints(self) -> JointMap:
        if self.joints is None:
            raise RuntimeError("ANYmal-C Adapter is not set up")
        return self.joints

    def _model(self) -> torch.jit.ScriptModule:
        if self.model is None:
            raise RuntimeError("ANYmal-C Adapter is not set up")
        return self.model


def _fall_reason(root_pose: np.ndarray) -> str | None:
    pose = np.asarray(root_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("ANYmal-C root pose must be a finite 4x4 matrix")
    height = float(pose[2, 3])
    tilt = math.acos(float(np.clip(pose[2, 2], -1.0, 1.0)))
    reasons = []
    if height < 0.25:
        reasons.append(f"base_height_below_minimum: {height:.3f} m")
    if tilt > math.pi * 0.4:
        reasons.append(f"bad_orientation: {tilt:.3f} rad")
    return "; ".join(reasons) or None
