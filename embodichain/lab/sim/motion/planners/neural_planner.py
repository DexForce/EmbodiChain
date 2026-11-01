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
from __future__ import annotations

from dataclasses import MISSING, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping
import numpy as np
import torch

from embodichain.lab.sim.motion.planners.base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    _infer_batch_size,
    _check_batch_consistency,
    validate_plan_options,
)
from embodichain.lab.sim.motion.planners.utils import (
    MoveType,
    PlanResult,
    PlanState,
    TrajectorySampleMethod,
)
from embodichain.utils import configclass, logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

__all__ = [
    "NeuralPlanner",
    "NeuralPlannerCfg",
    "NeuralPlanOptions",
]


_WAYPOINT_OBSERVATION_LAYOUT = "unified_constraint_tokens"
_WAYPOINT_OBSERVATION_DTYPE = "float32"
_WAYPOINT_OBSERVATION_QUATERNION_ORDER = "xyzw"

_MASK_SEMANTICS = {
    "active_onehot": "one_at_clamped_active_waypoint",
    "valid_mask": "one_for_episode_waypoints_zero_for_padding",
    "pos_mask": "one_when_position_is_constrained",
    "rot_mask": "one_when_orientation_is_constrained",
    "joint_mask": "one_when_joint_position_is_constrained",
}
_MASKED_VALUE_SEMANTICS = {
    "position": "zero",
    "quaternion": "identity_xyzw",
    "joint": "zero",
    "relative_quaternion": "target_times_inverse_eef_xyzw",
}


@dataclass(frozen=True)
class _WaypointObservationLayout:
    """Concrete NMG observation layout derived from planner configuration."""

    num_waypoints: int
    num_controlled_joints: int
    use_relative_obs: bool
    canonicalize_quat_obs: bool = True
    joint_periodic_features: bool = False

    def __post_init__(self) -> None:
        if self.num_waypoints < 1 or self.num_controlled_joints < 1:
            raise ValueError("num_waypoints and num_controlled_joints must be positive")

    @property
    def block_widths(self) -> tuple[tuple[str, int], ...]:
        """Return ordered block names and widths for this policy capacity."""
        n = int(self.num_waypoints)
        j = int(self.num_controlled_joints)
        widths = [
            ("joint", j),
            ("eef", 7),
            ("waypoint_pos", 3 * n),
            ("waypoint_quat", 4 * n),
            ("waypoint_joint", j * n),
            ("active_onehot", n),
            ("valid_mask", n),
            ("pos_mask", n),
            ("rot_mask", n),
            ("joint_mask", n),
            ("last_action", j),
        ]
        if self.use_relative_obs:
            widths.extend(
                [
                    ("waypoint_rel_pos", 3 * n),
                    ("waypoint_rel_quat", 4 * n),
                    ("waypoint_joint_err", j * n),
                ]
            )
        if self.joint_periodic_features:
            widths.extend((("joint_sin", j), ("joint_cos", j)))
        return tuple(widths)

    @property
    def slices(self) -> dict[str, slice]:
        """Return a named slice for every flat observation block."""
        result = {}
        cursor = 0
        for name, width in self.block_widths:
            result[name] = slice(cursor, cursor + width)
            cursor += width
        return result

    @property
    def dim(self) -> int:
        """Return the flat observation width."""
        return sum(width for _, width in self.block_widths)

    @property
    def fingerprint(self) -> str:
        """Return the NMG-compatible fingerprint for this concrete layout."""
        payload = {
            "blocks": self.block_widths,
            "canonicalize_quat_obs": bool(self.canonicalize_quat_obs),
            "dtype": _WAYPOINT_OBSERVATION_DTYPE,
            "layout": _WAYPOINT_OBSERVATION_LAYOUT,
            "mask_semantics": _MASK_SEMANTICS,
            "masked_value_semantics": _MASKED_VALUE_SEMANTICS,
            "num_controlled_joints": int(self.num_controlled_joints),
            "num_waypoints": int(self.num_waypoints),
            "quaternion_order": _WAYPOINT_OBSERVATION_QUATERNION_ORDER,
            "use_relative_obs": bool(self.use_relative_obs),
        }
        if self.joint_periodic_features:
            payload["joint_periodic_encoding"] = "sin_cos_radians"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def onnx_metadata(self) -> dict[str, str]:
        """Return metadata expected from an NMG ONNX export."""
        metadata = {
            "nmg.observation_layout": _WAYPOINT_OBSERVATION_LAYOUT,
            "nmg.observation_fingerprint": self.fingerprint,
            "nmg.num_waypoints": str(int(self.num_waypoints)),
            "nmg.num_controlled_joints": str(int(self.num_controlled_joints)),
            "nmg.use_relative_obs": str(bool(self.use_relative_obs)).lower(),
            "nmg.canonicalize_quat_obs": str(bool(self.canonicalize_quat_obs)).lower(),
            "nmg.quaternion_order": _WAYPOINT_OBSERVATION_QUATERNION_ORDER,
        }
        if self.joint_periodic_features:
            metadata["nmg.joint_periodic_encoding"] = "sin_cos_radians"
        return metadata

    def concatenate(self, blocks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Validate and concatenate named observation blocks in contract order."""
        expected_names = tuple(name for name, _ in self.block_widths)
        if set(blocks) != set(expected_names):
            missing = sorted(set(expected_names) - set(blocks))
            extra = sorted(set(blocks) - set(expected_names))
            raise ValueError(
                f"Observation blocks do not match layout; missing={missing}, extra={extra}."
            )
        values = []
        batch_size = None
        for name, width in self.block_widths:
            value = blocks[name]
            if value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(
                    f"Observation block {name!r} must have shape [batch, {width}], "
                    f"got {tuple(value.shape)}."
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"Observation block {name!r} has batch size {value.shape[0]}, "
                    f"expected {batch_size}."
                )
            values.append(value)
        return torch.cat(values, dim=-1)


class _OnnxPolicy:
    """Small ONNX Runtime wrapper that keeps NMG model code out of EmbodiChain."""

    def __init__(
        self,
        path: Path,
        providers: list[str] | None = None,
        *,
        cuda_use_tf32: bool | None = None,
        device: torch.device | None = None,
    ) -> None:
        if cuda_use_tf32 is not None and type(cuda_use_tf32) is not bool:
            raise TypeError("onnx_cuda_use_tf32 must be a bool or None.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "NeuralPlanner requires ONNX Runtime. Install it with "
                "pip install 'embodichain[policy-deploy]'."
            ) from exc

        options = ort.SessionOptions()
        # Respect the host's CPU budget, including isolated benchmark workers;
        # ORT's default otherwise creates its own machine-sized thread pool.
        options.intra_op_num_threads = torch.get_num_threads()
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cuda_options = {}
        if device is not None and device.type == "cuda":
            # ORT otherwise defaults to device 0 independently of Torch's
            # current device. These are process-local CUDA ordinals.
            cuda_options["device_id"] = str(
                torch.cuda.current_device() if device.index is None else device.index
            )
        if cuda_use_tf32 is not None:
            cuda_options["use_tf32"] = "1" if cuda_use_tf32 else "0"
        provider_kwargs = {}
        if cuda_options:
            if providers is None:
                # Explicit CUDA settings use the CUDA/CPU path; do not enable
                # every available EP (for example TensorRT).
                available = ort.get_available_providers()
                providers = [
                    name
                    for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if name in available
                ]
            provider_kwargs["provider_options"] = [
                dict(cuda_options) if name == "CUDAExecutionProvider" else {}
                for name in providers
            ]
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers, **provider_kwargs
        )
        logger.log_info(
            "NMG ONNX execution providers: " + ", ".join(self.session.get_providers())
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "NMG ONNX policy must expose exactly one input and one output; "
                f"got {len(inputs)} inputs and {len(outputs)} outputs."
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if len(input_shape) != 2 or not isinstance(input_shape[1], int):
            raise ValueError(
                f"Expected ONNX input shape [batch, obs], got {input_shape}."
            )
        if (
            len(output_shape) != 2
            or not isinstance(output_shape[1], int)
            or output_shape[1] < 1
        ):
            raise ValueError(
                f"Expected ONNX output shape [batch, joints], got {output_shape}."
            )
        self.obs_dim = int(input_shape[1])
        self.output_dim = int(output_shape[1])
        self.observation_metadata = dict(
            self.session.get_modelmeta().custom_metadata_map
        )
        self.fixed_batch_size = (
            int(input_shape[0]) if isinstance(input_shape[0], int) else None
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.fixed_batch_size is not None and obs.shape[0] != self.fixed_batch_size:
            raise ValueError(
                f"ONNX policy has fixed batch size {self.fixed_batch_size}, "
                f"but received {obs.shape[0]}."
            )
        obs_np = np.ascontiguousarray(obs.detach().cpu().numpy(), dtype=np.float32)
        action = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        return torch.as_tensor(action, dtype=torch.float32, device=obs.device)


def _waypoint_obs_dim(
    num_waypoints: int,
    use_relative_obs: bool,
    num_arm_joints: int = 7,
    joint_periodic_features: bool = False,
) -> int:
    """Return the unified NMG constraint-observation width."""
    return _WaypointObservationLayout(
        int(num_waypoints),
        int(num_arm_joints),
        bool(use_relative_obs),
        joint_periodic_features=bool(joint_periodic_features),
    ).dim


def _quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a unit quaternion stored as xyzw."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply quaternion tensors stored as xyzw."""
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _canonicalize_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Map equivalent quaternion signs to the training-time ``w >= 0`` half."""
    return torch.where(q[..., 3:4] < 0.0, -q, q)


@configclass
class NeuralPlannerCfg(BasePlannerCfg):
    planner_type: str = "neural"

    onnx_model_path: str = MISSING
    """Path to a standalone NMG ONNX policy, including observation normalization."""

    control_part: str | None = None
    """Robot control part used for FK and qpos, e.g. 'left_arm'."""

    batch_control_parts: bool = False
    """Pack independent named-arm requests into one shared policy batch.

    Each arm keeps its original joints, FK frame, targets and termination state.
    This opt-in combines inference rows; it does not couple collision planning.
    """

    max_steps: int = 240
    """Maximum closed-loop kinematic rollout steps."""

    action_scale: float = 0.2
    """Delta joint scaling factor in radians."""

    num_arm_joints: int = 7
    """Number of arm joints controlled by the APG policy."""

    num_waypoints: int = 5
    """Number of constraint slots encoded by the ONNX policy."""

    use_relative_obs: bool = True
    """Whether the exported policy uses the unified relative-observation blocks."""

    canonicalize_quat_obs: bool = True
    """Whether quaternion observations use the training-time ``w >= 0`` convention."""

    joint_periodic_features: bool = False
    """Append sin/cos of raw joints after the relative observation blocks."""

    intermediate_orientation: bool = True
    """Whether every Cartesian waypoint requires its orientation constraint."""

    pos_eps: float = 0.01
    """Waypoint position threshold in metres."""

    rot_eps: float = 0.1
    """Waypoint rotation threshold in radians."""

    joint_eps: float = 0.01
    """Maximum absolute per-joint arrival error in radians (L-infinity)."""

    onnx_providers: list[str] | None = None
    """Optional ONNX Runtime execution-provider priority list.

    The CUDA provider follows the runtime robot's local CUDA device ordinal.
    """

    onnx_cuda_use_tf32: bool | None = None
    """Optional explicit TF32 setting for the ONNX CUDA execution provider.

    ``None`` preserves provider defaults; ``False`` requires FP32 CUDA matrix
    arithmetic to reduce numerical changes between actor batch sizes. CPU
    provider options are unchanged. With an explicit setting and no provider
    list, use available CUDA/CPU providers in that order.
    """

    policy_frame_from_world: list[list[float]] | None = None
    """Optional left transform ``policy_T_world`` for runtime poses."""

    policy_frame_from_world_by_control_part: dict[str, list[list[float]]] | None = None
    """Optional explicit policy-base transforms for each controlled arm.

    A shared single-arm policy can serve a fixed-base dual-arm assembly when
    every arm's world poses are mapped into that arm's training frame. When
    configured, every requested control part must have an entry; no arm falls
    back to another arm's frame. The global transform remains the single-arm
    default when this mapping is omitted.
    """

    runtime_tcp_from_policy_tcp: list[list[float]] | None = None
    """Optional right transform ``runtime_tcp_T_policy_tcp`` for runtime poses.

    Together, pose conversion is
    ``policy_T_policy_tcp = policy_T_world @ world_T_runtime_tcp
    @ runtime_tcp_T_policy_tcp``.
    """

    dt: float = 0.01
    """Output sampling period in seconds.

    Without :attr:`constraints` this is bookkeeping rather than executable
    timing.

    The rollout integrates a joint delta per step and never solves a duration,
    so this value is bookkeeping rather than executable timing: at the default
    ``action_scale`` a saturated step implies ``action_scale / dt`` rad/s,
    which exceeds a typical arm's joint velocity limit several times over. Set
    :attr:`constraints` to replace it with a solved time parameterization, in
    which case this becomes the period that trajectory is sampled on.
    """

    constraints: dict[str, float | list[float]] | None = None
    """Optional ``velocity`` and ``acceleration`` limits for output retiming.

    ``None`` keeps the rollout's nominal timing and current behavior. When set,
    the closed-loop samples are treated as a geometric path and re-parameterized
    under these limits, so the returned timing and derivatives are executable.
    Keys and value shapes match :class:`ToppraPlanOptions`: a scalar or one
    value per controlled joint. Retiming resamples the path, so it changes the
    reported positions and poses as well as the timing, and it cannot reduce
    the path's own jerk -- only the duration over which it is traversed.
    """


@configclass
class NeuralPlanOptions(PlanOptions):
    control_part: str | None = None
    start_qpos: torch.Tensor | None = None
    max_steps: int | None = None
    waypoint_modalities: list[str] | None = None
    """Optional explicit joint/position/orientation/pose semantics per waypoint."""


class NeuralPlanner(BasePlanner):
    r"""Neural motion planner based on an APG waypoint transformer policy.

    The planner loads a standalone ONNX waypoint policy and rolls it out in
    closed loop to drive the arm toward a sequence of end-effector and/or
    joint-position waypoints. Velocities and accelerations in the returned
    :class:`PlanResult` are estimated via finite differences over the generated
    position trajectory.

    Args:
        cfg: Configuration for the neural planner.

    Raises:
        ValueError: If ``onnx_model_path`` is missing or invalid.
        FileNotFoundError: If the ONNX model file does not exist.
        ImportError: If ONNX Runtime is not installed.
    """

    supported_move_types = frozenset({MoveType.EEF_MOVE, MoveType.JOINT_MOVE})
    supports_heterogeneous_waypoints = True
    preserve_plan_samples = True
    """Keep native closed-loop states instead of distance-resampling them."""
    preserve_failed_plan_positions = True
    """Keep closed-loop rollout samples even when not all waypoints converge."""

    def __init__(self, cfg: NeuralPlannerCfg):
        super().__init__(cfg)

        self.cfg: NeuralPlannerCfg = cfg
        if cfg.onnx_model_path is MISSING or not str(cfg.onnx_model_path):
            logger.log_error("onnx_model_path is required", ValueError)
        self._load_onnx_model(Path(cfg.onnx_model_path))

    @property
    def supports_multi_part(self) -> bool:
        """Whether independent arm requests share one policy inference batch."""
        return bool(getattr(self.cfg, "batch_control_parts", False))

    def default_plan_options(self) -> NeuralPlanOptions:
        return NeuralPlanOptions()

    def with_motion_context(
        self,
        options: PlanOptions,
        *,
        start_qpos: torch.Tensor | None,
        control_part: str | None,
    ) -> NeuralPlanOptions:
        """Forward MotionGenerator context into :class:`NeuralPlanOptions`."""
        if not isinstance(options, NeuralPlanOptions):
            logger.log_error("NeuralPlanner requires NeuralPlanOptions", TypeError)
        if options.control_part is None:
            options.control_part = control_part
        if options.start_qpos is None:
            options.start_qpos = start_qpos
        return options

    def _load_onnx_model(self, model_path: Path) -> None:
        if not model_path.exists():
            logger.log_error(f"ONNX policy not found: {model_path}", FileNotFoundError)
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "NeuralPlanner only accepts standalone .onnx policies; "
                f"got {model_path}."
            )

        self._num_waypoints = int(self.cfg.num_waypoints)
        self._use_relative_obs = bool(self.cfg.use_relative_obs)
        self._canonicalize_quat_obs = bool(self.cfg.canonicalize_quat_obs)
        self._joint_periodic_features = bool(self.cfg.joint_periodic_features)
        self._action_dim = int(self.cfg.num_arm_joints)
        if self._action_dim < 1:
            raise ValueError("num_arm_joints must be positive.")
        self._max_steps = int(self.cfg.max_steps)
        self._pos_eps = float(self.cfg.pos_eps)
        self._rot_eps = float(self.cfg.rot_eps)
        self._joint_eps = float(self.cfg.joint_eps)
        self._intermediate_orientation = bool(self.cfg.intermediate_orientation)
        self._policy = _OnnxPolicy(
            model_path,
            self.cfg.onnx_providers,
            cuda_use_tf32=self.cfg.onnx_cuda_use_tf32,
            device=self.device,
        )
        self._obs_dim = self._policy.obs_dim
        output_dim = getattr(self._policy, "output_dim", self._action_dim)
        if output_dim != self._action_dim:
            raise ValueError(
                f"ONNX output has {output_dim} joints, expected {self._action_dim}."
            )
        self._observation_layout = _WaypointObservationLayout(
            self._num_waypoints,
            self._action_dim,
            self._use_relative_obs,
            self._canonicalize_quat_obs,
            self._joint_periodic_features,
        )
        self._policy_frame_from_world = self._as_transform(
            self.cfg.policy_frame_from_world, "policy_frame_from_world"
        )
        self._runtime_tcp_from_policy_tcp = self._as_transform(
            self.cfg.runtime_tcp_from_policy_tcp, "runtime_tcp_from_policy_tcp"
        )
        self._policy_frames_by_control_part = {
            part: self._as_transform(
                value, f"policy_frame_from_world_by_control_part[{part!r}]"
            )
            for part, value in (
                self.cfg.policy_frame_from_world_by_control_part or {}
            ).items()
        }
        if self.cfg.policy_frame_from_world_by_control_part == {}:
            raise ValueError("policy_frame_from_world_by_control_part cannot be empty.")
        expected_obs_dim = self._observation_layout.dim
        if self._obs_dim != expected_obs_dim:
            raise ValueError(
                f"ONNX input has obs dim {self._obs_dim}, but the configured "
                f"unified constraint layout requires {expected_obs_dim}."
            )
        self._validate_observation_metadata(self._policy.observation_metadata)

    def _validate_observation_metadata(self, metadata: Mapping[str, str]) -> None:
        """Require an exact observation contract from the NMG export."""
        expected = self._observation_layout.onnx_metadata
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"ONNX observation layout metadata mismatch: {details}.")

    def _as_transform(self, value: list[list[float]] | None, name: str) -> torch.Tensor:
        """Convert an optional homogeneous-transform config to a device tensor."""
        if value is None:
            return torch.eye(4, dtype=torch.float32, device=self.device)
        transform = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}.")
        return transform

    def _to_policy_frame(
        self, xpos: torch.Tensor, control_part: str | None = None
    ) -> torch.Tensor:
        """Map runtime-world TCP poses to the selected arm's training frame."""
        frames = getattr(self, "_policy_frames_by_control_part", {})
        if frames:
            part = control_part or self.cfg.control_part
            if part not in frames:
                raise ValueError(
                    f"No NMG policy-frame transform for control part {part!r}."
                )
            transform = frames[part]
        else:
            transform = self._policy_frame_from_world
        left = transform.expand(xpos.shape[0], -1, -1)
        right = self._runtime_tcp_from_policy_tcp.expand(xpos.shape[0], -1, -1)
        return torch.bmm(torch.bmm(left, xpos), right)

    @validate_plan_options(options_cls=NeuralPlanOptions)
    @torch.no_grad()
    def plan(
        self,
        target_states: list[PlanState],
        options: NeuralPlanOptions = NeuralPlanOptions(),
    ) -> PlanResult:
        r"""Execute neural trajectory planning.

        Runs the waypoint transformer policy in closed loop for each environment
        until all waypoints are reached or ``max_steps`` is exhausted.

        Args:
            target_states: List of :class:`PlanState` waypoints. Each entry uses
                :attr:`MoveType.EEF_MOVE` with ``xpos`` shape ``(B, 4, 4)`` or
                :attr:`MoveType.JOINT_MOVE` with ``qpos`` shape ``(B, num_arm_joints)``.
            options: :class:`NeuralPlanOptions` with ``control_part``,
                ``start_qpos``, and ``max_steps`` overrides.

        Returns:
            :class:`PlanResult` containing the planned trajectory. All tensor
            fields are env-batched with leading dim ``B``: ``success`` ``(B,)``,
            ``positions``/``velocities``/``accelerations`` ``(B, N, DOF)``,
            ``xpos_list`` ``(B, N, 4, 4)``, ``dt`` ``(B, N)``, and
            ``duration`` ``(B,)``. Velocities and accelerations are computed
            via finite differences and are therefore approximate.

        Raises:
            ValueError: If ``control_part`` is not provided, a target state is
                unsupported, or ``start_qpos`` has too few joints.
        """
        if not target_states:
            return PlanResult(
                success=torch.zeros(0, dtype=torch.bool, device=self.device),
                positions=None,
            )

        control_part = options.control_part or self.cfg.control_part
        if control_part is None:
            logger.log_error(
                "control_part is required for NeuralPlanner",
                ValueError,
            )

        return self._plan_groups([(control_part, target_states, options)])[control_part]

    @torch.no_grad()
    def plan_multi_part(
        self,
        target_states: Mapping[str, list[PlanState]],
        options: Mapping[str, NeuralPlanOptions],
    ) -> dict[str, PlanResult]:
        """Plan independent manipulators with one shared policy inference batch.

        The actor batch is arm-major: all ``B`` scene rows for the first part,
        then all ``B`` rows for the next part. Robot FK always receives the
        original scene batch for its named arm. Arms may have different goal
        counts and rollout budgets; completed or exhausted rows hold independently.

        Args:
            target_states: Ordered targets for each explicitly named control part.
                Every part must contain the same environment batch size ``B``.
            options: Matching per-part start states and native neural options.
                An explicit ``control_part`` must match its mapping key.

        Returns:
            One ordinary ``B``-row result per part, with original world-frame
            poses and independently validated raw or retimed trajectories.
            ``policy_batching`` describes the shared inference work.

        Raises:
            ValueError: If batching is disabled, keys or scene batches disagree,
                a request is empty, or a policy's fixed batch cannot hold the group.
        """
        if not self.supports_multi_part:
            raise ValueError(
                "Neural multi-part planning requires batch_control_parts=True."
            )
        if not target_states or set(target_states) != set(options):
            raise ValueError(
                "Multi-part targets and options need identical non-empty keys."
            )
        groups = []
        for part, targets in target_states.items():
            if not isinstance(part, str) or not part:
                raise ValueError(
                    "Multi-part keys must be non-empty control-part names."
                )
            option = options[part]
            if not isinstance(option, NeuralPlanOptions):
                raise TypeError(
                    "Multi-part neural planning requires NeuralPlanOptions."
                )
            if option.control_part not in (None, part):
                raise ValueError("Multi-part option.control_part must match its key.")
            if not targets:
                raise ValueError(
                    f"Multi-part target list for {part!r} cannot be empty."
                )
            groups.append((part, targets, option))
        return self._plan_groups(groups)

    def _plan_groups(
        self, groups: list[tuple[str, list[PlanState], NeuralPlanOptions]]
    ) -> dict[str, PlanResult]:
        """Run single-arm and packed-arm requests through the same rollout."""
        parsed = []
        initial = []
        bounds = []
        budgets = []
        batch_size = None
        robot_batch = getattr(self.robot, "num_instances", None)
        for part, targets, option in groups:
            size = _check_batch_consistency(targets, robot_batch, robot_batch)
            if batch_size is not None and size != batch_size:
                raise ValueError(
                    "All packed control parts must have the same scene batch."
                )
            batch_size = size
            values = self._parse_waypoints(targets, option.waypoint_modalities, part)
            qpos = self._initial_qpos(part, option.start_qpos)
            if qpos.shape[0] != size:
                raise ValueError("start_qpos must match the target scene batch.")
            if len(groups) > 1 and qpos.shape[-1] != self._action_dim:
                raise ValueError(
                    "Packed control parts must match the policy joint count."
                )
            limits = self.robot.get_qpos_limits(name=part)[0].to(self.device)
            lower = limits[: self._action_dim, 0]
            upper = limits[: self._action_dim, 1]
            budget = int(option.max_steps or self._max_steps)
            if budget < 1:
                raise ValueError("Neural rollout max_steps must be positive.")
            parsed.append(values)
            initial.append(qpos)
            bounds.append((lower, upper))
            budgets.append(budget)

        assert batch_size is not None
        b = batch_size
        actor_batch = b * len(groups)
        fixed_batch = getattr(self._policy, "fixed_batch_size", None)
        if len(groups) > 1 and fixed_batch is not None and fixed_batch != actor_batch:
            raise ValueError(
                f"Packed arms require actor batch {actor_batch}, but the ONNX policy "
                f"has fixed batch {fixed_batch}. Export a dynamic-batch policy."
            )
        qpos = torch.cat(initial, dim=0)
        lower = torch.cat([value[0].expand(b, -1) for value in bounds])
        upper = torch.cat([value[1].expand(b, -1) for value in bounds])
        # All arms use one observation schema, so padded capacities match even
        # when their valid constraint counts or modalities differ.
        (
            waypoints_pos,
            waypoints_quat,
            waypoints_joint,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
        ) = [torch.cat([values[index] for values in parsed]) for index in range(7)]
        episode_k = torch.cat(
            [
                torch.full((b,), values[7], device=self.device, dtype=torch.long)
                for values in parsed
            ]
        )
        step_limits = torch.cat(
            [
                torch.full((b,), budget, device=self.device, dtype=torch.long)
                for budget in budgets
            ]
        )
        slices = [slice(index * b, (index + 1) * b) for index in range(len(groups))]

        def forward_kinematics(
            packed_qpos: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            world = [
                self._fk_matrix(packed_qpos[rows], part)
                for rows, (part, _, _) in zip(slices, groups)
            ]
            local = [
                self._policy_pose_xyzw(pose, part)
                for pose, (part, _, _) in zip(world, groups)
            ]
            return torch.cat(world), torch.cat(local)

        last_action = torch.zeros(actor_batch, self._action_dim, device=self.device)
        active_idx = torch.zeros(actor_batch, dtype=torch.long, device=self.device)
        positions = [qpos.clone()]
        world_pose, ee_pose = forward_kinematics(qpos)
        xpos_list = [world_pose]
        converged = torch.zeros(actor_batch, dtype=torch.bool, device=self.device)
        path_lengths = torch.ones(actor_batch, dtype=torch.int32, device=self.device)
        forward_calls = 0
        for step in range(max(budgets)):
            obs = self._build_obs(
                qpos[:, : self._action_dim],
                ee_pose,
                waypoints_pos,
                waypoints_quat,
                waypoints_joint,
                valid_mask,
                pos_mask,
                rot_mask,
                joint_mask,
                active_idx,
                last_action,
            )
            action = self._policy(obs).clamp(-1.0, 1.0)
            forward_calls += 1
            active = ~converged & (step < step_limits)
            action = torch.where(active.unsqueeze(-1), action, torch.zeros_like(action))
            qpos[:, : self._action_dim] += action * float(self.cfg.action_scale)
            qpos[:, : self._action_dim] = torch.clamp(
                qpos[:, : self._action_dim], lower, upper
            )
            last_action = torch.where(active.unsqueeze(-1), action, last_action)
            positions.append(qpos.clone())
            world_pose, ee_pose = forward_kinematics(qpos)
            xpos_list.append(world_pose)
            path_lengths += active.to(path_lengths.dtype)
            reached = active & self._is_active_reached(
                qpos[:, : self._action_dim],
                ee_pose,
                waypoints_pos,
                waypoints_quat,
                waypoints_joint,
                pos_mask,
                rot_mask,
                joint_mask,
                active_idx,
            )
            active_idx = torch.where(reached, active_idx + 1, active_idx)
            converged |= active_idx >= episode_k
            if (converged | (step + 1 >= step_limits)).all():
                break

        positions_t = torch.stack(positions, dim=1)
        xpos_t = torch.stack(xpos_list, dim=1)
        batching = {
            "scene_batch_size": b,
            "control_part_count": len(groups),
            "actor_batch_size": actor_batch,
            "policy_forward_calls": forward_calls,
        }
        results = {}
        for index, (part, _, _) in enumerate(groups):
            rows = slices[index]
            lengths = path_lengths[rows]
            # Remove another arm's terminal padding before timing or reporting.
            # This is one host read per part after the shared rollout, not per step.
            count = int(lengths.max().item())
            arm_positions = positions_t[rows, :count]
            success = converged[rows]
            if self.cfg.constraints is not None:
                result = self._retimed_result(
                    success,
                    arm_positions,
                    part,
                    path_lengths=lengths,
                    waypoints=(
                        parsed[index][0],
                        parsed[index][1],
                        parsed[index][2],
                        parsed[index][4],
                        parsed[index][5],
                        parsed[index][6],
                    ),
                    episode_k=parsed[index][7],
                    qpos_limits=bounds[index],
                )
            else:
                dt = torch.full(
                    arm_positions.shape[:2],
                    float(self.cfg.dt),
                    dtype=torch.float32,
                    device=self.device,
                )
                dt[:, 0] = 0.0
                velocities, accelerations = self._compute_vel_acc_via_finite_diff(
                    arm_positions, dt
                )
                result = PlanResult(
                    success=success,
                    positions=arm_positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    xpos_list=xpos_t[rows, :count],
                    dt=dt,
                    policy_rollout_positions=arm_positions,
                    policy_rollout_lengths=lengths,
                )
            result.policy_batching = dict(batching)
            results[part] = result
        return results

    def _retimed_result(
        self,
        success: torch.Tensor,
        positions: torch.Tensor,
        control_part: str,
        *,
        path_lengths: torch.Tensor,
        waypoints: tuple[torch.Tensor, ...],
        episode_k: int,
        qpos_limits: tuple[torch.Tensor, torch.Tensor],
    ) -> PlanResult:
        """Re-parameterize the rollout path under the configured limits.

        Retiming fits a spline through the rollout samples and resamples it, so
        the returned trajectory is not the one the rollout verified. Success is
        therefore re-derived from the samples actually returned rather than
        carried over: the resampled grid can step past a waypoint the rollout
        stopped on, and a spline through samples clamped at a joint limit can
        overshoot that limit between them. A row is reported as successful only
        when its returned trajectory parameterizes, moves in nonzero time,
        stays inside the joint limits, and still reaches every waypoint.

        Args:
            success: Per-env rollout convergence of shape ``(B,)``.
            positions: Rollout joint samples of shape ``(B, N, DOF)``.
            control_part: Robot control part used for forward kinematics.
            path_lengths: Valid rollout prefix length for each environment.
                Samples after that prefix only hold an already-converged row
                while other rows continue and must not affect spline fitting.
            waypoints: Parsed waypoint targets and masks from
                :meth:`_parse_waypoints`.
            episode_k: Number of waypoints this episode must reach.
            qpos_limits: Lower and upper joint-position limits.

        Returns:
            PlanResult with solved timing, derivatives and recomputed poses.
        """
        from .toppra_planner import retime_joint_paths

        # Sample the solved trajectory on the configured control period rather
        # than on the rollout's step count. The rollout emits however many
        # steps it happened to take, which is far too coarse once the duration
        # stretches: a waypoint the rollout stopped on then falls between two
        # output samples and is reported as missed.
        retimed = retime_joint_paths(
            positions,
            constraints=dict(self.cfg.constraints),
            path_lengths=path_lengths,
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=float(self.cfg.dt),
            device=self.device,
        )
        retimed_positions = retimed.positions
        # One batched FK over every sample; the waypoint re-check reuses it.
        poses = self.robot.compute_batch_fk(
            qpos=retimed_positions, name=control_part, to_matrix=True
        )
        success = success & retimed.success.to(success.device)
        success = success & self._retimed_is_executable(
            retimed_positions, retimed.dt, qpos_limits
        )
        success = success & self._retimed_reaches_waypoints(
            retimed_positions, poses, waypoints, episode_k, control_part
        )
        return PlanResult(
            success=success,
            positions=retimed_positions,
            velocities=retimed.velocities,
            accelerations=retimed.accelerations,
            xpos_list=poses,
            dt=retimed.dt,
            policy_rollout_positions=positions,
            policy_rollout_lengths=path_lengths,
        )

    def _retimed_is_executable(
        self,
        positions: torch.Tensor,
        dt: torch.Tensor,
        qpos_limits: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return whether each retimed row has real timing and legal positions.

        The shared retiming kernel returns a zero-duration result for a path
        whose endpoints nearly coincide, which would claim a move with no time
        to execute it. Spline interpolation through samples the rollout clamped
        at a joint limit can also overshoot that limit between them, since the
        fit sees only the sampled values.
        """
        lower, upper = qpos_limits
        arm = positions[..., : self._action_dim]
        moves = (arm - arm[:, :1]).abs().amax(dim=(1, 2)) > 1.0e-9
        timed = dt.sum(dim=-1) > 0.0
        within = ((arm >= lower) & (arm <= upper)).all(dim=-1).all(dim=-1)
        return within & (timed | ~moves)

    def _retimed_reaches_waypoints(
        self,
        positions: torch.Tensor,
        poses: torch.Tensor,
        waypoints: tuple[torch.Tensor, ...],
        episode_k: int,
        control_part: str | None = None,
    ) -> torch.Tensor:
        """Return whether each retimed row still reaches every waypoint in order.

        Applies the rollout's own arrival test to the resampled grid, so a
        trajectory that passes a waypoint only between output samples is not
        reported as reaching it.

        The test runs once per waypoint over every sample at once rather than
        once per sample, so its cost does not grow with the solved duration
        and nothing synchronizes with the host inside the loop. Order follows
        the rollout with nondecreasing arrival indices, so targets independently
        satisfied by one stationary sample do not require duplicate output.
        """
        (
            waypoints_pos,
            waypoints_quat,
            waypoints_joint,
            pos_mask,
            rot_mask,
            joint_mask,
        ) = waypoints
        batch, samples = positions.shape[:2]
        flat_joint = positions[..., : self._action_dim].flatten(0, 1)
        flat_pose = self._policy_pose_xyzw(poses.flatten(0, 1), control_part)
        per_sample = lambda tensor: tensor.repeat_interleave(samples, dim=0)
        targets = tuple(
            per_sample(tensor)
            for tensor in (
                waypoints_pos,
                waypoints_quat,
                waypoints_joint,
                pos_mask,
                rot_mask,
                joint_mask,
            )
        )
        sample_index = torch.arange(samples, device=self.device)
        # Sample at which the previous waypoint was first reached; -1 before
        # the first, and `samples` once a row has missed one.
        reached_at = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        for waypoint in range(episode_k):
            active = torch.full(
                (batch * samples,), waypoint, dtype=torch.long, device=self.device
            )
            reached = self._is_active_reached(
                flat_joint, flat_pose, *targets, active
            ).view(batch, samples)
            eligible = reached & (sample_index >= reached_at.unsqueeze(-1))
            reached_at = torch.where(
                eligible, sample_index, torch.full_like(sample_index, samples)
            ).amin(dim=-1)
        return reached_at < samples

    def _parse_waypoints(
        self,
        target_states: list[PlanState],
        modalities: list[str] | None = None,
        control_part: str | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        if len(target_states) > self._num_waypoints:
            logger.log_error(
                f"Received {len(target_states)} waypoints, but the ONNX policy supports "
                f"at most {self._num_waypoints}.",
                ValueError,
            )
        if modalities is not None:
            if len(modalities) != len(target_states):
                raise ValueError("waypoint_modalities must match the target count.")
            for target, modality in zip(target_states, modalities):
                expected = (
                    MoveType.JOINT_MOVE if modality == "joint" else MoveType.EEF_MOVE
                )
                if (
                    modality not in {"joint", "position", "orientation", "pose"}
                    or target.move_type != expected
                ):
                    raise ValueError(
                        "waypoint_modalities must agree with each target's move_type."
                    )
        if modalities is not None and "position" in modalities:
            offset = self._runtime_tcp_from_policy_tcp[:3, 3]
            if bool(torch.any(offset.abs() > 1.0e-8)):
                raise ValueError(
                    "Position-only constraints require the runtime and policy TCP origins to coincide."
                )
        b = _infer_batch_size(target_states) or 1
        waypoint_pos = torch.zeros(b, self._num_waypoints, 3, device=self.device)
        waypoint_quat = torch.zeros(b, self._num_waypoints, 4, device=self.device)
        waypoint_quat[..., 3] = 1.0
        waypoint_joint = torch.zeros(
            b, self._num_waypoints, self._action_dim, device=self.device
        )
        valid_mask = torch.zeros(b, self._num_waypoints, device=self.device)
        pos_mask = torch.zeros_like(valid_mask)
        rot_mask = torch.zeros_like(valid_mask)
        joint_mask = torch.zeros_like(valid_mask)
        for idx, target in enumerate(target_states):
            if target.move_type == MoveType.EEF_MOVE and target.xpos is not None:
                xpos = torch.as_tensor(
                    target.xpos, dtype=torch.float32, device=self.device
                )
                if xpos.dim() == 2:
                    xpos = xpos.unsqueeze(0)
                policy_xpos = self._to_policy_frame(xpos, control_part)
                waypoint_pos[:, idx] = policy_xpos[:, :3, 3]
                # ``quat_from_matrix`` already returns EmbodiChain's public
                # ``xyzw`` convention.  Do not pass it through ``convert_quat``:
                # that helper interprets its input as the opposite convention
                # and would rotate the components a second time.
                quat_xyzw = quat_from_matrix(policy_xpos[:, :3, :3])
                waypoint_quat[:, idx] = (
                    _canonicalize_quat_xyzw(quat_xyzw)
                    if getattr(self, "_canonicalize_quat_obs", False)
                    else quat_xyzw
                )
                pos_mask[:, idx] = 1.0
                rot_mask[:, idx] = 1.0
            elif target.move_type == MoveType.JOINT_MOVE and target.qpos is not None:
                qpos = torch.as_tensor(
                    target.qpos, dtype=torch.float32, device=self.device
                )
                if qpos.dim() == 1:
                    qpos = qpos.unsqueeze(0)
                if qpos.shape != (b, self._action_dim):
                    logger.log_error(
                        "NeuralPlanner JOINT_MOVE qpos must have shape "
                        f"({b}, {self._action_dim}), got {tuple(qpos.shape)}.",
                        ValueError,
                    )
                waypoint_joint[:, idx] = qpos
                joint_mask[:, idx] = 1.0
            else:
                logger.log_error(
                    "NeuralPlanner expects EEF_MOVE entries with xpos or "
                    "JOINT_MOVE entries with qpos.",
                    ValueError,
                )
            valid_mask[:, idx] = 1.0
        if modalities is not None:
            for idx, modality in enumerate(modalities):
                pos_mask[:, idx] = float(modality in {"position", "pose"})
                rot_mask[:, idx] = float(modality in {"orientation", "pose"})
        if modalities is None and not self._intermediate_orientation:
            final_mask = torch.zeros_like(rot_mask)
            final_mask[:, len(target_states) - 1] = 1.0
            rot_mask *= final_mask
        return (
            waypoint_pos,
            waypoint_quat,
            waypoint_joint,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
            len(target_states),
        )

    def _initial_qpos(
        self, control_part: str, start_qpos: torch.Tensor | None
    ) -> torch.Tensor:
        if start_qpos is None:
            qpos = self.robot.get_qpos(name=control_part)
        else:
            qpos = torch.as_tensor(start_qpos, dtype=torch.float32, device=self.device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[-1] < self._action_dim:
            logger.log_error(
                f"start_qpos has {qpos.shape[-1]} joints, but policy expects "
                f"{self._action_dim}.",
                ValueError,
            )
        return qpos.to(self.device).clone()

    def _fk_matrix(self, qpos: torch.Tensor, control_part: str) -> torch.Tensor:
        return self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)

    def _policy_pose_xyzw(
        self, fk_matrix: torch.Tensor, control_part: str | None = None
    ) -> torch.Tensor:
        """Map ``(M, 4, 4)`` runtime TCP poses to policy-frame ``xyz + xyzw``."""
        fk = self._to_policy_frame(fk_matrix, control_part)
        pos = fk[:, :3, 3]
        # ``quat_from_matrix`` is an EmbodiChain ``xyzw`` producer; converting
        # it again would turn a valid pose into a different rotation.
        quat_xyzw = quat_from_matrix(fk[:, :3, :3])
        if getattr(self, "_canonicalize_quat_obs", False):
            quat_xyzw = _canonicalize_quat_xyzw(quat_xyzw)
        return torch.cat([pos, quat_xyzw], dim=-1)

    def _build_obs(
        self,
        joint_pos: torch.Tensor,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        waypoint_joint: torch.Tensor,
        valid_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        active_idx: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        b = joint_pos.shape[0]
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_onehot = torch.zeros(b, self._num_waypoints, device=self.device)
        active_onehot.scatter_(1, active_idx_clamped.unsqueeze(1), 1.0)
        pos_block = waypoint_pos * pos_mask.unsqueeze(-1)
        joint_block = waypoint_joint * joint_mask.unsqueeze(-1)
        identity = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=waypoint_quat.dtype,
            device=self.device,
        )
        quat_block = torch.where(
            rot_mask.unsqueeze(-1) > 0.5,
            waypoint_quat,
            identity.view(1, 1, 4),
        )
        obs_blocks = {
            "joint": joint_pos,
            "eef": ee_pose,
            "waypoint_pos": pos_block.reshape(b, self._num_waypoints * 3),
            "waypoint_quat": quat_block.reshape(b, self._num_waypoints * 4),
            "waypoint_joint": joint_block.reshape(
                b, self._num_waypoints * self._action_dim
            ),
            "active_onehot": active_onehot,
            "valid_mask": valid_mask,
            "pos_mask": pos_mask,
            "rot_mask": rot_mask,
            "joint_mask": joint_mask,
            "last_action": last_action,
        }
        if self._use_relative_obs:
            inv_eef = _quat_inverse_xyzw(ee_pose[:, 3:7])
            rel_pos = (pos_block - ee_pose[:, None, :3]) * pos_mask.unsqueeze(-1)
            rel_quat = _quat_mul_xyzw(
                quat_block,
                inv_eef[:, None, :].expand_as(quat_block),
            )
            if getattr(self, "_canonicalize_quat_obs", False):
                rel_quat = _canonicalize_quat_xyzw(rel_quat)
            rel_quat = torch.where(
                rot_mask.unsqueeze(-1) > 0.5,
                rel_quat,
                identity.view(1, 1, 4),
            )
            joint_err = (joint_block - joint_pos[:, None]) * joint_mask.unsqueeze(-1)
            obs_blocks.update(
                {
                    "waypoint_rel_pos": rel_pos.reshape(b, self._num_waypoints * 3),
                    "waypoint_rel_quat": rel_quat.reshape(b, self._num_waypoints * 4),
                    "waypoint_joint_err": joint_err.reshape(
                        b, self._num_waypoints * self._action_dim
                    ),
                }
            )
        if getattr(self, "_joint_periodic_features", False):
            obs_blocks.update(joint_sin=joint_pos.sin(), joint_cos=joint_pos.cos())
        layout = getattr(
            self,
            "_observation_layout",
            _WaypointObservationLayout(
                self._num_waypoints,
                self._action_dim,
                self._use_relative_obs,
                getattr(self, "_canonicalize_quat_obs", False),
                getattr(self, "_joint_periodic_features", False),
            ),
        )
        obs = layout.concatenate(obs_blocks)
        if obs.shape[-1] != self._obs_dim:
            raise ValueError(
                f"Built obs dim {obs.shape[-1]}, expected {self._obs_dim}."
            )
        return obs

    def _is_active_reached(
        self,
        joint_pos: torch.Tensor,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        waypoint_joint: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        b = ee_pose.shape[0]
        idx = torch.arange(b, device=self.device)
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_pos = waypoint_pos[idx, active_idx_clamped]
        active_quat_xyzw = waypoint_quat[idx, active_idx_clamped]
        active_joint = waypoint_joint[idx, active_idx_clamped]
        active_pos_mask = pos_mask[idx, active_idx_clamped] > 0.5
        active_rot_mask = rot_mask[idx, active_idx_clamped] > 0.5
        active_joint_mask = joint_mask[idx, active_idx_clamped] > 0.5
        pos_dist = (ee_pose[:, :3] - active_pos).norm(dim=-1)
        # All planner poses are ``xyz + xyzw`` and ``quat_error_magnitude``
        # consumes xyzw.  Keeping the values in that convention avoids a
        # needless round trip through the native/external ``wxyz`` layout.
        ee_quat_xyzw = ee_pose[:, 3:7]
        rot_dist = quat_error_magnitude(ee_quat_xyzw, active_quat_xyzw)
        rot_ok = torch.where(
            active_rot_mask,
            rot_dist < self._rot_eps,
            torch.ones_like(rot_dist, dtype=torch.bool),
        )
        pos_ok = torch.where(
            active_pos_mask,
            pos_dist < self._pos_eps,
            torch.ones_like(pos_dist, dtype=torch.bool),
        )
        joint_dist = torch.amax(torch.abs(joint_pos - active_joint), dim=-1)
        joint_ok = torch.where(
            active_joint_mask,
            joint_dist <= self._joint_eps,
            torch.ones_like(joint_dist, dtype=torch.bool),
        )
        reached = pos_ok & rot_ok & joint_ok
        return reached

    @staticmethod
    def _compute_vel_acc_via_finite_diff(
        positions: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate velocities and accelerations via finite differences.

        Uses a second-order central difference for interior points and one-sided
        differences at the boundaries. The estimates are approximate because the
        neural policy does not produce velocities or accelerations directly.

        Args:
            positions: Joint positions of shape ``(B, N, DOF)``.
            dt: Per-point time deltas of shape ``(B, N)``. ``dt[:, t]`` is the
                interval used to reach point ``t`` from point ``t - 1``.

        Returns:
            Tuple of ``(velocities, accelerations)``, each of shape
            ``(B, N, DOF)``.
        """
        b, n, dof = positions.shape
        if n == 1:
            zeros = torch.zeros_like(positions)
            return zeros, zeros

        # Forward difference for the first point: (p[1] - p[0]) / dt[1]
        v_first = (positions[:, 1] - positions[:, 0]) / dt[:, 1].unsqueeze(-1)
        # Backward difference for the last point: (p[N-1] - p[N-2]) / dt[N-1]
        v_last = (positions[:, -1] - positions[:, -2]) / dt[:, -1].unsqueeze(-1)

        if n == 2:
            velocities = torch.stack([v_first, v_last], dim=1)
            return velocities, torch.zeros_like(velocities)

        # Central difference for interior points:
        # (p[i+1] - p[i-1]) / (dt[i] + dt[i+1])
        p_next = positions[:, 2:]
        p_prev = positions[:, :-2]
        dt_sum = (dt[:, 1:-1] + dt[:, 2:]).unsqueeze(-1)
        v_interior = (p_next - p_prev) / dt_sum.clamp_min(1e-12)
        velocities = torch.cat(
            [v_first.unsqueeze(1), v_interior, v_last.unsqueeze(1)], dim=1
        )

        # Acceleration via second-order finite differences.
        # Boundary points use a one-sided stencil; interior points use
        # (p[i+1] - 2*p[i] + p[i-1]) / dt[i]^2
        a_first = (positions[:, 2] - 2.0 * positions[:, 1] + positions[:, 0]) / (
            dt[:, 1].unsqueeze(-1) ** 2
        )
        a_last = (positions[:, -1] - 2.0 * positions[:, -2] + positions[:, -3]) / (
            dt[:, -1].unsqueeze(-1) ** 2
        )
        a_interior = (
            positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]
        ) / (dt[:, 1:-1].unsqueeze(-1) ** 2)
        accelerations = torch.cat(
            [a_first.unsqueeze(1), a_interior, a_last.unsqueeze(1)], dim=1
        )

        return velocities, accelerations
