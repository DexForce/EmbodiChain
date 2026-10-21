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

"""Explicit historical-Warp rollout qualification using original NeRD runtime.

Run in an isolated subprocess with pinned Warp 1.8.0 (Cartpole) or 1.5.1 (Ant).
This harness does not deploy old physical models into current Newton/DexSim.
"""

from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any
from collections.abc import Mapping, Sequence
from .reference import OFFICIAL_COMMIT, reference_identity, _hash

__all__ = ["evaluate_reference_rollouts"]


def _positive(name: str, value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@contextmanager
def _cpu_stream_compatibility(wp: Any, device: str):
    """Adapt only the original runtime's CUDA-stream request on CPU.

    NeuralIntegrator.simulate unconditionally requests a Torch CUDA stream.
    torch.cuda.stream(None) is the official no-op CPU compatibility boundary;
    simulation, preprocessing, network, decoding and FK methods stay unmodified.
    """
    original = wp.stream_to_torch
    if not wp.get_device(device).is_cuda:

        def compatible(stream_or_device: Any):
            candidate = getattr(stream_or_device, "device", stream_or_device)
            if not wp.get_device(candidate).is_cuda:
                return None
            return original(stream_or_device)

        wp.stream_to_torch = compatible
    try:
        yield
    finally:
        wp.stream_to_torch = original


def _physical_arrays(model: Any) -> dict[str, Any]:
    """Read the same physical fields recorded by original-teacher collection."""
    import numpy as np

    result = {}
    for name in (
        "body_mass",
        "body_inertia",
        "body_com",
        "joint_armature",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_axis_mode",
        "joint_target_ke",
        "joint_target_kd",
    ):
        value = getattr(model, name, None)
        if value is not None:
            result[name] = (
                value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            )
    for field in ("mu", "ke", "kd", "kf"):
        value = getattr(model.shape_materials, field, None)
        if value is not None:
            result["shape_material_" + field] = value.numpy()
    result["shape_thickness"] = model.shape_geo.thickness.numpy()
    result["shape_body"] = model.shape_body.numpy()
    result["ground_plane"] = model.ground_plane.numpy()
    result["gravity"] = np.asarray(model.gravity, dtype=np.float64)
    return result


def _canonical_physics(
    arrays: Mapping[str, Any], batch: int, bodies: int
) -> dict[str, Any]:
    """Remove replica indexing while proving the recorded replicas homogeneous."""
    import numpy as np

    result = {}
    shape_body = np.asarray(arrays["shape_body"])
    # Original Warp always appends one shared ground-plane shape.
    if (len(shape_body) - 1) % batch:
        raise ValueError("Historical shape replication is not homogeneous")
    shapes = (len(shape_body) - 1) // batch
    for key, values in arrays.items():
        value = np.asarray(values)
        if key in {"ground_plane", "gravity"}:
            result[key] = value.tolist()
            continue
        if key.startswith("shape_"):
            if len(value) != batch * shapes + 1:
                raise ValueError(f"Unexpected historical shape field {key}")
            rows = value[:-1].reshape(batch, shapes, *value.shape[1:]).copy()
            if key == "shape_body":
                rows = np.where(
                    rows >= 0, rows - np.arange(batch)[:, None] * bodies, rows
                )
            tail = value[-1:].tolist()
        else:
            if len(value) % batch:
                raise ValueError(f"Unexpected historical replication for {key}")
            rows = value.reshape(batch, len(value) // batch, *value.shape[1:])
            tail = None
        if not np.allclose(rows, rows[:1], rtol=1e-6, atol=1e-7, equal_nan=False):
            raise ValueError(f"Historical physical field {key} differs across replicas")
        result[key] = {"per_environment": rows[0].tolist(), "shared": tail}
    return result


def _verify_historical_physics(
    environment: Any,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    warp_version: str,
) -> dict[str, Any]:
    """Require actual historical physical data, never relabel a Newton manifest."""
    import numpy as np

    recorded = manifest.get("teacher_origin", {})
    if (
        recorded.get("kind") != "official_warp"
        or recorded.get("source_commit") != OFFICIAL_COMMIT
        or recorded.get("warp_version") != warp_version
    ):
        raise ValueError(
            "Model does not record the required original-Warp teacher identity"
        )
    if not isinstance(manifest.get("source_physical_arrays"), Mapping):
        raise ValueError(
            "Historical source_physical_arrays are required; do not relabel a Newton model"
        )
    if (
        recorded.get("up_axis") != int(environment.model.up_axis)
        or recorded.get("num_substeps") != environment.sim_substeps_gt
    ):
        raise ValueError("Original teacher axis/substeps differ from model provenance")
    if recorded.get("integrator") != str(environment.integrator_type_gt):
        raise ValueError("Original teacher integrator differs from model provenance")
    if "gravity" in recorded and not np.allclose(
        np.asarray(recorded["gravity"]),
        np.asarray(environment.model.gravity, dtype=np.float64),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Original teacher gravity differs from recorded provenance")
    if not math.isclose(
        float(manifest.get("dt", 0)),
        float(environment.frame_dt),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Original teacher timestep differs from model provenance")
    robot = manifest["codec"]["robot"]
    expected_continuous = [1] if robot == "cartpole" else []
    if (
        list(manifest["codec"].get("continuous_q_indices", expected_continuous))
        != expected_continuous
    ):
        raise ValueError(
            "Historical codec continuous coordinates differ from original runtime"
        )
    if not np.allclose(
        manifest["codec"].get("base_com", [0.0, 0.0, 0.0]),
        environment.model.body_com.numpy()[0],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Historical codec base COM differs from actual source model")
    asset = (
        Path(source["root"])
        / "envs"
        / "warp_sim_envs"
        / "assets"
        / ("ant.xml" if robot == "ant" else "cartpole_single.urdf")
    )
    if manifest.get("asset_sha256") != _hash(asset):
        raise ValueError("Historical model asset checksum differs")
    batch = manifest.get("sampling", {}).get("batch_size")
    split_batch = (
        manifest.get("source_split_identities", {}).get("train", {}).get("num_envs")
    )
    if batch is not None and split_batch is not None and batch != split_batch:
        raise ValueError("Conflicting source sampling batch sizes")
    batch = _positive(
        "source sampling batch_size", batch if batch is not None else split_batch
    )
    bodies = _positive("source bodies_per_env", recorded.get("bodies_per_env"))
    if bodies != environment.bodies_per_env:
        raise ValueError("Historical body count differs")
    actual = _canonical_physics(
        _physical_arrays(environment.model), environment.num_envs, bodies
    )
    expected = _canonical_physics(manifest["source_physical_arrays"], batch, bodies)
    missing_fields = sorted(set(actual) - set(expected))
    if set(expected) - set(actual) or set(missing_fields) - {"gravity"}:
        raise ValueError("Historical physical field set differs")
    for key in expected:
        if key in {"ground_plane", "gravity"}:
            pairs = [(actual[key], expected[key])]
        else:
            pairs = [
                (actual[key][name], expected[key][name])
                for name in ("per_environment", "shared")
                if actual[key][name] is not None or expected[key][name] is not None
            ]
        for left, right in pairs:
            if (
                left is None
                or right is None
                or np.shape(left) != np.shape(right)
                or not np.allclose(left, right, rtol=1e-6, atol=1e-7)
            ):
                raise ValueError(f"Historical physical mismatch in {key}")
    identity = hashlib.sha256(
        json.dumps(
            actual, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "schema": "official-warp-physical-arrays-v2",
        "sha256": identity,
        "matched_fields": sorted(expected),
        "missing_recorded_fields": missing_fields,
        "matched_recorded_fields": True,
        "complete": not missing_fields,
        "status": (
            "matched" if not missing_fields else "pending_missing_recorded_physics"
        ),
        "gravity_live": np.asarray(
            environment.model.gravity, dtype=np.float64
        ).tolist(),
        "gravity_magnitude_in_training_metadata": "gravity" in expected,
    }


def _canonical_states(states: Any, codec: Any, rotation: Any) -> tuple[Any, Any]:
    import torch
    from embodichain.compute.dynamics import quat_rotate, quat_multiply

    q = states[..., : codec.q_dim]
    qd = codec.legacy_to_native(q, states[..., codec.q_dim :])
    if codec.robot == "ant":
        q = torch.cat(
            (
                quat_rotate(rotation, q[..., :3]),
                quat_multiply(rotation, q[..., 3:7]),
                q[..., 7:],
            ),
            -1,
        )
        qd = torch.cat(
            (
                quat_rotate(rotation, qd[..., :3]),
                quat_rotate(rotation, qd[..., 3:6]),
                qd[..., 6:],
            ),
            -1,
        )
    return q, qd


class _HistoricalPD:
    """Declared fixed PD baseline operating on the shared canonical state schema."""

    def __init__(
        self, robot: str, config: Mapping[str, Any], limits: Any, target: Any
    ) -> None:
        import torch

        self.robot, self.limits, self.target = robot, limits, target
        if config.get("kind") != "pd":
            raise ValueError(
                "Historical runtime currently supports policy.kind=pd only; actor artifacts are unsupported"
            )
        allowed = {"kind", "gains"} if robot == "cartpole" else {"kind", "kp", "kd"}
        if set(config) - allowed:
            raise ValueError("Unknown historical PD setting")
        self.identity = {
            "kind": "pd",
            "implementation": "embodichain-reference-feedback-v1",
            "observation_schema": (
                "cartpole-state-4-v1" if robot == "cartpole" else "ant-com-state-27-v1"
            ),
            "deterministic": True,
            "scope": "fixed_baseline_not_paper_policy",
            "effort_limits": limits.detach().cpu().tolist(),
        }
        if robot == "cartpole":
            values = config.get("gains", [5.0, 50.0, 5.0, 20.0])
            if (
                not isinstance(values, Sequence)
                or len(values) != 4
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    for x in values
                )
            ):
                raise ValueError("Cartpole gains must be four finite values")
            self.gains = torch.tensor(values, device=limits.device)
            self.identity["gains"] = list(values)
        else:
            self.kp, self.kd = config.get("kp", 20.0), config.get("kd", 1.0)
            if any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x < 0
                for x in (self.kp, self.kd)
            ):
                raise ValueError("Ant gains must be finite and nonnegative")
            self.identity.update(
                kp=float(self.kp),
                kd=float(self.kd),
                target=target.detach().cpu().tolist(),
            )
        self.identity["sha256"] = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()

    def __call__(self, q: Any, qd: Any) -> Any:
        import torch
        from embodichain.compute.dynamics import reference_task_values

        observation, _, _ = reference_task_values(self.robot, q, qd)
        finite = torch.isfinite(observation).all(-1)
        if self.robot == "cartpole":
            force = torch.zeros_like(qd)
            force[:, 0] = (
                -(torch.nan_to_num(observation) * self.gains)
                .sum(-1)
                .clamp(-self.limits[0], self.limits[0])
            )
        else:
            force = (
                self.kp * (self.target - torch.nan_to_num(q[:, 7:]))
                - self.kd * torch.nan_to_num(qd[:, 6:])
            ).clamp(-self.limits, self.limits)
        return torch.where(finite[:, None], force, 0)


def evaluate_reference_rollouts(
    spec: Mapping[str, Any], *, trust_external_code: bool = False
) -> dict[str, Any]:
    """Run original physics and original neural integration in their historical domain.

    Args:
        spec: source checkout, portable model export, robot, output JSON path,
            device, num_envs, trajectories, horizons, seed, motion_mode and
            initial_states(default/sample). Fixed-policy mode requires a PD policy
            mapping. verify_portable_steps bounds an extra equivalence probe.
            teacher_dataset optionally selects immutable evaluation traces made
            by the standalone generator in the original supported environment.
        trust_external_code: Explicit permission to execute the pinned external
            checkout. Run in a fresh subprocess with its legacy Warp on PYTHONPATH.

    Returns:
        Measured error curves, returns, failures and hashed trajectory/source
        receipts compatible with qualification report attachment. This provides
        no evidence of deployment compatibility with current Newton.
    """
    if not trust_external_code:
        raise ValueError("Historical runtime requires trust_external_code=True")
    allowed = {
        "source",
        "model",
        "robot",
        "output",
        "device",
        "num_envs",
        "trajectories",
        "horizons",
        "seed",
        "motion_mode",
        "initial_states",
        "policy",
        "warp_version",
        "cpu_threads",
        "verify_portable_steps",
        "teacher_dataset",
    }
    if set(spec) - allowed:
        raise ValueError(
            f"Unknown historical rollout settings: {sorted(set(spec)-allowed)}"
        )
    robot = spec["robot"]
    if robot not in {"cartpole", "ant"}:
        raise ValueError("Historical rollout supports cartpole and ant")
    batch = _positive("num_envs", spec.get("num_envs", 8))
    count = _positive("trajectories", spec.get("trajectories", 2048))
    seed = _positive("seed", spec.get("seed", 500), 0)
    horizons = sorted(
        {_positive("horizon", h) for h in spec.get("horizons", [1, 10, 100, 1000])}
    )
    if not horizons:
        raise ValueError("horizons must not be empty")
    horizon = horizons[-1]
    mode = spec.get("motion_mode", "random_effort")
    if mode not in {"passive", "random_effort", "fixed_policy"}:
        raise ValueError("Unknown motion_mode")
    if (mode == "fixed_policy") != ("policy" in spec):
        raise ValueError(
            "fixed_policy requires an explicit PD policy only in that mode"
        )
    initial = spec.get("initial_states", "sample")
    if initial not in {"default", "sample"}:
        raise ValueError("initial_states must be default or sample")
    expected_warp = "1.5.1" if robot == "ant" else "1.8.0"
    if spec.get("warp_version", expected_warp) != expected_warp:
        raise ValueError("Historical robot requires its pinned Warp version")
    identity = reference_identity(spec["source"])
    import warp as wp

    if str(wp.__version__) != expected_warp:
        raise ValueError(
            f"Run in isolated Warp {expected_warp}; loaded {wp.__version__}"
        )
    sys.path.insert(0, identity["root"])
    import torch
    import numpy as np
    import h5py
    from envs.neural_environment import NeuralEnvironment
    from generate.trajectory_sampler import TrajectorySampler
    from generate.trajectory_sampler_ant import TrajectorySamplerAnt
    from utils.commons import (
        JOINT_Q_MIN,
        JOINT_Q_MAX,
        JOINT_QD_MIN,
        JOINT_QD_MAX,
        JOINT_ACT_SCALE,
    )
    from utils.python_utils import set_random_seed
    from embodichain.compute.dynamics import NeRDStateCodec, reference_task_values
    from embodichain.compute.dynamics._evaluation import (
        _error_values,
        _safe_number,
        _teacher_row_sha256,
    )
    from embodichain.learning.models.nerd import load_model
    from .reference import _restore_official_model

    device = str(spec.get("device", "cpu"))
    torch.set_num_threads(_positive("cpu_threads", spec.get("cpu_threads", 1)))
    model_path = Path(spec["model"])
    output = Path(spec["output"])
    model_sha, manifest_sha = _hash(model_path / "weights.pt"), _hash(
        model_path / "manifest.json"
    )
    common, manifest = load_model(model_path, device=device)
    if (
        next(common.parameters()).dtype != torch.float32
        or manifest.get("codec", {}).get("robot") != robot
    ):
        raise ValueError("Historical runtime requires a matching FP32 model")
    codec = NeRDStateCodec(**manifest["codec"])
    root_half = math.sqrt(0.5)
    expected_rotation = (
        [0.5, 0.5, 0.5, 0.5]
        if robot == "cartpole"
        else [root_half, 0.0, 0.0, root_half]
    )
    rotation = torch.tensor(
        manifest.get("world_rotation_xyzw") or expected_rotation,
        device=device,
        dtype=torch.float32,
    )
    if not torch.allclose(
        rotation, rotation.new_tensor(expected_rotation), atol=1e-6, rtol=0
    ):
        raise ValueError("Unsupported historical-to-canonical world rotation")
    verify_steps = min(
        horizon,
        _positive(
            "verify_portable_steps",
            spec.get("verify_portable_steps", common.cfg.history_length + 1),
        ),
    )
    trace_path = output.with_name(output.stem + "-evidence.h5")
    rows_path = output.with_name(output.stem + "-trajectories.jsonl")
    for path in (output, trace_path, rows_path):
        if path.exists():
            raise FileExistsError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_files = (
        "envs/neural_environment.py",
        "envs/abstract_contact_environment.py",
        "envs/warp_sim_envs/environment.py",
        "envs/warp_sim_envs/env_cartpole.py",
        "envs/warp_sim_envs/env_ant.py",
        "integrators/integrator_neural.py",
        "integrators/integrator_neural_stateful.py",
        "integrators/integrator_neural_transformer.py",
        "generate/trajectory_sampler.py",
        "generate/trajectory_sampler_ant.py",
        "generate/simulation_sampler.py",
    )
    identity["runtime_files"] = {
        name: _hash(Path(identity["root"]) / name) for name in runtime_files
    }
    identity["adapter"] = (
        "Original NeuralEnvironment, TransformerNeuralIntegrator, sampler and torque APIs; portable weights restored into original model; canonical coordinate transport and metric collection only"
    )
    integrator_cfg = {
        "name": "TransformerNeuralIntegrator",
        "num_states_history": common.cfg.history_length,
        "states_frame": "body" if robot == "ant" else "world",
        "anchor_frame_step": "every",
        "states_embedding_type": "identical",
        "prediction_type": "relative",
        "orientation_prediction_parameterization": "quaternion",
    }

    def environment(model=None):
        return NeuralEnvironment(
            env_name="Ant" if robot == "ant" else "Cartpole",
            num_envs=batch,
            warp_env_cfg={
                "seed": seed,
                "random_reset": False,
                "env_offset": (0.0, 0.0, 0.0),
                "use_graph_capture": False,
            },
            neural_integrator_cfg=integrator_cfg,
            neural_model=model,
            default_env_mode="ground-truth" if model is None else "neural",
            device=device,
            render=False,
        )

    class PortableAdapter(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.common = model

        def evaluate(self, inputs):
            return self.common.evaluate(
                {
                    "joint_f" if key == "joint_acts" else key: value
                    for key, value in inputs.items()
                }
            )

        def init_rnn(self, batch_size):
            del batch_size

    teacher = student = portable = None
    started = time.perf_counter()
    sums = torch.zeros(horizon, 4, device=device, dtype=torch.float64)
    counts = torch.zeros(horizon, 5, device=device, dtype=torch.long)
    returns = torch.zeros(horizon, 2, device=device, dtype=torch.float64)
    hsum = {h: torch.zeros(4, device=device, dtype=torch.float64) for h in horizons}
    hcount = {h: torch.zeros((), device=device, dtype=torch.long) for h in horizons}
    selected_digest = hashlib.sha256()
    parity_max = 0.0
    parity_nonfinite = False
    teacher_dataset = None
    teacher_feedback_max_abs_error = 0.0
    teacher_transport_max_abs_error = 0.0
    teacher_trace_physical = None
    names = ("q_mae", "qd_mae", "base_position_error", "base_orientation_error")
    with wp.ScopedDevice(device):
        try:
            teacher = environment()
            expected_inputs = {
                "states_embedding": teacher.state_dim,
                "joint_f": teacher.joint_act_dim,
            }
            if robot == "ant":
                slots = teacher.num_contacts_per_env
                expected_inputs.update(
                    contact_normals=3 * slots,
                    contact_points_1=3 * slots,
                    contact_depths=slots,
                    gravity_dir=3,
                )
            if (
                dict(common.cfg.input_dims) != expected_inputs
                or common.cfg.output_dim != teacher.state_dim
            ):
                raise ValueError(
                    "Model feature/target schema does not match original runtime"
                )
            physical = _verify_historical_physics(
                teacher, manifest, identity, expected_warp
            )
            original = _restore_official_model(common, device=device)
            student = environment(original)
            _verify_historical_physics(student, manifest, identity, expected_warp)
            portable = environment(PortableAdapter(common))
            _verify_historical_physics(portable, manifest, identity, expected_warp)
            name = teacher.robot_name
            sampler_type = TrajectorySamplerAnt if robot == "ant" else TrajectorySampler
            sampler = sampler_type(
                teacher,
                JOINT_Q_MIN[name],
                JOINT_Q_MAX[name],
                JOINT_QD_MIN[name],
                JOINT_QD_MAX[name],
                JOINT_ACT_SCALE[name],
                contact_prob=0.0,
            )
            default = torch.cat(
                (
                    wp.to_torch(teacher.model.joint_q).reshape(batch, -1),
                    wp.to_torch(teacher.model.joint_qd).reshape(batch, -1),
                ),
                -1,
            )[0].clone()
            limits = sampler.joint_act_scale
            canonical_default, _ = _canonical_states(default[None], codec, rotation)
            policy = (
                _HistoricalPD(robot, spec["policy"], limits, canonical_default[0, 7:])
                if mode == "fixed_policy"
                else None
            )
            if spec.get("teacher_dataset") is not None:
                from ._reference_traces import (
                    _load_reference_trace,
                    _reference_trace_batches,
                )

                teacher_directory = Path(spec["teacher_dataset"])
                teacher_dataset, teacher_records = _load_reference_trace(
                    teacher_directory,
                    spec,
                    identity,
                    manifest,
                    None if policy is None else policy.identity,
                    count,
                    horizon,
                )
                teacher_trace_physical = _verify_historical_physics(
                    teacher, teacher_dataset["metadata"], identity, expected_warp
                )
                if not teacher_trace_physical["complete"]:
                    raise ValueError(
                        "Teacher trace is missing measured physical fields"
                    )
                batches = _reference_trace_batches(
                    teacher_directory, teacher_records, batch, horizon, robot, device
                )
            else:
                batches = (None for _ in range(0, count, batch))
            with (
                _cpu_stream_compatibility(wp, device),
                torch.no_grad(),
                h5py.File(trace_path, "w") as trace,
                rows_path.open("w") as rows_file,
            ):
                for start, recorded in zip(range(0, count, batch), batches):
                    n = min(batch, count - start)
                    cached = None if recorded is None else recorded[1]
                    if cached is None:
                        samples = []
                        commands = []
                        for trajectory in range(start, start + n):
                            set_random_seed(seed + trajectory)
                            raw = default[None].clone()
                            if initial == "sample":
                                sampler.sample_initial_states(1, raw)
                            command = torch.zeros(
                                horizon, teacher.joint_act_dim, device=device
                            )
                            if mode == "random_effort":
                                sampler.sampler.sample(
                                    horizon, -limits, limits, command
                                )
                            samples.append(raw[0])
                            commands.append(command)
                        samples.extend([samples[-1]] * (batch - n))
                        commands.extend([commands[-1]] * (batch - n))
                        initial_raw = torch.stack(samples)
                        command = torch.stack(commands)
                        teacher.reset(initial_states=initial_raw.clone())
                    else:
                        initial_raw = cached["raw"][:, 0].contiguous()
                        command = cached["raw_joint_f"]
                        transported = _canonical_states(cached["raw"], codec, rotation)
                        for actual, key in zip(transported, ("q", "qd")):
                            if not torch.allclose(
                                actual,
                                cached[key],
                                atol=1e-5,
                                rtol=1e-4,
                                equal_nan=True,
                            ):
                                raise ValueError(
                                    "Recorded raw/canonical teacher state transport differs"
                                )
                            error = (actual - cached[key]).abs()
                            teacher_transport_max_abs_error = max(
                                teacher_transport_max_abs_error,
                                float(
                                    torch.where(torch.isfinite(error), error, 0).max()
                                ),
                            )
                        if mode == "passive" and bool((command != 0).any()):
                            raise ValueError(
                                "Passive teacher trace contains nonzero effort"
                            )
                    student.reset(initial_states=initial_raw.clone())
                    if start == 0:
                        portable.reset(initial_states=initial_raw.clone())
                    teacher_raw = [teacher.states.clone()] if cached is None else None
                    student_raw = [student.states.clone()]
                    teacher_controls = []
                    student_controls = []
                    for step in range(horizon):
                        tq, tqd = (
                            _canonical_states(teacher.states, codec, rotation)
                            if cached is None
                            else (cached["q"][:, step], cached["qd"][:, step])
                        )
                        nq, nqd = _canonical_states(student.states, codec, rotation)
                        tf = (
                            command[:, step]
                            if policy is None or cached is not None
                            else policy(tq, tqd)
                        )
                        nf = command[:, step] if policy is None else policy(nq, nqd)
                        if cached is not None and policy is not None:
                            expected = policy(tq, tqd)
                            if not torch.allclose(tf, expected, atol=1e-5, rtol=1e-4):
                                raise ValueError(
                                    "Recorded original teacher actions differ from declared feedback"
                                )
                            teacher_feedback_max_abs_error = max(
                                teacher_feedback_max_abs_error,
                                float((tf - expected).abs().max()),
                            )
                        # Original torque API uses view(-1), so transport a
                        # contiguous frame from batch-major command storage.
                        tf, nf = tf.contiguous(), nf.contiguous()
                        teacher_controls.append(tf.clone())
                        student_controls.append(nf.clone())
                        if cached is None:
                            teacher_raw.append(
                                teacher.step_with_joint_act(
                                    tf, env_mode="ground-truth"
                                ).clone()
                            )
                        student_raw.append(
                            student.step_with_joint_act(nf, env_mode="neural").clone()
                        )
                        if start == 0 and step < verify_steps:
                            alternate = portable.step_with_joint_act(
                                nf, env_mode="neural"
                            ).clone()
                            torch.testing.assert_close(
                                alternate,
                                student_raw[-1],
                                atol=1e-5,
                                rtol=1e-4,
                                equal_nan=True,
                            )
                            difference = (alternate - student_raw[-1]).abs()
                            parity_nonfinite = parity_nonfinite or not bool(
                                torch.isfinite(difference).all()
                            )
                            parity_max = max(
                                parity_max,
                                float(
                                    torch.nan_to_num(
                                        difference, nan=0.0, posinf=0.0
                                    ).max()
                                ),
                            )
                    traw = (
                        torch.stack(teacher_raw, 1)[:n]
                        if cached is None
                        else cached["raw"][:n]
                    )
                    nraw = torch.stack(student_raw, 1)[:n]
                    tq, tqd = (
                        _canonical_states(traw, codec, rotation)
                        if cached is None
                        else (cached["q"][:n], cached["qd"][:n])
                    )
                    q, qd = _canonical_states(nraw, codec, rotation)
                    tc = torch.stack(teacher_controls, 1)[:n]
                    nc = torch.stack(student_controls, 1)[:n]
                    if robot == "ant":
                        tc = torch.cat((tc.new_zeros(n, horizon, 6), tc), -1)
                        nc = torch.cat((nc.new_zeros(n, horizon, 6), nc), -1)
                    tb = ~(torch.isfinite(tq).all(-1) & torch.isfinite(tqd).all(-1))
                    nb = ~(torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1))
                    tn = tb.cummax(1).values[:, 1:]
                    nn = nb.cummax(1).values[:, 1:]
                    valid = ~(tn | nn)
                    _, treward, tfail = reference_task_values(robot, tq, tqd)
                    _, nreward, nfail = reference_task_values(robot, q, qd)
                    tfail = (tfail | tb | ~torch.isfinite(treward)).cummax(1).values
                    nfail = (nfail | nb | ~torch.isfinite(nreward)).cummax(1).values
                    tret = torch.where(
                        ~tfail[:, :-1] & torch.isfinite(treward[:, 1:]),
                        treward[:, 1:],
                        0,
                    ).cumsum(1)
                    nret = torch.where(
                        ~nfail[:, :-1] & torch.isfinite(nreward[:, 1:]),
                        nreward[:, 1:],
                        0,
                    ).cumsum(1)
                    error = torch.where(
                        valid[..., None],
                        _error_values(
                            robot, q[:, 1:], qd[:, 1:], tq[:, 1:], tqd[:, 1:]
                        ),
                        0,
                    )
                    sums += error.double().sum(0)
                    counts += torch.stack(
                        (
                            valid.sum(0),
                            tn.sum(0),
                            nn.sum(0),
                            tfail[:, 1:].sum(0),
                            nfail[:, 1:].sum(0),
                        ),
                        -1,
                    )
                    returns += torch.stack(
                        (tret.double().sum(0), nret.double().sum(0)), -1
                    )
                    cumulative = error.cumsum(1)
                    for h in horizons:
                        hsum[h] += (
                            torch.where(
                                valid[:, h - 1, None], cumulative[:, h - 1] / h, 0
                            )
                            .double()
                            .sum(0)
                        )
                        hcount[h] += valid[:, h - 1].sum()
                    arrays = {
                        key: value.detach().cpu().numpy()
                        for key, value in {
                            "teacher_raw": traw,
                            "neural_raw": nraw,
                            "teacher_q": tq,
                            "teacher_qd": tqd,
                            "q": q,
                            "qd": qd,
                            "teacher_joint_f": tc,
                            "joint_f": nc,
                            "errors": error,
                            "valid": valid,
                            "teacher_nonfinite": tn,
                            "neural_nonfinite": nn,
                            "teacher_failed": tfail[:, 1:],
                            "neural_failed": nfail[:, 1:],
                            "teacher_return": tret,
                            "neural_return": nret,
                        }.items()
                    }
                    group = trace.create_group(str(start // batch))
                    for key, value in arrays.items():
                        group.create_dataset(key, data=value, compression="lzf")
                    for row in range(n):
                        record = {
                            "id": f"{seed}:{start+row}",
                            "metadata": {"sampling_seed": seed + start + row},
                        }
                        digest = _teacher_row_sha256(
                            record,
                            [
                                arrays["teacher_q"][row],
                                arrays["teacher_qd"][row],
                                arrays["teacher_joint_f"][row],
                            ],
                        )
                        selected_digest.update(bytes.fromhex(digest))
                        evidence = {
                            "trajectory_id": record["id"],
                            "sampling_seed": seed + start + row,
                            "reference_sha256": digest,
                            "evidence_group": str(start // batch),
                            "evidence_row": row,
                            "horizons": {},
                        }
                        for h in horizons:
                            mean = (
                                arrays["errors"][row, :h].mean(0)
                                if arrays["valid"][row, h - 1]
                                else [float("nan")] * 4
                            )
                            evidence["horizons"][str(h)] = {
                                **{
                                    key: _safe_number(value)
                                    for key, value in zip(names, mean)
                                },
                                "teacher_nonfinite": bool(
                                    arrays["teacher_nonfinite"][row, h - 1]
                                ),
                                "neural_nonfinite": bool(
                                    arrays["neural_nonfinite"][row, h - 1]
                                ),
                                "teacher_failed": bool(
                                    arrays["teacher_failed"][row, h - 1]
                                ),
                                "neural_failed": bool(
                                    arrays["neural_failed"][row, h - 1]
                                ),
                                "teacher_policy_return": _safe_number(
                                    arrays["teacher_return"][row, h - 1]
                                ),
                                "policy_return": _safe_number(
                                    arrays["neural_return"][row, h - 1]
                                ),
                            }
                        rows_file.write(json.dumps(evidence, allow_nan=False) + "\n")
            origin = {
                "kind": "official_warp",
                "source_commit": OFFICIAL_COMMIT,
                "warp_version": expected_warp,
                "integrator": str(teacher.integrator_type_gt),
                "num_substeps": teacher.sim_substeps_gt,
                "up_axis": int(teacher.model.up_axis),
                "bodies_per_env": teacher.bodies_per_env,
                "device": device,
                "physical_arrays_sha256": physical["sha256"],
                "gravity": physical["gravity_live"],
            }
        finally:
            for value in (portable, student, teacher):
                if value is not None:
                    value.close()
    if (
        _hash(model_path / "weights.pt") != model_sha
        or _hash(model_path / "manifest.json") != manifest_sha
    ):
        raise ValueError("Model export changed during historical evaluation")
    curve = []
    means = (sums / counts[:, 0].clamp_min(1)[:, None]).cpu().numpy()
    count_array = counts.cpu().numpy()
    return_array = (returns / count).cpu().numpy()
    for step in range(horizon):
        row = count_array[step]
        curve.append(
            {
                "step": step + 1,
                **{
                    key: (_safe_number(v) if row[0] else None)
                    for key, v in zip(names, means[step])
                },
                "valid_trajectories": int(row[0]),
                "teacher_nonfinite_count": int(row[1]),
                "neural_nonfinite_count": int(row[2]),
                "teacher_failed_count": int(row[3]),
                "failed_trajectories": int(row[4]),
                "teacher_nonfinite_rate": float(row[1] / count),
                "nonfinite_rate": float(row[2] / count),
                "teacher_failure_rate": float(row[3] / count),
                "failure_rate": float(row[4] / count),
                "teacher_policy_return": float(return_array[step, 0]),
                "policy_return": float(return_array[step, 1]),
            }
        )
    metrics = {}
    for h in horizons:
        number = int(hcount[h].cpu())
        mean = (hsum[h] / max(number, 1)).cpu().tolist()
        metrics[str(h)] = {
            **curve[h - 1],
            **{
                key: (_safe_number(v) if number else None)
                for key, v in zip(names, mean)
            },
            "valid_trajectories": number,
            "numerically_invalid_trajectories": count - number,
        }
    result = {
        "status": "completed",
        "qualification_status": (
            "runtime_measurements_only"
            if physical["complete"]
            else "pending_missing_recorded_physics"
        ),
        "scope": "historical_official_warp_runtime_training_quality_only_not_current_newton_deployment",
        "robot": robot,
        "motion_mode": mode,
        "seed": seed,
        "model": str(model_path.resolve()),
        "model_sha256": model_sha,
        "model_manifest_sha256": manifest_sha,
        "teacher_origin": origin,
        "teacher_split": "test",
        "teacher_dataset": (
            None
            if teacher_dataset is None
            else str(Path(spec["teacher_dataset"]).resolve())
        ),
        "teacher_dataset_manifest_sha256": (
            None
            if teacher_dataset is None
            else _hash(Path(spec["teacher_dataset"]) / "manifest.json")
        ),
        "teacher_generation_environment": (
            None
            if teacher_dataset is None
            else teacher_dataset["metadata"]["generation_environment"]
        ),
        "teacher_feedback_verification": (
            "all_selected_actions_verified_against_teacher_states"
            if teacher_dataset is not None and policy is not None
            else "live_feedback_execution" if policy is not None else None
        ),
        "teacher_feedback_max_abs_error": teacher_feedback_max_abs_error,
        "teacher_transport_max_abs_error": teacher_transport_max_abs_error,
        "teacher_physical_verification": teacher_trace_physical,
        "trajectories": count,
        "num_envs": batch,
        "state_dtype": "float32",
        "selected_teacher_sha256": selected_digest.hexdigest(),
        "selected_teacher_hash_schema": "ordered-teacher-content-v1",
        "runtime_identity": {
            "engine": "official_warp",
            "reference_commit": OFFICIAL_COMMIT,
            "device": device,
            "state_dtype": "float32",
            "python": platform.python_version(),
            "versions": {
                "torch": str(torch.__version__),
                "warp-lang": str(wp.__version__),
            },
            "cuda_runtime": torch.version.cuda,
        },
        "reference": identity,
        "physical_verification": physical,
        "world_rotation_xyzw": rotation.cpu().tolist(),
        "compatibility_adapters": (
            []
            if wp.get_device(device).is_cuda
            else [
                "wp.stream_to_torch(cpu)->None within scoped adapter; original simulate unchanged"
            ]
        ),
        "replica_placement": [0.0, 0.0, 0.0],
        "sampling": "original sample_initial_states/UniformSampler with per-trajectory seed; no rejection/resampling",
        "portable_runtime_equivalence": {
            "steps": verify_steps,
            "batch_size": batch,
            "atol": 1e-5,
            "rtol": 1e-4,
            "max_abs_error": parity_max,
            "matched_nonfinite": parity_nonfinite,
            "status": "matched_nonfinite" if parity_nonfinite else "passed",
        },
        "policy": None if policy is None else policy.identity,
        "policy_sha256": None if policy is None else policy.identity["sha256"],
        "horizons": metrics,
        "error_vs_time": curve,
        "trajectory_evidence": str(rows_path.resolve()),
        "trajectory_evidence_sha256": _hash(rows_path),
        "trace_evidence": str(trace_path.resolve()),
        "trace_evidence_sha256": _hash(trace_path),
        "failure_rate": curve[-1]["failure_rate"],
        "nonfinite_rate": curve[-1]["nonfinite_rate"],
        "teacher_failure_rate": curve[-1]["teacher_failure_rate"],
        "teacher_nonfinite_rate": curve[-1]["teacher_nonfinite_rate"],
        "policy_return": curve[-1]["policy_return"] if policy else None,
        "teacher_policy_return": curve[-1]["teacher_policy_return"] if policy else None,
        "metric_population": "Error means condition on no numerical failure through the horizon; physical failures stay included. Returns stop after first failure. All requested seeds are retained.",
        "adapter_source_sha256": _hash(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Run the isolated historical qualification worker from an explicit spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    result = evaluate_reference_rollouts(
        json.loads(args.spec.read_text()), trust_external_code=args.trust_external_code
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
