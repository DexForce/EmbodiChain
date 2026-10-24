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

"""Replay identical canonical states/efforts in isolated Warp and Newton.

Run one worker per engine/version; compare the saved NPZ arrays, never live
simulators in one process. The old source remains pinned and unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

__all__: list[str] = []


def _compare(args: argparse.Namespace) -> dict:
    import torch
    from embodichain.compute.dynamics._evaluation import _error_values

    if args.compare_with is None:
        raise ValueError("compare requires --compare-with reference.npz")
    paths = (args.compare_with, args.inputs)
    receipts = [json.loads(path.with_suffix(".json").read_text()) for path in paths]
    if any(row["robot"] != args.robot for row in receipts):
        raise ValueError("Trace robot identities differ")
    if receipts[0]["inputs_sha256"] != receipts[1]["inputs_sha256"]:
        raise ValueError("Traces do not share identical input content")
    for path, receipt in zip(paths, receipts):
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["trace_sha256"]:
            raise ValueError(f"Trace checksum mismatch: {path}")
    with np.load(paths[0]) as ref, np.load(paths[1]) as actual:
        q, qd, rq, rqd = [
            torch.from_numpy(value.copy())
            for value in (actual["q"], actual["qd"], ref["q"], ref["qd"])
        ]
    if q.shape != rq.shape or qd.shape != rqd.shape:
        raise ValueError("Trace shapes differ")
    if args.robot == "ant":
        torch.testing.assert_close(q[0, :, :3], rq[0, :, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(q[0, :, 7:], rq[0, :, 7:], rtol=1e-5, atol=1e-6)
        a = q[0, :, 3:7].double()
        b = rq[0, :, 3:7].double()
        a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
        if not bool(torch.all(torch.abs((a * b).sum(-1)) > 1 - 1e-10)):
            raise ValueError("Initial root orientations differ")
    else:
        torch.testing.assert_close(q[0], rq[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(qd[0], rqd[0], rtol=1e-5, atol=1e-6)
    errors = _error_values(args.robot, q, qd, rq, rqd)
    finite = torch.isfinite(torch.cat((q, qd, rq, rqd), -1)).all(-1)
    results = {}
    for h in (1, 10, 100, 1000):
        if h >= q.shape[0]:
            continue
        valid = finite[: h + 1].all(0)
        results[str(h)] = {
            "finite_trajectories": int(valid.sum()),
            "nonfinite_trajectories": int((~valid).sum()),
            "errors": (
                None
                if not bool(valid.all())
                else dict(
                    zip(
                        (
                            "q_mae",
                            "qd_mae",
                            "base_position_error",
                            "base_orientation_error_rad",
                        ),
                        errors[1 : h + 1].mean((0, 1)).tolist(),
                    )
                )
            ),
        }
    return {
        "robot": args.robot,
        "reference": receipts[0],
        "actual": receipts[1],
        "horizons": results,
        "population": "All requested trajectories; errors remain undefined when any trajectory becomes nonfinite.",
    }


def _inputs(
    robot: str,
    count: int,
    horizon: int,
    motion_mode: str = "random_effort",
    initial_states: str = "near_upright",
    effort_scale: float = 0.02,
) -> dict[str, np.ndarray]:
    from embodichain.lab.sim.physics.solvers.collection import _sample

    rows = [
        _sample(
            robot,
            42 + i,
            horizon,
            motion_mode,
            "sample" if initial_states == "sample" else "default",
            effort_scale,
        )
        for i in range(count)
    ]
    q, qd, efforts = [np.stack([row[j].numpy() for row in rows]) for j in range(3)]
    rng = np.random.default_rng(42)
    if initial_states == "sample":
        return {"q": q, "qd": qd, "efforts": efforts}
    if robot == "cartpole":
        q[:, 0] = rng.uniform(-0.25, 0.25, count)
        q[:, 1] = rng.uniform(-0.4, 0.4, count)
        qd[:] = rng.uniform(-0.3, 0.3, qd.shape)
    else:
        q[:, 2] = np.resize([0.55, 1.0, 3.0, 1.0], count)
        q[:, 7:] += rng.uniform(-0.03, 0.03, (count, 8))
        qd[:, 6:] = rng.uniform(-0.3, 0.3, (count, 8))
    return {"q": q, "qd": qd, "efforts": efforts}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _physical_arrays(model: Any) -> dict[str, Any]:
    """Record measured parameters without requiring cross-version equality."""
    result = {}
    for name in (
        "body_mass",
        "body_com",
        "body_inertia",
        "joint_armature",
        "joint_target_ke",
        "joint_target_kd",
        "joint_damping",
    ):
        value = (
            model.get(name) if isinstance(model, dict) else getattr(model, name, None)
        )
        if value is not None:
            result[name] = _numpy(value).tolist()
    return result


def _legacy(
    args: argparse.Namespace, data: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict]:
    from embodichain.learning.dynamics.reference import reference_identity

    source = reference_identity(args.source)
    sys.path.insert(0, source["root"])
    import warp as wp
    from scipy.spatial.transform import Rotation
    from utils.env_utils import create_abstract_contact_env

    expected = "1.8.0" if args.robot == "cartpole" else "1.5.1"
    if wp.__version__ != expected:
        raise ValueError(f"Expected Warp {expected}, found {wp.__version__}")
    env = create_abstract_contact_env(
        "Cartpole" if args.robot == "cartpole" else "Ant",
        args.count,
        device="cpu",
        random_reset=False,
    )
    q, qd = data["q"].copy(), data["qd"].copy()
    rotation = Rotation.from_euler("x", np.pi / 2)
    if args.robot == "ant":
        q[:, :3] = rotation.inv().apply(q[:, :3])
        q[:, 3:7] = (rotation.inv() * Rotation.from_quat(q[:, 3:7])).as_quat()
        omega = rotation.inv().apply(qd[:, 3:6])
        linear = rotation.inv().apply(qd[:, :3]) - np.cross(omega, q[:, :3])
        qd[:, :6] = np.concatenate((omega, linear), axis=-1)
    env.state_0.joint_q.assign(q.astype(np.float32).ravel())
    env.state_0.joint_qd.assign(qd.astype(np.float32).ravel())
    wp.sim.eval_fk(
        env.model, env.state_0.joint_q, env.state_0.joint_qd, None, env.state_0
    )
    qs, vs, bodies = [], [], []
    body_count = env.model.body_count // args.count
    names = [str(name).rsplit("/", 1)[-1] for name in env.model.body_name[:body_count]]
    if len(names) != len(set(names)):
        raise ValueError("Legacy body names are not unique within one world")
    for step in range(args.horizon + 1):
        a = env.state_0.joint_q.numpy().reshape(args.count, -1).copy()
        v = env.state_0.joint_qd.numpy().reshape(args.count, -1).copy()
        if args.robot == "ant":
            v[:, :6] = np.concatenate(
                (
                    rotation.apply(v[:, 3:6] + np.cross(v[:, :3], a[:, :3])),
                    rotation.apply(v[:, :3]),
                ),
                axis=-1,
            )
            a[:, :3] = rotation.apply(a[:, :3])
            a[:, 3:7] = (rotation * Rotation.from_quat(a[:, 3:7])).as_quat()
        qs.append(a)
        vs.append(v)
        body = env.state_0.body_q.numpy().reshape(args.count, body_count, 7).copy()
        flat = body.reshape(-1, 7)
        flat[:, :3] = rotation.apply(flat[:, :3])
        # Invalid simulator quaternions remain in the trace and finite gate.
        valid = np.isfinite(flat[:, 3:]).all(-1) & (
            np.linalg.norm(flat[:, 3:], axis=-1) > 0
        )
        flat[valid, 3:] = (rotation * Rotation.from_quat(flat[valid, 3:])).as_quat()
        bodies.append(body)
        if step < args.horizon:
            force = data["efforts"][:, step]
            env.control.joint_act.assign(
                (force[:, 6:] if args.robot == "ant" else force).ravel()
            )
            env.update()
    return (
        {"q": np.stack(qs), "qd": np.stack(vs), "body_q": np.stack(bodies)},
        {
            "source": source,
            "warp": wp.__version__,
            "substeps": env.sim_substeps,
            "mass_update_interval": env.integrator.update_mass_matrix_every,
            "gravity": list(env.model.gravity),
            "body_names": names,
            "body_pose_frame": "Z-up world; legacy Y-up poses rotated +pi/2 around X",
            "physical_arrays": _physical_arrays(env.model),
        },
    )


def _current(
    args: argparse.Namespace, data: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict]:
    import torch
    import dexsim
    from embodichain.lab.sim.physics.solvers.collection import DynamicsSimulation

    substeps = args.substeps or (5 if args.robot == "cartpole" else 16)
    solver = (
        None
        if args.solver == "featherstone"
        else {"solver_type": "mujoco_warp", "use_mujoco_cpu": args.device == "cpu"}
    )
    if args.profile == "native" and solver is None:
        solver = {
            "solver_type": "featherstone",
            "update_mass_matrix_interval": substeps,
        }
    with DynamicsSimulation(
        args.robot,
        device=args.device,
        num_envs=args.count,
        physics_profile=args.profile,
        num_substeps=substeps,
        solver=solver,
    ) as env:
        env.reset(
            torch.from_numpy(data["q"]).to(env.device),
            torch.from_numpy(data["qd"]).to(env.device),
        )
        qs, vs, bodies = [], [], []
        model = dict(env.sim.spawn_result.dynamics_metadata())
        worlds = _numpy(model["body_world"])
        body_ids = np.flatnonzero(worlds == (0 if int(model["world_count"]) else -1))
        names = [str(model["body_label"][i]).rsplit("/", 1)[-1] for i in body_ids]
        if len(names) != len(set(names)):
            raise ValueError("Current body names are not unique within one world")
        all_ids = []
        for world in range(args.count):
            selected = np.flatnonzero(
                worlds == (world if int(model["world_count"]) else -1)
            )
            labels = {
                str(model["body_label"][i]).rsplit("/", 1)[-1]: int(i) for i in selected
            }
            all_ids.append([labels[name] for name in names])
        body_indices = torch.tensor(all_ids, device=env.device, dtype=torch.long)
        for step in range(args.horizon + 1):
            state, _ = env.snapshot()
            qs.append(_numpy(state["q"]).copy())
            vs.append(_numpy(state["qd"]).copy())
            physical = env.sim.spawn_result.dynamics_snapshot().physical
            bodies.append(physical["body_q"][body_indices].cpu().numpy().copy())
            if step < args.horizon:
                env.step(torch.from_numpy(data["efforts"][:, step]).to(env.device))
        metadata = {**env.manifest, "dexsim_commit": dexsim.__commit_id__}
        metadata.update(
            body_names=names,
            body_pose_frame="Z-up world",
            physical_arrays=_physical_arrays(model),
            actual_device=str(env.device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            actual_gpu_uuid=(
                str(torch.cuda.get_device_properties(env.device).uuid)
                if env.device.type == "cuda"
                else None
            ),
        )
    return {"q": np.stack(qs), "qd": np.stack(vs), "body_q": np.stack(bodies)}, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["legacy", "newton", "compare"], required=True
    )
    parser.add_argument("--robot", choices=["cartpole", "ant"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--trust-external-code", action="store_true")
    parser.add_argument(
        "--profile", choices=["native", "warp_reference"], default="warp_reference"
    )
    parser.add_argument(
        "--solver", choices=["featherstone", "mujoco_warp"], default="featherstone"
    )
    parser.add_argument("--substeps", type=int)
    parser.add_argument(
        "--motion-mode", choices=["passive", "random_effort"], default="random_effort"
    )
    parser.add_argument(
        "--initial-states", choices=["near_upright", "sample"], default="near_upright"
    )
    parser.add_argument("--effort-scale", type=float, default=0.02)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.engine == "legacy" and args.device != "cpu":
        parser.error("legacy execution remains pinned to --device cpu")
    if args.engine == "compare":
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(_compare(args), indent=2, allow_nan=False))
        print(args.output)
        return
    if min(args.count, args.horizon) < 1:
        parser.error("count and horizon must be positive")
    if args.engine == "legacy" and (
        not args.trust_external_code or args.source is None
    ):
        parser.error("legacy execution requires --source and --trust-external-code")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inputs.exists():
        with np.load(args.inputs) as rows:
            data = {k: rows[k] for k in rows.files}
    else:
        if args.engine != "newton":
            parser.error("generate shared inputs with the current worker first")
        data = _inputs(
            args.robot,
            args.count,
            args.horizon,
            args.motion_mode,
            args.initial_states,
            args.effort_scale,
        )
        args.inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.inputs, **data)
    if data["efforts"].shape[:2] != (args.count, args.horizon):
        raise ValueError("input shape differs from count/horizon")
    traces, meta = (_legacy if args.engine == "legacy" else _current)(args, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **traces)
    meta.update(
        engine=args.engine,
        profile=args.profile,
        robot=args.robot,
        inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        horizon=args.horizon,
        count=args.count,
        motion_mode=args.motion_mode,
        initial_states=args.initial_states,
        effort_scale=args.effort_scale,
        python=platform.python_version(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(args.output)


if __name__ == "__main__":
    main()
