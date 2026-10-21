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
from pathlib import Path
import sys

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


def _inputs(robot: str, count: int, horizon: int) -> dict[str, np.ndarray]:
    from embodichain.lab.sim.physics.solvers.collection import _sample

    rows = [
        _sample(robot, 42 + i, horizon, "random_effort", "default", 0.02)
        for i in range(count)
    ]
    q, qd, efforts = [np.stack([row[j].numpy() for row in rows]) for j in range(3)]
    rng = np.random.default_rng(42)
    if robot == "cartpole":
        q[:, 0] = rng.uniform(-0.25, 0.25, count)
        q[:, 1] = rng.uniform(-0.4, 0.4, count)
        qd[:] = rng.uniform(-0.3, 0.3, qd.shape)
    else:
        q[:, 2] = np.resize([0.55, 1.0, 3.0, 1.0], count)
        q[:, 7:] += rng.uniform(-0.03, 0.03, (count, 8))
        qd[:, 6:] = rng.uniform(-0.3, 0.3, (count, 8))
    return {"q": q, "qd": qd, "efforts": efforts}


def _legacy(
    args: argparse.Namespace, data: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict]:
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
    qs, vs = [], []
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
        if step < args.horizon:
            force = data["efforts"][:, step]
            env.control.joint_act.assign(
                (force[:, 6:] if args.robot == "ant" else force).ravel()
            )
            env.update()
    return (
        np.stack(qs),
        np.stack(vs),
        {
            "source": source,
            "warp": wp.__version__,
            "substeps": env.sim_substeps,
            "mass_update_interval": env.integrator.update_mass_matrix_every,
            "gravity": list(env.model.gravity),
        },
    )


def _current(
    args: argparse.Namespace, data: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    import dexsim
    from embodichain.lab.sim.physics.solvers.collection import DynamicsSimulation

    substeps = args.substeps or (5 if args.robot == "cartpole" else 16)
    solver = (
        None
        if args.solver == "featherstone"
        else {"solver_type": "mujoco_warp", "use_mujoco_cpu": True}
    )
    if args.profile == "native" and solver is None:
        solver = {
            "solver_type": "featherstone",
            "update_mass_matrix_interval": substeps,
        }
    with DynamicsSimulation(
        args.robot,
        num_envs=args.count,
        physics_profile=args.profile,
        num_substeps=substeps,
        solver=solver,
    ) as env:
        env.reset(torch.from_numpy(data["q"]), torch.from_numpy(data["qd"]))
        qs, vs = [], []
        for step in range(args.horizon + 1):
            state, _ = env.snapshot()
            qs.append(state["q"].numpy())
            vs.append(state["qd"].numpy())
            if step < args.horizon:
                env.step(torch.from_numpy(data["efforts"][:, step]))
        metadata = {**env.manifest, "dexsim_commit": dexsim.__commit_id__}
    return np.stack(qs), np.stack(vs), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["legacy", "newton", "compare"], required=True
    )
    parser.add_argument("--robot", choices=["cartpole", "ant"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--trust-external-code", action="store_true")
    parser.add_argument(
        "--profile", choices=["native", "warp_reference"], default="warp_reference"
    )
    parser.add_argument(
        "--solver", choices=["featherstone", "mujoco_warp"], default="featherstone"
    )
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        data = _inputs(args.robot, args.count, args.horizon)
        args.inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.inputs, **data)
    if data["efforts"].shape[:2] != (args.count, args.horizon):
        raise ValueError("input shape differs from count/horizon")
    q, qd, meta = (_legacy if args.engine == "legacy" else _current)(args, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, q=q, qd=qd)
    meta.update(
        engine=args.engine,
        profile=args.profile,
        robot=args.robot,
        inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        horizon=args.horizon,
        count=args.count,
    )
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(args.output)


if __name__ == "__main__":
    main()
