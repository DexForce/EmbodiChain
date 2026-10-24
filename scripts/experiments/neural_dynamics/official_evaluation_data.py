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

"""Freeze original-Warp evaluation data in Python3.8/Torch2.2, without the SDK.

Only pure Torch coordinate/observation modules are loaded under an isolated
namespace for canonical transport. Physics and sampling execute unmodified
upstream code. Memory is bounded by one B x (T+1) batch; failures are never filtered.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from typing import Any

__all__: list[str] = []
_COMMIT = "370e89d2b8ce68aa3578aa7812859c681b7502bc"


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _transport(directory: Path) -> tuple[Any, Any]:
    package = types.ModuleType("_isolated_nerd_transport")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    loaded = []
    for name in ("codec", "reference"):
        spec = importlib.util.spec_from_file_location(
            package.__name__ + "." + name, directory / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def _canonical(
    raw: Any, codec: Any, rotation: Any, math_module: Any
) -> tuple[Any, Any]:
    import torch

    q = raw[..., : codec.q_dim]
    qd = codec.legacy_to_native(q, raw[..., codec.q_dim :])
    if codec.robot == "ant":
        q = torch.cat(
            (
                math_module.quat_rotate(rotation, q[..., :3]),
                math_module.quat_multiply(rotation, q[..., 3:7]),
                q[..., 7:],
            ),
            -1,
        )
        qd = torch.cat(
            (
                math_module.quat_rotate(rotation, qd[..., :3]),
                math_module.quat_rotate(rotation, qd[..., 3:6]),
                qd[..., 6:],
            ),
            -1,
        )
    return q, qd


def _physics(model: Any) -> dict[str, Any]:
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
            result[name] = value.numpy().tolist()
    for name in ("mu", "ke", "kd", "kf"):
        result["shape_material_" + name] = (
            getattr(model.shape_materials, name).numpy().tolist()
        )
    result["shape_thickness"] = model.shape_geo.thickness.numpy().tolist()
    result["shape_body"] = model.shape_body.numpy().tolist()
    result["ground_plane"] = model.ground_plane.numpy().tolist()
    result["gravity"] = np.asarray(model.gravity, dtype=np.float64).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    allowed = {
        "source",
        "output_dir",
        "robot",
        "num_envs",
        "trajectories",
        "horizon",
        "seed",
        "initial_states",
        "motion_mode",
        "policy",
        "transport_root",
    }
    if set(cfg) - allowed:
        raise ValueError("Unknown original evaluation generation settings")
    robot, mode = cfg["robot"], cfg["motion_mode"]
    if robot not in {"cartpole", "ant"} or mode not in {
        "passive",
        "random_effort",
        "fixed_policy",
    }:
        raise ValueError("Unsupported robot or motion mode")
    if (mode == "fixed_policy") != ("policy" in cfg):
        raise ValueError("Only fixed_policy mode requires an explicit PD policy")
    batch, count, horizon, seed = (
        cfg[name] for name in ("num_envs", "trajectories", "horizon", "seed")
    )
    if (
        any(type(x) is not int or x < 1 for x in (batch, count, horizon))
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError(
            "Positive integer batch/count/horizon and nonnegative seed required"
        )
    if cfg.get("initial_states", "sample") not in {"default", "sample"}:
        raise ValueError("Unsupported initial state sampling")
    source = Path(cfg["source"]).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != _COMMIT or changed:
        raise ValueError("Original source must be the pinned unchanged checkout")
    out = Path(cfg["output_dir"])
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(source))
    import h5py
    import numpy as np
    import torch
    import warp as wp
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

    version = "1.5.1" if robot == "ant" else "1.8.0"
    if (
        platform.python_version() != "3.8.20"
        or torch.__version__.split("+")[0] != "2.2.2"
        or np.__version__ != "1.24.4"
        or wp.__version__ != version
    ):
        raise ValueError(
            "Use supported Python3.8.20/Torch2.2.2/NumPy1.24.4 and robot-pinned Warp"
        )
    torch.set_num_threads(1)
    transport = Path(
        cfg.get(
            "transport_root",
            Path(__file__).resolve().parents[3] / "embodichain/compute/dynamics",
        )
    )
    math_module, observations = _transport(transport)
    set_random_seed(seed)
    started = time.monotonic()
    with wp.ScopedDevice("cpu"):
        env = NeuralEnvironment(
            env_name=robot.capitalize(),
            num_envs=batch,
            neural_model=None,
            neural_integrator_cfg={
                "name": "TransformerNeuralIntegrator",
                "num_states_history": 10,
                "states_frame": "body" if robot == "ant" else "world",
                "anchor_frame_step": "every",
                "states_embedding_type": "identical",
                "prediction_type": "relative",
                "orientation_prediction_parameterization": "quaternion",
            },
            default_env_mode="ground-truth",
            warp_env_cfg={
                "seed": seed,
                "random_reset": False,
                "env_offset": (0.0, 0.0, 0.0),
            },
            device="cpu",
            render=False,
        )
        try:
            sampler_cls = TrajectorySamplerAnt if robot == "ant" else TrajectorySampler
            name = env.robot_name
            sampler = sampler_cls(
                env,
                JOINT_Q_MIN[name],
                JOINT_Q_MAX[name],
                JOINT_QD_MIN[name],
                JOINT_QD_MAX[name],
                JOINT_ACT_SCALE[name],
                contact_prob=0.0,
            )
            limits = sampler.joint_act_scale
            codec = math_module.NeRDStateCodec(
                robot, base_com=env.model.body_com.numpy()[0].tolist()
            )
            rotation = torch.tensor(
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
                if robot == "ant"
                else [0.5, 0.5, 0.5, 0.5]
            )
            default = torch.cat(
                (
                    wp.to_torch(env.model.joint_q).reshape(batch, -1),
                    wp.to_torch(env.model.joint_qd).reshape(batch, -1),
                ),
                -1,
            )[0].clone()
            target = default[7:15].clone()
            policy = cfg.get("policy")
            identity = None
            if policy is not None:
                keys = (
                    {"kind", "gains"} if robot == "cartpole" else {"kind", "kp", "kd"}
                )
                if policy.get("kind") != "pd" or set(policy) - keys:
                    raise ValueError("Only declared PD baseline settings supported")
                identity = {
                    "kind": "pd",
                    "implementation": "embodichain-reference-feedback-v1",
                    "observation_schema": (
                        "cartpole-state-4-v1"
                        if robot == "cartpole"
                        else "ant-com-state-27-v1"
                    ),
                    "deterministic": True,
                    "scope": "fixed_baseline_not_paper_policy",
                    "effort_limits": limits.tolist(),
                }
                if robot == "cartpole":
                    gains = policy.get("gains", [5.0, 50.0, 5.0, 20.0])
                    if len(gains) != 4 or any(
                        type(x) not in (int, float) or not math.isfinite(x)
                        for x in gains
                    ):
                        raise ValueError("Four finite PD gains required")
                    identity["gains"] = gains
                    gains = torch.tensor(gains)
                else:
                    kp, kd = policy.get("kp", 20.0), policy.get("kd", 1.0)
                    if any(
                        type(x) not in (int, float) or not math.isfinite(x) or x < 0
                        for x in (kp, kd)
                    ):
                        raise ValueError("Finite nonnegative PD gains required")
                    identity.update(kp=float(kp), kd=float(kd), target=target.tolist())
                identity["sha256"] = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
            runtime_files = [
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
            ]
            origin = {
                "kind": "official_warp",
                "source_commit": commit,
                "warp_version": wp.__version__,
                "device": "cpu",
                "integrator": str(env.integrator_type_gt),
                "num_substeps": env.sim_substeps_gt,
                "up_axis": int(env.model.up_axis),
                "bodies_per_env": env.bodies_per_env,
                "gravity": np.asarray(env.model.gravity, dtype=np.float64).tolist(),
            }
            asset = (
                source
                / "envs/warp_sim_envs/assets"
                / ("ant.xml" if robot == "ant" else "cartpole_single.urdf")
            )
            manifest = {
                "schema_version": 1,
                "trace_type": "evaluation_including_failures",
                "split_seed": 0,
                "metadata": {
                    "robot": robot,
                    "dt": env.frame_dt,
                    "codec": {
                        "robot": robot,
                        "base_com": list(codec.base_com),
                        "continuous_q_indices": list(codec.continuous_q_indices),
                    },
                    "world_rotation_xyzw": rotation.tolist(),
                    "asset_sha256": _hash(asset),
                    "source_physical_arrays": _physics(env.model),
                    "sampling": {"batch_size": batch},
                    "teacher_origin": origin,
                    "source_runtime_files": {
                        name: _hash(source / name) for name in runtime_files
                    },
                    "generation_environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "warp": wp.__version__,
                        "runtime_imports_embodichain": False,
                        "runtime_imports_dexsim": False,
                        "runtime_imports_newton": False,
                    },
                    "canonical_transport_modules": {
                        name: {
                            "path": str(transport / (name + ".py")),
                            "sha256": _hash(transport / (name + ".py")),
                        }
                        for name in ("codec", "reference")
                    },
                    "generator_sha256": _hash(Path(__file__)),
                    "config_sha256": _hash(args.config),
                },
                "evaluation": {
                    "seed": seed,
                    "trajectories": count,
                    "horizon": horizon,
                    "num_envs": batch,
                    "initial_states": cfg.get("initial_states", "sample"),
                    "motion_mode": mode,
                    "policy": identity,
                },
                "shards": [],
            }
            out.mkdir(parents=True)
            for start in range(0, count, batch):
                n = min(batch, count - start)
                samples, commands = [], []
                for index in range(start, start + n):
                    set_random_seed(seed + index)
                    raw = default[None].clone()
                    if cfg.get("initial_states", "sample") == "sample":
                        sampler.sample_initial_states(1, raw)
                    command = torch.zeros(horizon, env.joint_act_dim)
                    if mode == "random_effort":
                        sampler.sampler.sample(horizon, -limits, limits, command)
                    samples.append(raw[0])
                    commands.append(command)
                samples.extend([samples[-1]] * (batch - n))
                commands.extend([commands[-1]] * (batch - n))
                command = torch.stack(commands)
                env.reset(initial_states=torch.stack(samples).clone())
                raw_rows, controls = [env.states.clone()], []
                with torch.no_grad():
                    for step in range(horizon):
                        force = command[:, step]
                        if policy is not None:
                            q, qd = _canonical(env.states, codec, rotation, math_module)
                            obs, _, _ = observations.reference_task_values(robot, q, qd)
                            if robot == "cartpole":
                                force = torch.zeros_like(force)
                                force[:, 0] = (
                                    -(torch.nan_to_num(obs) * gains)
                                    .sum(-1)
                                    .clamp(-limits[0], limits[0])
                                )
                            else:
                                force = (
                                    kp * (target - torch.nan_to_num(q[:, 7:]))
                                    - kd * torch.nan_to_num(qd[:, 6:])
                                ).clamp(-limits, limits)
                            force = torch.where(
                                torch.isfinite(obs).all(-1)[:, None], force, 0
                            )
                        force = force.contiguous()
                        controls.append(force.clone())
                        raw_rows.append(
                            env.step_with_joint_act(
                                force, env_mode="ground-truth"
                            ).clone()
                        )
                raw = torch.stack(raw_rows, 1)[:n]
                q, qd = _canonical(raw, codec, rotation, math_module)
                raw_force = torch.stack(controls, 1)[:n]
                canonical_force = (
                    torch.cat((raw_force.new_zeros(n, horizon, 6), raw_force), -1)
                    if robot == "ant"
                    else raw_force
                )
                file = out / ("teacher-%06d.h5" % len(manifest["shards"]))
                rows = []
                with h5py.File(file, "w") as stream:
                    for row in range(n):
                        group = stream.create_group(str(row))
                        for key, value in {
                            "states/raw": raw,
                            "states/q": q,
                            "states/qd": qd,
                            "transitions/joint_f": canonical_force,
                            "transitions/raw_joint_f": raw_force,
                        }.items():
                            group.create_dataset(
                                key, data=value[row].numpy(), compression="lzf"
                            )
                        rows.append(
                            {
                                "id": "%d:%d" % (seed, start + row),
                                "group": str(row),
                                "length": horizon,
                                "split": "test",
                                "metadata": {"sampling_seed": seed + start + row},
                            }
                        )
                manifest["shards"].append(
                    {"file": file.name, "sha256": _hash(file), "trajectories": rows}
                )
                forbidden = {key.split(".")[0] for key in sys.modules} & {
                    "embodichain",
                    "dexsim",
                    "newton",
                }
                if forbidden:
                    raise RuntimeError(
                        "Unsupported packages imported in original environment"
                    )
                temporary = out / "manifest.pending.json"
                temporary.write_text(
                    json.dumps(manifest, indent=2, allow_nan=False) + "\n"
                )
                temporary.replace(out / "manifest.json")
                print(
                    json.dumps(
                        {
                            "event": "committed_batch",
                            "trajectories": start + n,
                            "requested": count,
                        }
                    ),
                    flush=True,
                )
        finally:
            env.close()
    print(
        json.dumps(
            {
                "status": "completed",
                "scope": "Original supported-environment evaluation teacher data only, no model evaluated",
                "trajectories": count,
                "horizon": horizon,
                "manifest_sha256": _hash(out / "manifest.json"),
                "elapsed_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
