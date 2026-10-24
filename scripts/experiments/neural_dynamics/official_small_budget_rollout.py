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

"""Replay fixed supported-environment teacher prefixes through original NeRD.

Runs both trained weight sets in the same original Python3.8/Torch2.2 runtime.
No ground-truth physics is regenerated. The original integrator evolves its own
state, contacts and Transformer history. Fixed-policy actions use its own state.
Pure tensor transport and metrics load independently of the EmbodiChain package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any

from official_evaluation_data import _canonical, _hash, _transport

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.teacher / "manifest.json").read_text())
    metadata, evaluation = manifest["metadata"], manifest["evaluation"]
    robot = metadata["robot"]
    sys.path.insert(0, str(args.source))
    import h5py
    import numpy as np
    import torch
    import warp as wp
    from envs.neural_environment import NeuralEnvironment

    expected_warp = "1.5.1" if robot == "ant" else "1.8.0"
    if (
        platform.python_version(),
        torch.__version__.split("+")[0],
        np.__version__,
        wp.__version__,
    ) != ("3.8.20", "2.2.2", "1.24.4", expected_warp):
        raise ValueError("Replay requires the original supported environment")
    generation = metadata["generation_environment"]
    if (
        generation["python"] != "3.8.20"
        or generation["torch"].split("+")[0] != "2.2.2"
        or generation["warp"] != expected_warp
    ):
        raise ValueError("Teacher was not generated in its supported environment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    math_module, task_module = _transport(args.transport_root)
    metric_spec = importlib.util.spec_from_file_location(
        "_isolated_nerd_transport._evaluation", args.transport_root / "_evaluation.py"
    )
    metric_module = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(metric_module)
    codec = math_module.NeRDStateCodec(**metadata["codec"])
    rotation = torch.tensor(metadata["world_rotation_xyzw"], dtype=torch.float32)
    horizon, count = 100, args.count
    if not 0 < count <= evaluation["trajectories"] or evaluation["horizon"] < horizon:
        raise ValueError("Invalid fixed prefix selection")
    rows = []
    identities = []
    verified_shards = {}
    for shard in manifest["shards"]:
        if len(rows) >= count:
            break
        path = args.teacher / shard["file"]
        actual_hash = _hash(path)
        if actual_hash != shard["sha256"]:
            raise ValueError("Teacher shard checksum mismatch")
        verified_shards[shard["file"]] = actual_hash
        with h5py.File(str(path), "r") as stream:
            for record in shard["trajectories"]:
                if len(rows) >= count:
                    break
                group = stream[record["group"]]
                row = {
                    "raw": group["states/raw"][: horizon + 1],
                    "q": group["states/q"][: horizon + 1],
                    "qd": group["states/qd"][: horizon + 1],
                    "raw_joint_f": group["transitions/raw_joint_f"][:horizon],
                    "joint_f": group["transitions/joint_f"][:horizon],
                }
                rows.append(row)
                identities.append(
                    {
                        **record,
                        "prefix_sha256": metric_module._teacher_row_sha256(
                            record, [row["q"], row["qd"], row["joint_f"]]
                        ),
                    }
                )
    teacher = {
        key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]
    }
    tq, tqd = _canonical(teacher["raw"], codec, rotation, math_module)
    for actual, key in ((tq, "q"), (tqd, "qd")):
        torch.testing.assert_close(actual, teacher[key], atol=0, rtol=0)
    model = torch.load(args.checkpoint, map_location="cpu")[0].eval()
    env = NeuralEnvironment(
        env_name=robot.capitalize(),
        num_envs=count,
        neural_model=model,
        neural_integrator_cfg={
            "name": "TransformerNeuralIntegrator",
            "num_states_history": 10,
            "states_frame": "body" if robot == "ant" else "world",
            "anchor_frame_step": "every",
            "states_embedding_type": "identical",
            "prediction_type": "relative",
            "orientation_prediction_parameterization": "quaternion",
        },
        warp_env_cfg={
            "seed": evaluation["seed"],
            "random_reset": False,
            "env_offset": (0.0, 0.0, 0.0),
            "use_graph_capture": False,
        },
        default_env_mode="neural",
        device="cpu",
        render=False,
    )
    original_stream = wp.stream_to_torch
    wp.stream_to_torch = lambda device: (
        None if wp.get_device(device).is_cpu else original_stream(device)
    )
    env.reset(initial_states=teacher["raw"][:, 0].contiguous())
    policy = evaluation["policy"]

    def controls(q: Any, qd: Any) -> Any:
        observation, _, _ = task_module.reference_task_values(robot, q, qd)
        finite = torch.isfinite(observation).all(-1)
        limits = torch.tensor(policy["effort_limits"])
        if robot == "cartpole":
            effort = torch.zeros_like(qd)
            effort[:, 0] = (
                -(torch.nan_to_num(observation) * torch.tensor(policy["gains"]))
                .sum(-1)
                .clamp(-limits[0], limits[0])
            )
        else:
            effort = (
                policy["kp"]
                * (torch.tensor(policy["target"]) - torch.nan_to_num(q[:, 7:]))
                - policy["kd"] * torch.nan_to_num(qd[:, 6:])
            ).clamp(-limits, limits)
        return torch.where(finite[:, None], effort, 0)

    predicted, efforts = [env.states.clone()], []
    with torch.no_grad():
        for step in range(horizon):
            if evaluation["motion_mode"] == "fixed_policy":
                torch.testing.assert_close(
                    controls(tq[:, step], tqd[:, step]),
                    teacher["raw_joint_f"][:, step],
                    atol=0,
                    rtol=0,
                )
                q, qd = _canonical(env.states, codec, rotation, math_module)
                effort = controls(q, qd)
            else:
                effort = teacher["raw_joint_f"][:, step]
            effort = effort.contiguous()
            efforts.append(effort.clone())
            predicted.append(env.step_with_joint_act(effort, env_mode="neural").clone())
    raw = torch.stack(predicted, 1)
    q, qd = _canonical(raw, codec, rotation, math_module)
    errors = metric_module._error_values(
        robot, q[:, 1:], qd[:, 1:], tq[:, 1:], tqd[:, 1:]
    )
    finite = torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1)
    teacher_finite = torch.isfinite(tq).all(-1) & torch.isfinite(tqd).all(-1)
    valid = (finite & teacher_finite).cummin(1).values[:, 1:]
    _, _, teacher_failed = task_module.reference_task_values(robot, tq, tqd)
    _, _, neural_failed = task_module.reference_task_values(robot, q, qd)
    teacher_failed, neural_failed = (
        teacher_failed.cummax(1).values[:, 1:],
        neural_failed.cummax(1).values[:, 1:],
    )
    names = ("q_mae", "qd_mae", "base_position_error", "base_orientation_error_rad")
    metrics = {}
    for h in (1, 10, 100):
        mask = valid[:, h - 1]
        endpoint = errors[:, h - 1][mask]
        prefix = errors[:, :h][mask]
        metrics[str(h)] = {
            "endpoint": {
                name: metric_module._safe_number(endpoint[:, i].mean())
                for i, name in enumerate(names)
            },
            "prefix_mean": {
                name: metric_module._safe_number(prefix[..., i].mean())
                for i, name in enumerate(names)
            },
            "valid_trajectories": int(mask.sum()),
            "nonfinite_trajectories": int((~mask).sum()),
            "teacher_nonfinite_trajectories": int(
                (~teacher_finite.cummin(1).values[:, h]).sum()
            ),
            "teacher_failed_trajectories": int(teacher_failed[:, h - 1].sum()),
            "neural_failed_trajectories": int(neural_failed[:, h - 1].sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = args.output.with_suffix(".npz")
    np.savez_compressed(
        str(evidence),
        q=q.numpy(),
        qd=qd.numpy(),
        teacher_q=tq.numpy(),
        teacher_qd=tqd.numpy(),
        raw=raw.numpy(),
        efforts=torch.stack(efforts, 1).numpy(),
        teacher_efforts=teacher["raw_joint_f"].numpy(),
        errors=errors.numpy(),
        valid=valid.numpy(),
    )
    result = {
        "robot": robot,
        "motion_mode": evaluation["motion_mode"],
        "trajectories": count,
        "horizons": [1, 10, 100],
        "seed": evaluation["seed"],
        "metrics": metrics,
        "checkpoint_sha256": _hash(args.checkpoint),
        "teacher_manifest_sha256": _hash(args.teacher / "manifest.json"),
        "teacher_shards": verified_shards,
        "teacher_prefixes": identities,
        "evidence_sha256": _hash(evidence),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "runtime_imports_embodichain": "embodichain" in sys.modules,
            "runtime_imports_dexsim": "dexsim" in sys.modules,
        },
        "teacher_generation_environment": generation,
        "evaluation_scope": "original neural integration and common original supported runtime; training-weight comparison; not native Newton deployment",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
