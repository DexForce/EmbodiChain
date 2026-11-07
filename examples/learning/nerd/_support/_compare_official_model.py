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

"""Compare the migrated model with frozen outputs from the tested official env.

No Newton or DexSim teacher is used. Invoke with the robot's isolated old Warp
on PYTHONPATH. Reference data/outputs come from official_model_reference.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-oracle", type=Path)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads((args.reference / "reference-identity.json").read_text())
    for name, digest in identity["files"].items():
        if hashlib.sha256((args.reference / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Reference artifact checksum mismatch: " + name)
    import h5py
    import numpy as np
    import torch
    import warp as wp
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd import load_model
    from examples.learning.nerd._support._reference import (
        reference_identity,
        import_legacy_datasets,
        _official_batch,
    )
    from examples.learning.nerd._support._reference_rollout import (
        _cpu_stream_compatibility,
    )
    from examples.learning.nerd._support._trainer import encode_batch
    from embodichain_models.dynamics.nerd.data import DynamicsWindowDataset

    source = reference_identity(args.source)
    if (
        source["commit"] != identity["source_commit"]
        or wp.__version__ != identity["warp"]
    ):
        raise ValueError("Original source or Warp identity mismatch")
    common, manifest = load_model(args.model)
    if manifest["legacy_checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Port must retain the same original checkpoint identity")
    torch.set_num_threads(1)
    codec = NeRDStateCodec(**manifest["codec"])
    robot = identity["robot"]
    canonical = args.output.parent / (args.output.stem + "-data")
    rotation = (
        [0.5, 0.5, 0.5, 0.5]
        if robot == "cartpole"
        else [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    )
    import_legacy_datasets(
        {"test": args.reference / "test.raw.h5"},
        canonical,
        {
            "robot": robot,
            "dt": 1 / 60,
            "codec": manifest["codec"],
            "reference_identity": identity,
        },
        world_rotation_xyzw=rotation,
        gravity_direction=tuple(identity["gravity_transport"]["direction"]),
    )
    results = {}
    oracle_identity = None
    oracle_windows = None
    if args.canonical_oracle is not None:
        oracle_identity = json.loads(
            (args.canonical_oracle / "reference-identity.json").read_text()
        )
        oracle_file = args.canonical_oracle / "official-windows.npz"
        if (
            oracle_identity["checkpoint_sha256"] != identity["checkpoint_sha256"]
            or hashlib.sha256(oracle_file.read_bytes()).hexdigest()
            != oracle_identity["predictions_sha256"]
        ):
            raise ValueError("Common-frame oracle identity mismatch")
        with np.load(oracle_file, allow_pickle=False) as values:
            oracle_windows = {
                key: torch.from_numpy(values[key].copy()) for key in values.files
            }
    with np.load(args.reference / "official-windows.npz") as fixtures:
        for length in [1, 9, 10]:
            prefix = str(length) + "__"
            expected = torch.from_numpy(fixtures[prefix + "prediction"])
            direct_features = {
                ("joint_f" if name == "joint_acts" else name): torch.from_numpy(
                    fixtures[prefix + "features__" + name]
                )
                for name in identity["model_inputs"]
            }
            with torch.no_grad():
                direct = common(direct_features)
            ds = DynamicsWindowDataset(canonical, "test", length, verify=True)
            offsets = {}
            offset = 0
            for shard in ds.manifest["shards"]:
                for row in shard["trajectories"]:
                    offsets[row["metadata"]["legacy_index"]] = offset
                    offset += row["length"] - length + 1
            ids = [
                offsets[int(episode)] + int(start)
                for episode, start in fixtures[prefix + "indices"]
            ]
            batch = ds.get_batch(ids)
            features, target, mask = encode_batch(batch, codec, "cpu")
            with torch.no_grad():
                actual = common(features)
            expected_target = torch.from_numpy(fixtures[prefix + "target"])
            source_frame_prediction = expected
            if oracle_windows is not None:
                for key, value in _official_batch(batch, codec).items():
                    actual_hash = hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    if (
                        actual_hash
                        != oracle_identity["window_field_sha256"][
                            str(length) + "__raw__" + key
                        ]
                    ):
                        raise ValueError(
                            "Reference and candidate did not receive identical floating-point inputs"
                        )
                expected = oracle_windows[prefix + "prediction"]
                expected_target = oracle_windows[prefix + "target"]
                direct_features = {
                    ("joint_f" if name == "joint_acts" else name): oracle_windows[
                        prefix + "features__" + name
                    ]
                    for name in identity["model_inputs"]
                }
            variance = common.output_rms.var
            official_loss = (
                (expected - expected_target).square() / (variance + 1e-5)
            ).mean()
            port_loss = ((actual - target).square() / (variance + 1e-5)).mean()
            results[str(length)] = {
                "windows": len(ids),
                "direct_model_max_abs": float(
                    (direct - source_frame_prediction).abs().max()
                ),
                "reference_frame_rotation_effect": float(
                    (expected - source_frame_prediction).abs().max()
                ),
                "codec_and_model_max_abs": float((actual - expected).abs().max()),
                "target_max_abs": float((target - expected_target).abs().max()),
                "feature_max_abs": {
                    key: float((features[key] - direct_features[key]).abs().max())
                    for key in features
                },
                "official_normalized_mse": float(official_loss),
                "migrated_normalized_mse": float(port_loss),
                "relative_mse_change": float(port_loss / official_loss - 1),
                "forward_tolerance_passed": bool(
                    torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                ),
            }
            ds.close()
    sys.path.insert(0, source["root"])
    from envs.neural_environment import NeuralEnvironment

    class PortableAdapter(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.common = model

        def evaluate(self, values):
            return self.common.evaluate(
                {
                    ("joint_f" if key == "joint_acts" else key): v
                    for key, v in values.items()
                }
            )

        def init_rnn(self, batch_size):
            del batch_size

    with np.load(args.reference / "official-rollout.npz") as trace:
        initial = trace["states"][0].copy()
        controls = trace["joint_acts"].copy()
        expected_rollout = trace["states"].copy()
    cfg = {
        "name": "TransformerNeuralIntegrator",
        "num_states_history": 10,
        "states_frame": "body" if robot == "ant" else "world",
        "anchor_frame_step": "every",
        "states_embedding_type": "identical",
        "prediction_type": "relative",
        "orientation_prediction_parameterization": "quaternion",
    }
    env = NeuralEnvironment(
        env_name=robot.capitalize(),
        num_envs=identity["count"],
        neural_model=PortableAdapter(common),
        neural_integrator_cfg=cfg,
        default_env_mode="neural",
        warp_env_cfg={"seed": 500, "random_reset": False},
        device="cpu",
        render=False,
    )
    outputs = []
    with _cpu_stream_compatibility(wp, "cpu"):
        env.reset(initial_states=torch.from_numpy(initial))
        outputs.append(env.states.numpy().copy())
        for force in controls:
            state = env.step_with_joint_act(torch.from_numpy(force), env_mode="neural")
            outputs.append(state.numpy().copy())
    outputs = np.stack(outputs)
    np.savez(args.output.with_suffix(".npz"), states=outputs)
    rollout = {}
    for h in [1, 10, 100, 1000]:
        if h >= len(outputs):
            continue
        delta = np.abs(outputs[1 : h + 1] - expected_rollout[1 : h + 1])
        rollout[str(h)] = {
            "max_abs_state_difference": float(delta.max()),
            "mean_abs_state_difference": float(delta.mean()),
            "all_finite": bool(np.isfinite(outputs[: h + 1]).all()),
        }
    report = {
        "scope": "Original pretrained model versus migrated copy of the same weights; official supported environment generates comparison data. No Newton teacher.",
        "reference": identity,
        "common_frame_oracle": oracle_identity,
        "migrated_environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "warp": wp.__version__,
        },
        "model_sha256": hashlib.sha256(
            (args.model / "weights.pt").read_bytes()
        ).hexdigest(),
        "same_original_checkpoint": True,
        "histories": results,
        "rollouts": rollout,
        "current_simulator_imports": [
            key for key in sys.modules if key.split(".")[0] in {"dexsim", "newton"}
        ],
        "qualification": "bounded_real_data_model_port_comparison_not_full_training_quality",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
