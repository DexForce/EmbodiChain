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

"""Evaluate original checkpoints on fixed held-out data in Python3.8/Torch2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

__all__: list[str] = []


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--robot", choices=("cartpole", "ant"), required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    import warp as wp
    from torch.utils.data import DataLoader, Subset
    from algorithms.vanilla_trainer import VanillaTrainer
    from envs.neural_environment import NeuralEnvironment
    from utils.datasets import TrajectoryDataset

    if (platform.python_version(), torch.__version__.split("+")[0], wp.__version__) != (
        "3.8.20",
        "2.2.2",
        "1.5.1" if args.robot == "ant" else "1.8.0",
    ):
        raise ValueError("Unexpected original evaluation environment")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    model_path = (
        args.official_dir / "original-log/nn/best_valid_exp_trajectory_model.pt"
    )
    model = torch.load(model_path, map_location="cpu")[0].eval()
    env = NeuralEnvironment(
        env_name=args.robot.capitalize(),
        num_envs=512,
        neural_model=model,
        neural_integrator_cfg={
            "name": "TransformerNeuralIntegrator",
            "num_states_history": 10,
            "states_frame": "body" if args.robot == "ant" else "world",
            "anchor_frame_step": "every",
            "states_embedding_type": "identical",
            "prediction_type": "relative",
            "orientation_prediction_parameterization": "quaternion",
        },
        device="cpu",
        render=False,
    )
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device, trainer.batch_size = "cpu", 512
    trainer.neural_env, trainer.neural_integrator = env, env.integrator_neural
    trainer.neural_model = model

    @torch.no_grad()
    def measure(dataset: Any, golden: bool = False) -> dict[str, Any]:
        accumulated = torch.zeros(())
        itemized: dict[str, Any] = {}
        loader = DataLoader(
            dataset, batch_size=512, shuffle=False, drop_last=True, num_workers=4
        )
        for index, data in enumerate(loader):
            data = trainer.preprocess_data_batch(data)
            if golden and index == 0:
                inputs = {
                    key: data[key].clone()
                    for key in trainer.neural_model.low_dim_input_names
                }
                with torch.no_grad():
                    prediction = trainer.neural_model(
                        {key: value.clone() for key, value in inputs.items()}
                    )
                torch.save(
                    {
                        "inputs": inputs,
                        "prediction": prediction,
                        "target": data["target"].clone(),
                    },
                    args.official_dir / "test-first-batch-golden.pt",
                )
            loss, items = trainer.compute_loss(data, False)
            accumulated += loss
            for key, value in items.items():
                itemized[key] = itemized.get(key, 0) + value
        return {
            "normalized_mse": float(accumulated) / len(loader),
            "itemized": {
                key: float(value) / len(loader) for key, value in itemized.items()
            },
            "windows": len(loader) * 512,
            "valid_positions": len(loader) * 5120,
        }

    test = TrajectoryDataset(str(args.test_data), sample_sequence_length=10)
    test_result = measure(test, golden=True)
    validation = TrajectoryDataset(str(args.validation_data), sample_sequence_length=10)
    fixed_indices = (
        np.load(str(args.official_dir / "validation-0000-indices.npy"))[:8]
        .reshape(-1)
        .tolist()
    )
    fixed = Subset(validation, fixed_indices)
    curves = []
    for step in (0, 250, 500, 750, 1000):
        path = args.official_dir / (
            "initial.pt" if step == 0 else "step-%04d.pt" % step
        )
        trainer.neural_model = torch.load(path, map_location="cpu")[0].eval()
        curves.append({"step": step, **measure(fixed)})
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "selected_model_sha256": _hash(model_path),
        "test_data_sha256": _hash(args.test_data),
        "test": test_result,
        "fixed_validation_curve": curves,
        "curve_indices_sha256": hashlib.sha256(
            np.asarray(fixed_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "golden_sha256": _hash(args.official_dir / "test-first-batch-golden.pt"),
    }
    path = args.official_dir / "heldout-evaluation.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
