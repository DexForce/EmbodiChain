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

"""Measure official-width resident NeRD optimizer updates and budget lower bounds.

Run: python -m embodichain run-example learning/nerd/_support/_benchmark_dynamics_training --source /path/to/main
     --trust-external-code --output /path/to/result.json

Synthetic, seeded device-resident features measure computation only. Data loading,
state/contact encoding, validation, exports and simulation are excluded; projected
full training times are optimistic lower bounds, never accuracy qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

__all__ = ["benchmark_training"]


def benchmark_training(
    source: str | Path,
    *,
    device: str = "cuda:0",
    batch_size: int = 512,
    history: int = 10,
    warmup: int = 20,
    repeats: int = 100,
    trust_external_code: bool = False,
) -> dict[str, Any]:
    """Time synchronized completed FP32 updates using the real two model classes."""
    if not trust_external_code:
        raise ValueError("Reference model code requires explicit trust")
    if min(batch_size, history, warmup, repeats) < 1:
        raise ValueError("Workload sizes must be positive")
    import torch
    from examples.learning.nerd._support._reference import reference_identity
    from examples.learning.nerd._support._trainer import normalized_mse
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg

    identity = reference_identity(source)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable"}
    sys.path.insert(0, str(Path(source).resolve()))
    from models.models import ModelMixedInput
    from utils.running_mean_std import RunningMeanStd

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sync = (
        (lambda: torch.cuda.synchronize(device))
        if device.startswith("cuda")
        else (lambda: None)
    )
    records = []
    for robot in ("cartpole", "ant"):
        torch.manual_seed(42)
        dims = (
            {"states_embedding": 4, "joint_f": 2}
            if robot == "cartpole"
            else {
                "states_embedding": 29,
                "contact_normals": 75,
                "contact_points_1": 75,
                "contact_depths": 25,
                "joint_f": 8,
                "gravity_dir": 3,
            }
        )
        width = 192 if robot == "cartpole" else 384
        output = 4 if robot == "cartpole" else 29
        inputs = {
            key: torch.randn(batch_size, history, size, device=device)
            for key, size in dims.items()
        }
        target = torch.randn(batch_size, history, output, device=device)
        migrated = NeRDModel(
            NeRDModelCfg(
                input_dims=dims,
                output_dim=output,
                n_embd=width,
                n_layer=6,
                n_head=12,
                history_length=history,
                block_size=max(32, history),
            )
        ).to(device)
        rename = lambda key: "joint_acts" if key == "joint_f" else key
        official = ModelMixedInput(
            input_sample={rename(key): value for key, value in inputs.items()},
            output_dim=output,
            input_cfg={"low_dim": [rename(key) for key in dims]},
            network_cfg={
                "encoder": {
                    "low_dim": {
                        "activation": "relu",
                        "layer_sizes": [],
                        "layernorm": False,
                    }
                },
                "model": {
                    "mlp": {
                        "activation": "relu",
                        "layer_sizes": [64],
                        "layernorm": False,
                    }
                },
                "normalize_input": True,
                "normalize_output": True,
                "transformer": {
                    "n_embd": width,
                    "n_layer": 6,
                    "n_head": 12,
                    "block_size": max(32, history),
                    "bias": False,
                    "dropout": 0.0,
                },
            },
            device=device,
        )
        official.load_state_dict(
            {
                key: value
                for key, value in migrated.state_dict().items()
                if not key.startswith(("input_rms.", "output_rms."))
            }
        )
        stats = {}
        for key, size in dims.items():
            stats[rename(key)] = RunningMeanStd(shape=(size,), device=device)
        official.set_input_rms(stats)
        official.set_output_rms(RunningMeanStd(shape=(output,), device=device))
        with torch.no_grad():
            torch.testing.assert_close(
                official({rename(k): v for k, v in inputs.items()}),
                migrated(inputs),
                atol=1e-5,
                rtol=1e-4,
            )
        for name, model in (("official", official), ("embodichain", migrated)):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)

            def update() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    {rename(k): v for k, v in inputs.items()}
                    if name == "official"
                    else dict(inputs)
                )
                if name == "official":
                    weight = torch.rsqrt(model.output_rms.var + 1e-5)
                    loss = torch.nn.functional.mse_loss(
                        prediction * weight, target * weight
                    )
                else:
                    loss = normalized_mse(prediction, target, model.output_rms.var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                return loss.detach()

            start = time.perf_counter()
            last = update()
            sync()
            first = time.perf_counter() - start
            for _ in range(warmup):
                last = update()
            sync()
            samples = []
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                last = update()
                sync()
                samples.append(time.perf_counter() - start)
            if not torch.isfinite(last):
                raise FloatingPointError("Non-finite timing workload")
            median = statistics.median(samples)
            records.append(
                {
                    "robot": robot,
                    "implementation": name,
                    "median_seconds_per_update": median,
                    "mean_seconds_per_update": statistics.mean(samples),
                    "min_seconds_per_update": min(samples),
                    "max_seconds_per_update": max(samples),
                    "first_update_seconds": first,
                    "samples_seconds": samples,
                    "peak_cuda_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.startswith("cuda")
                        else None
                    ),
                    "last_loss": float(last),
                    "one_4995000_update_run_hours": median * 4995000 / 3600,
                }
            )
            del optimizer
        delta = max(
            (dict(official.named_parameters())[key] - value).abs().max().item()
            for key, value in migrated.named_parameters()
        )
        for record in records[-2:]:
            record["matched_final_parameter_max_abs"] = delta
        del official, migrated, inputs, target, last
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    # 2 dataset rows x 5 seeds for each implementation and robot.
    total = sum(
        record["median_seconds_per_update"] * 4995000 * 10 for record in records
    )
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {
        "status": "completed",
        "scope": "device_resident_forward_normalized_mse_backward_clip_adam",
        "excluded": [
            "data I/O",
            "feature encoding",
            "validation",
            "checkpoint/export",
            "teacher generation",
            "rollout evaluation",
        ],
        "precision": "float32, TF32 disabled",
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(device) if device.startswith("cuda") else None
        ),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "python": platform.python_version(),
        "cpu_threads": 1,
        "batch_size": batch_size,
        "history": history,
        "warmup_updates": warmup,
        "measured_updates": repeats,
        "source": identity,
        "runs": records,
        "full_40_run_matrix_optimistic_hours": total / 3600,
        "full_40_run_matrix_optimistic_days": total / 86400,
        "accuracy_qualification": "not_measured_synthetic_performance_inputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_training(
        args.source,
        device=args.device,
        batch_size=args.batch_size,
        history=args.history,
        warmup=args.warmup,
        repeats=args.repeats,
        trust_external_code=args.trust_external_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "runs"}, indent=2
        )
    )
    for record in result.get("runs", []):
        print({key: value for key, value in record.items() if key != "samples_seconds"})


if __name__ == "__main__":
    main()
