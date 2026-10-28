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

"""Measure candidate-batched LeWM planning on device-resident inputs.

Run: python scripts/benchmark/learning/benchmark_lewm_cem.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from embodichain_models.dynamics.lewm import LeWM, LeWMCfg, LeWMCEMCfg, LeWMCEMPlanner


def measure(
    device: str, chunk: int, samples: int, iterations: int, repeats: int
) -> dict:
    torch.manual_seed(4)
    model = LeWM(LeWMCfg(action_dim=6)).to(device).eval()
    # The reference initializes AdaLN gates to zero, making a fresh model
    # action-independent. Use nonzero gates to measure a meaningful search.
    for layer in model.predictor.transformer.layers:
        torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.02)
    history = torch.randn(1, 3, 192, device=device)
    past = torch.randn(1, 2, 6, device=device)
    goal = torch.randn(1, 192, device=device)
    config = LeWMCEMCfg(
        num_samples=samples,
        iterations=iterations,
        topk=min(30, samples),
        candidate_batch_size=chunk,
    )
    planner = LeWMCEMPlanner(
        model,
        -torch.ones(6, device=device),
        torch.ones(6, device=device),
        config,
        seed=11,
    )

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    planner.plan(history, past, goal)
    sync()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        action, cost = planner.plan(history, past, goal)
        sync()
        durations.append(time.perf_counter() - start)
    return {
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(device)
            if torch.device(device).type == "cuda"
            else None
        ),
        "torch": torch.__version__,
        "samples": samples,
        "iterations": iterations,
        "horizon": config.horizon,
        "chunk": chunk,
        "repeats": repeats,
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "peak_bytes": (
            torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == "cuda"
            else None
        ),
        "final_cost": float(cost[0]),
        "action": action[0].cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            [
                measure(args.device, chunk, args.samples, args.iterations, args.repeats)
                for chunk in (min(128, args.samples), args.samples)
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
