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

"""Demo: OnlineDataset with item mode and batch mode.

This script demonstrates how to use OnlineDataset backed by an OnlineDataEngine
streaming live simulation data.  Two DataLoader patterns are shown:

- **Item mode**: ``DataLoader(dataset, batch_size=4)`` — DataLoader handles
  collation; each worker independently draws single chunks from the engine.

- **Batch mode**: ``DataLoader(dataset, batch_size=None)`` — the dataset yields
  a pre-batched TensorDict; DataLoader passes it through unchanged for maximum
  engine efficiency.

Usage::

    python examples/agents/datasets/online_dataset_demo.py 
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from embodichain.agents.datasets.sampler import UniformChunkSampler, GMMChunkSampler
from embodichain.agents.datasets import OnlineDataset
from embodichain.agents.engine.data import OnlineDataEngine, OnlineDataEngineCfg
from embodichain.utils.logger import log_info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OnlineDataset demo")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Simulation device, e.g. 'cpu' or 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gym/special/simple_task_ur10.json",
        help="Path to the gym JSON config (default: configs/gym/special/simple_task_ur10.json).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of timesteps per trajectory chunk (default: 32).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to draw in each mode demo (default: 5).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _build_engine(args: argparse.Namespace) -> OnlineDataEngine:
    """Construct and start an OnlineDataEngine from the given CLI args."""
    config_path = Path(
        "/root/sources/EmbodiChain/configs/gym/special/simple_task_ur10.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Gym config not found: {config_path}. "
            "Provide a valid path via --config."
        )

    from embodichain.utils.utility import load_json

    gym_config = load_json(config_path)

    gym_config["headless"] = True
    gym_config["enable_rt"] = True
    gym_config["gpu_id"] = 0
    gym_config["device"] = args.device
    cfg = OnlineDataEngineCfg(buffer_size=4, state_dim=6, gym_config=gym_config)
    engine = OnlineDataEngine(cfg)
    engine.start()
    from IPython import embed

    embed()  # Debug breakpoint: inspect engine state after startup
    return engine


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def _demo_item_mode(
    engine: OnlineDataEngine, chunk_size: int, num_batches: int
) -> None:
    """Item mode: DataLoader collates individual chunks into batches."""
    batch_size = 4
    log_info(
        f"\n[Demo] ── Item mode ──────────────────────────────────────────\n"
        f"  DataLoader(dataset, batch_size={batch_size})\n"
        f"  Each worker draws single chunks [chunk_size={chunk_size}];\n"
        f"  DataLoader stacks them into [{batch_size}, {chunk_size}] batches.",
        color="cyan",
    )

    dataset = OnlineDataset(engine, chunk_size=chunk_size)
    loader = DataLoader(dataset, batch_size=batch_size)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        # Print the batch size of a representative tensor.
        first_key = next(iter(batch.keys()))
        shape = tuple(batch[first_key].shape)
        log_info(
            f"  batch {i + 1}/{num_batches}  key='{first_key}'  shape={shape}",
            color="white",
        )

    log_info("[Demo] Item mode complete.", color="green")


def _demo_batch_mode(
    engine: OnlineDataEngine, chunk_size: int, num_batches: int
) -> None:
    """Batch mode: dataset yields pre-batched TensorDicts; DataLoader passes them through."""
    batch_size = 4
    log_info(
        f"\n[Demo] ── Batch mode ────────────────────────────────────────\n"
        f"  DataLoader(dataset, batch_size=None)\n"
        f"  Dataset draws [{batch_size}, {chunk_size}] TensorDicts directly\n"
        f"  from the engine; DataLoader passes them through unchanged.",
        color="cyan",
    )

    dataset = OnlineDataset(engine, chunk_size=chunk_size, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=None)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        first_key = next(iter(batch.keys()))
        shape = tuple(batch[first_key].shape)
        log_info(
            f"  batch {i + 1}/{num_batches}  key='{first_key}'  shape={shape}",
            color="white",
        )

    log_info("[Demo] Batch mode complete.", color="green")


def _demo_uniform_dynamic(engine: OnlineDataEngine, num_batches: int) -> None:
    """Dynamic chunk size via UniformChunkSampler: chunk dim varies each step."""
    low, high = 16, 64
    log_info(
        f"\n[Demo] ── Dynamic chunk (Uniform) ───────────────────────────\n"
        f"  UniformChunkSampler(low={low}, high={high})\n"
        f"  Chunk size is resampled each iteration step.",
        color="cyan",
    )

    sampler = UniformChunkSampler(low=low, high=high)
    dataset = OnlineDataset(engine, chunk_size=sampler)
    loader = DataLoader(dataset, batch_size=4)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        first_key = next(iter(batch.keys()))
        shape = tuple(batch[first_key].shape)
        log_info(
            f"  batch {i + 1}/{num_batches}  key='{first_key}'  shape={shape}",
            color="white",
        )

    log_info("[Demo] Dynamic uniform chunk mode complete.", color="green")


def _demo_gmm_dynamic(engine: OnlineDataEngine, num_batches: int) -> None:
    """Dynamic chunk size via GMMChunkSampler: bimodal distribution."""
    means = [16.0, 64.0]
    stds = [4.0, 8.0]
    weights = [0.6, 0.4]
    log_info(
        f"\n[Demo] ── Dynamic chunk (GMM) ───────────────────────────────\n"
        f"  GMMChunkSampler(means={means}, stds={stds}, weights={weights}, low=8, high=96)\n"
        f"  Chunk size drawn from a two-component Gaussian mixture.",
        color="cyan",
    )

    sampler = GMMChunkSampler(means=means, stds=stds, weights=weights, low=8, high=96)
    dataset = OnlineDataset(engine, chunk_size=sampler, batch_size=4)
    loader = DataLoader(dataset, batch_size=None)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        first_key = next(iter(batch.keys()))
        shape = tuple(batch[first_key].shape)
        log_info(
            f"  batch {i + 1}/{num_batches}  key='{first_key}'  shape={shape}",
            color="white",
        )

    log_info("[Demo] Dynamic GMM chunk mode complete.", color="green")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    engine = _build_engine(args)

    try:
        _demo_item_mode(
            engine, chunk_size=args.chunk_size, num_batches=args.num_batches
        )
        _demo_batch_mode(
            engine, chunk_size=args.chunk_size, num_batches=args.num_batches
        )
        _demo_uniform_dynamic(engine, num_batches=args.num_batches)
        _demo_gmm_dynamic(engine, num_batches=args.num_batches)
    finally:
        engine.stop()
        log_info("[Demo] Engine stopped.", color="green")


if __name__ == "__main__":
    main()
