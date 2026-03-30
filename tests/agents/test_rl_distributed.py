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

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def _create_minimal_distributed_config():
    """Create a minimal train config for distributed testing."""
    config_path = Path("configs/agents/rl/basic/cart_pole/train_config_grpo.json")
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg["trainer"]["iterations"] = 2
    cfg["trainer"]["buffer_size"] = 64
    cfg["trainer"]["eval_freq"] = 1000000
    cfg["trainer"]["save_freq"] = 1000000
    cfg["trainer"]["use_wandb"] = False
    cfg["trainer"]["enable_eval"] = False
    cfg["trainer"]["num_envs"] = 4
    cfg["algorithm"]["cfg"]["batch_size"] = 64

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Distributed test with NCCL requires CUDA",
)
@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is not available",
)
def test_distributed_training_via_torchrun():
    """Run distributed training via torchrun (subprocess) to exercise the distributed path.

    Uses 2 processes when 2+ GPUs are available to validate multi-rank behavior
    (all_reduce, all_gather). Falls back to 1 process otherwise.
    """
    config_path = _create_minimal_distributed_config()
    nproc = 2 if torch.cuda.device_count() >= 2 else 1
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "--standalone",
                "-m",
                "embodichain.agents.rl.train",
                "--config",
                config_path,
                "--distributed",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).resolve().parents[2],
        )
        assert (
            result.returncode == 0
        ), f"Distributed training failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
