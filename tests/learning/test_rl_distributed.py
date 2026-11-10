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

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest
import torch


def _run_training_command(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Terminate the owned launcher and its rank processes on timeout."""
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Elastic starts each rank in a separate session. The launcher's
            # process group alone cannot terminate those GPU-owning workers.
            try:
                workers = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                workers = []
            for worker in reversed(workers):
                try:
                    worker.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.mark.skipif(
    os.name != "posix", reason="NCCL launcher uses POSIX process groups"
)
@pytest.mark.parametrize(
    "worker_new_session", [False, True], ids=["same-session", "rank-session"]
)
def test_training_timeout_terminates_worker_process(
    tmp_path: Path, worker_new_session: bool
) -> None:
    """A timed-out launcher cannot leave a training rank using GPU resources."""
    pid_file = tmp_path / "worker.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        f"worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], "
        f"start_new_session={worker_new_session!r}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "pathlib.Path(sys.argv[1]).write_text(str(worker.pid)); "
        "worker.wait()"
    )
    worker = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_training_command(
                [sys.executable, "-c", script, str(pid_file)],
                cwd=tmp_path,
                env=os.environ.copy(),
                timeout=2,
            )
        assert pid_file.exists(), "Launcher did not start its worker"
        try:
            worker = psutil.Process(int(pid_file.read_text()))
            deadline = time.monotonic() + 2
            while worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE:
                assert time.monotonic() < deadline, "Worker survived launcher timeout"
                time.sleep(0.01)
        except psutil.NoSuchProcess:
            pass
    finally:
        if worker is not None:
            try:
                worker.kill()
            except psutil.NoSuchProcess:
                pass


def _create_minimal_distributed_config():
    """Create a minimal train config for distributed testing."""
    config_path = Path(
        "extensions/embodichain_tasks/configs/tasks/classic_control/cart_pole/agents/grpo.json"
    )
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
@pytest.mark.gpu
@pytest.mark.slow
def test_distributed_training_via_torchrun():
    """Run distributed training via torchrun (subprocess) to exercise the distributed path.

    Uses 2 processes when 2+ GPUs are available to validate multi-rank behavior
    (all_reduce, all_gather). Falls back to 1 process otherwise.
    """
    config_path = _create_minimal_distributed_config()
    nproc = 2 if torch.cuda.device_count() >= 2 else 1
    child_env = os.environ.copy()
    # conftest disables process exit so in-process simulation tests can defer
    # DexSim cleanup to the top-level pytest fixture. This child is a standalone
    # training process and must use the normal runtime shutdown path instead.
    child_env["EMBODICHAIN_SIM_EXIT_PROCESS"] = "1"
    try:
        result = _run_training_command(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "--standalone",
                "-m",
                "embodichain.learning.rl.train",
                "--config",
                config_path,
                "--distributed",
            ],
            # Distributed sim startup (per-rank Warp/CUDA + DexSim + NCCL
            # init) dominates wall time on multi-GPU hosts and can exceed 120s
            # even though the training itself is short.
            timeout=300,
            cwd=Path(__file__).resolve().parents[2],
            env=child_env,
        )
        assert (
            result.returncode == 0
        ), f"Distributed training failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
