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

"""Exercise standalone PickUp planning in fresh processes with the same cache."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.subprocess_sim
def test_standalone_pickup_prepares_then_reuses_physical_labels(tmp_path: Path) -> None:
    cache_dir = tmp_path / "labels"
    command = [
        sys.executable,
        __file__,
        str(cache_dir),
        "--headless",
        "--renderer",
        "no-render",
        "--robot",
        "ur5",
        "--planner",
        "trapezoidal",
        "--no_vis_eef_axis",
        "--n_sample",
        "1000",
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for phase, workers in (("cold", 1), ("warm", 0)):
        log = tmp_path / f"{phase}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=300,
                env=env,
            )
        output = log.read_text()
        assert result.returncode == 0, output[-16000:]
        assert output.count("Started physical labeling worker") == workers
        (labels,) = cache_dir.glob("*.pt")
        assert f"validated grasps from {labels.name}" in output
        assert "(pick_up:grasp): success=True" in output


if __name__ == "__main__":
    import runpy
    from functools import partial
    from unittest.mock import patch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from embodichain.lab.sim.affordance.grasp import cache, config

    def observe_worker(event: str, args: tuple) -> None:
        if event == "subprocess.Popen" and args[1][1:3] == ["-m", cache.__name__]:
            print("Started physical labeling worker", flush=True)

    sys.addaudithook(observe_worker)
    cache_dir = Path(sys.argv.pop(1))
    # Keep the physical run small and use identical job settings on both invocations.
    sampling = partial(
        config.GraspSamplingCfg, stable_poses=2, directions=16, max_drops=16
    )
    with (
        patch.object(cache, "_LABEL_CACHE_DIR", cache_dir),
        patch.object(config, "GraspSamplingCfg", sampling),
        patch.object(config, "DEFAULT_NUM_ENVS", 256),
    ):
        runpy.run_module("examples.sim.atomic_skill.pickup", run_name="__main__")
