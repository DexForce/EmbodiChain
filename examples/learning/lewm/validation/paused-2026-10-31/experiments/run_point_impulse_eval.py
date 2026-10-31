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

"""Private 25+25 official source-action replay; no production registry edits."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from cube_point_impulse_trial import PointImpulseCube
import embodichain_tasks.manipulation.ogbench_cube as task
from examples.learning.lewm import _official_eval as evaluator

MODE = sys.argv[1]
if MODE not in ("baseline", "point"):
    raise ValueError(MODE)
if MODE == "point":
    # Executed in each fresh spawned child, before its deferred task import.
    task.OGBenchCubeEnv = PointImpulseCube

original_identity = evaluator._native_runtime_identity


def identity(reference):
    result = original_identity(reference)
    files = [Path(__file__)]
    if MODE == "point":
        files += [
            Path(__file__).with_name(n)
            for n in ("cube_point_impulse_trial.py", "cube_point_kinematics.py")
        ]
    result["private_trial"] = MODE
    result["private_source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    return result


evaluator._native_runtime_identity = identity

if __name__ == "__main__":
    root = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
    output = root / "evaluation" / f"dexsim-point-impulse-{MODE}-source-replay"
    if output.exists():
        raise FileExistsError(output)
    report = evaluator.evaluate_dexsim(
        reference=root / "alignment-point-response-pgs",
        queries=root / "alignment/official-queries.h5",
        weights=root / "weights/weights.pt",
        output=output,
        device="cuda:7",
        replay=root / "evaluation/mujoco/trace.npz",
        worker_devices=["cuda:6", "cuda:5"],
    )
    print(
        {
            "mode": MODE,
            "success_count": report["success_count"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
        flush=True,
    )
