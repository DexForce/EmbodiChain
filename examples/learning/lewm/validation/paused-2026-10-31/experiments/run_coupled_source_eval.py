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

"""Private official-query replay of the source-stage coupled adapter."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

from cube_coupled_source_trial import CoupledSourceCube, archived
from examples.learning.lewm import _official_eval as evaluator

# Repeated in each fresh spawned child before its deferred task import.
archived.OGBenchCubeEnv = CoupledSourceCube
original_identity = evaluator._native_runtime_identity
ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")


def identity(reference):
    report = original_identity(reference)
    paths = [
        Path(__file__),
        Path(__file__).with_name("cube_coupled_source_trial.py"),
        Path(__file__).with_name("real14_link_bias.py"),
        Path(__file__).with_name("cube_point_kinematics.py"),
        Path(__file__).with_name("load_source_servo_archive.py"),
        ROOT / "alignment/real14-coupled-impulse/source.json",
        ROOT
        / "alignment/rejected-source-servo-production/embodichain/compute/dynamics/_inertia.py",
        Path("/root/sources/EmbodiChain-pr895-lewm/embodichain/lab/sim/cfg/scene.py"),
        Path("/root/sources/EmbodiChain-pr895-lewm/embodichain/lab/sim/sim_manager.py"),
    ]
    report["private_source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    report["private_trial"] = (
        "source-stage point anchors, articulated bias and compliant mimic velocity term; PGS velocity passes zero"
    )
    return report


evaluator._native_runtime_identity = identity

if __name__ == "__main__":
    output = ROOT / "evaluation/dexsim-coupled-source-stage-source-replay"
    if output.exists():
        raise FileExistsError(output)
    report = evaluator.evaluate_dexsim(
        reference=ROOT / "alignment-point-response-pgs",
        queries=ROOT / "alignment/official-queries.h5",
        weights=ROOT / "weights/weights.pt",
        output=output,
        device="cuda:7",
        replay=ROOT / "evaluation/mujoco/trace.npz",
        worker_devices=["cuda:6", "cuda:5"],
    )
    print(
        {
            "success_count": report["success_count"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
        flush=True,
    )
