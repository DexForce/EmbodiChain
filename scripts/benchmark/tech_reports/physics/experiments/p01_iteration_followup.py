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

"""Run configured P-01 groups in separate, sequential simulator processes."""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs/P-01/iteration-followup.json"
    )
    plan = parser.parse_args().plan.resolve()
    batch = json.loads(plan.read_text())
    for item in batch["runs"]:
        out = (
            ROOT
            / "results/P-01"
            / (batch.get("run_prefix", "20261014_study_") + item["id"])
        )
        assert not out.exists(), out
        out.mkdir()
        command = [
            "/home/oem/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "isaaclab",
            "python",
            "code/isaaclab/p01_dynamics.py",
            "--phase",
            item.get("phase", "investigation"),
            "--visualizer",
            "none",
            "--device",
            "cuda:0",
            "--output",
            str(out),
        ]
        settings = {
            "position_iterations": 4,
            "velocity_iterations": 1,
            "solver": "TGS",
            **batch,
            **item,
        }
        for key in [
            "case",
            "dt",
            "position_iterations",
            "velocity_iterations",
            "solver",
            "count",
        ]:
            command += ["--" + key.replace("_", "-"), str(settings[key])]
        if item["case"] == "pendulum":
            command += ["--probe-body-velocity"]
        if not settings.get("external_forces_every_iteration", True):
            command += ["--no-external-forces-every-iteration"]
        if item.get("phase") == "capture":
            command += ["--record"]
        shutil.copy2(plan, out / plan.name)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        print("Starting", out.name, flush=True)
        with (
            (out / "stdout.log").open("w") as stdout,
            (out / "stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        shutil.copy2(plan, out / "config" / plan.name)
        (out / "followup-plan.json").write_text(
            json.dumps(
                dict(
                    plan=str(plan.relative_to(ROOT)),
                    sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                    command=command,
                ),
                indent=2,
            )
            + "\n"
        )
        assert (
            result.returncode == 0
            and json.loads((out / "run.json").read_text())["status"] == "completed"
        ), out
        print("Completed", out.name, flush=True)


if __name__ == "__main__":
    main()
