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

"""Compare one free-fall case in separate EmbodiChain and Isaac Lab processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ...assets.runtime import add_asset_arguments, prepare_inputs

__all__ = ["main"]


def main() -> None:
    """Download the common inputs, run both simulators, and compare their traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-python", type=Path, required=True)
    parser.add_argument("--isaaclab-repo", type=Path)
    parser.add_argument("--embodichain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    add_asset_arguments(parser)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[5]
    prepared = prepare_inputs(
        "p-01",
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    input_dir, provenance = prepared.bundle, prepared.provenance
    inputs = json.loads((input_dir / "initial-states.json").read_text())
    z0 = inputs["freefall_z_m"][0]
    isaac_cfg = json.loads((input_dir / "isaaclab.json").read_text())
    dexuni_cfg = json.loads((input_dir / "embodichain-dexuni.json").read_text())
    for key in ("gravity_m_s2", "mass_kg", "cube_edge_m", "collision"):
        if isaac_cfg[key] != dexuni_cfg[key]:
            raise ValueError(f"P-01 platform configurations disagree on {key}")
    gravity = isaac_cfg["gravity_m_s2"]
    output.mkdir(parents=True)
    (output / "assets.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result = {}
    for platform, python in (
        ("embodichain", args.embodichain_python),
        ("isaaclab", args.isaaclab_python),
    ):
        run_dir = output / platform
        command = [
            str(python.absolute()),
            "-m",
            f"scripts.benchmark.tech_reports.physics.platforms.{platform}.p01_dynamics",
            "--input-bundle",
            str(input_dir),
            "--case",
            "freefall",
            "--count",
            "1",
            "--duration",
            "1",
            "--dt",
            "0.001",
            "--output",
            str(run_dir),
        ]
        if platform == "isaaclab":
            command.extend(["--visualizer", "none"])
            if args.isaaclab_repo:
                command.extend(["--source-repo", str(args.isaaclab_repo.resolve())])
        print(f"Running {platform}; logs in {output}", flush=True)
        with (output / f"{platform}.log").open("w") as log:
            process = subprocess.run(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT
            )
        if process.returncode:
            raise RuntimeError(f"{platform} failed; see {output / (platform + '.log')}")
        run = json.loads((run_dir / "run.json").read_text())
        with np.load(run_dir / "episode_00.npz") as data:
            t = data["t"]
            state = data["state"]
        if run["status"] != "completed" or not np.isfinite(state).all():
            raise ValueError(f"{platform} did not produce a complete trajectory")
        if (
            t.shape != (1001,)
            or state.shape != (1001, 2)
            or not np.allclose(t, np.arange(1001) * 0.001)
        ):
            raise ValueError(f"{platform} returned a different time grid")
        if not np.allclose(state[0], [z0, 0], atol=1e-6, rtol=0):
            raise ValueError(
                f"{platform} returned a different initial state: {state[0]}"
            )
        error = state - np.column_stack((z0 - 0.5 * gravity * t**2, -gravity * t))
        result[platform] = {
            "solver": run.get("resolved_solver", run["solver"]),
            "source_revision": run.get("repository_commit", run.get("isaaclab_commit")),
            "packages": run["packages"],
            "end_z_m": float(state[-1, 0]),
            "max_position_error_mm": float(1000 * np.abs(error[:, 0]).max()),
            "max_speed_error_mm_s": float(1000 * np.abs(error[:, 1]).max()),
            "position_rmse_mm": float(1000 * np.sqrt(np.mean(error[:, 0] ** 2))),
        }
    summary = {
        "case": "freefall",
        "dataset_revision": provenance["revision"],
        "samples_per_platform": 1001,
        "dt_s": 0.001,
        "duration_s": 1.0,
        "gravity_m_s2": gravity,
        "initial_z_m": z0,
        "platforms": result,
    }
    formatted = json.dumps(summary, indent=2)
    (output / "comparison.json").write_text(formatted + "\n")
    print(formatted)


if __name__ == "__main__":
    main()
