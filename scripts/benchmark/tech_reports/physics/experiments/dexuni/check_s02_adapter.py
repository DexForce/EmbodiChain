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
"""Temporarily check native DexSim against an EmbodiChain S02 run.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/check_s02_adapter.py --reference-run PATH
Native measurements and logs are removed on exit and never enter benchmark reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[6]
TECH = REPO / "scripts/benchmark/tech_reports"
__all__ = ["compare_states", "main"]


def compare_states(reference: dict, native: dict) -> None:
    """Check phase-aligned material-node motion with millimetre tolerances.

    Args:
        reference: Recorded EmbodiChain state arrays.
        native: Ephemeral native DexSim state arrays in the same node order.
    """
    if not np.array_equal(reference["joint_names"], native["joint_names"]):
        raise AssertionError("Joint order differs")
    if reference["particles"].shape[1:] != native["particles"].shape[1:]:
        raise AssertionError("Particle topology differs")
    for phase in range(10):
        left = np.flatnonzero(reference["phase"] == phase)
        right = np.flatnonzero(native["phase"] == phase)
        if not len(left) or not len(right):
            raise AssertionError(f"Missing phase {phase}")
        li = left[np.rint(np.linspace(0, len(left) - 1, 5)).astype(int)]
        ri = right[np.rint(np.linspace(0, len(right) - 1, 5)).astype(int)]
        delta = reference["particles"][li] - native["particles"][ri]
        rms_mm = 1000 * np.sqrt(np.mean(np.sum(delta**2, axis=-1)))
        center_mm = 1000 * np.linalg.norm(delta.mean(axis=1), axis=1).max()
        width_mm = 1000 * np.max(np.abs(reference["width"][li] - native["width"][ri]))
        # Opening contacts are nonsmooth: independent runs of the same backend
        # can rebound on adjacent frames. Check average deformation and the
        # settled endpoint there, while retaining the material-node motion gate.
        if phase == 7:
            width_mm = 1000 * max(
                abs(reference["width"][left].mean() - native["width"][right].mean()),
                abs(reference["width"][left[-1]] - native["width"][right[-1]]),
            )
        if rms_mm > 2 or center_mm > 2 or width_mm > (1 if phase == 7 else 2):
            raise AssertionError(
                f"Adapter differs in phase {phase}: particle RMS {rms_mm:.3f}mm, center {center_mm:.3f}mm, width {width_mm:.3f}mm"
            )


def main() -> None:
    """Run native checking in a temporary directory and delete all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--repeat-only",
        action="store_true",
        help="Diagnose within-EmbodiChain stochastic variation before attributing a mismatch",
    )
    args = parser.parse_args()
    reference_path = args.reference_run.resolve()
    manifest = json.loads((reference_path / "manifest.json").read_text())
    if (
        manifest["platform"] != "embodichain"
        or Path(manifest["executable"]).resolve() != Path(sys.executable).resolve()
    ):
        raise ValueError(
            "Use the same Python/backend installation as the EmbodiChain reference"
        )
    # A custom candidate snapshot is the parent of the task inputs; use exactly its config.
    cache = (
        Path(
            os.environ.get(
                "DEXUNI_INPUT_ROOT",
                "~/.cache/embodichain/report-assets/local/dexuni/pilot_v1",
            )
        )
        .expanduser()
        .resolve()
    )
    env = dict(os.environ, DEXUNI_ALLOW_PILOT="1", DEXUNI_INPUT_ROOT=str(cache))
    spec = importlib.util.spec_from_file_location(
        "s02_metrics", TECH / "reporting/dexuni/s02/analyze.py"
    )
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.INPUTS = cache / "s02"
    reference_run = analyzer.load_run(reference_path)
    reference_metrics = analyzer.task_metrics(reference_run)
    with tempfile.TemporaryDirectory(
        prefix=".s02_adapter_",
        dir=TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni",
    ) as temp:
        temporary = Path(temp)
        command = [
            sys.executable,
            str(
                TECH
                / f"physics/platforms/dexuni/{'embodichain' if args.repeat_only else 'dexsim'}/s02.py"
            ),
            "--state",
            manifest["state"]["id"],
            "--variant",
            manifest["variant"],
            "--kind",
            "full",
            "--images",
            "none",
            "--out-root",
            str(temporary),
        ]
        with (temporary / "native.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=TECH,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1200,
            )
        if result.returncode:
            print((temporary / "native.log").read_text()[-3000:], file=sys.stderr)
            raise RuntimeError(f"Native diagnostic exited {result.returncode}")
        native_path = next(
            (
                temporary
                / ("raw_data/embodichain" if args.repeat_only else "raw_data/dexsim")
            ).iterdir()
        )
        native_run = analyzer.load_run(native_path)
        native_metrics = analyzer.task_metrics(native_run)
        if reference_run["config"] != native_run["config"]:
            raise AssertionError("Input configurations differ")
        if (
            reference_run["manifest"]["dexsim_version"]
            != native_run["manifest"]["dexsim_version"]
        ):
            raise AssertionError("Backend versions differ")
        if not reference_metrics["success"] or not native_metrics["success"]:
            raise AssertionError("One side fails the task acceptance checks")
        # Invariant model arrays prove the facade did not alter physical inputs.
        with (
            np.load(reference_path / "newton_model.npz") as ref_model,
            np.load(native_path / "newton_model.npz") as native_model,
        ):
            for key in (
                "particle_mass",
                "particle_q",
                "particle_qd",
                "tet_indices",
                "tet_materials",
                "joint_armature",
                "joint_target_ke",
                "joint_target_kd",
                "joint_target_mode",
                "joint_limit_lower",
                "joint_limit_upper",
                "body_mass",
                "body_inertia",
                "joint_axis",
            ):
                if not np.allclose(
                    ref_model[key],
                    native_model[key],
                    rtol=1e-6,
                    atol=1e-8,
                    equal_nan=True,
                ):
                    raise AssertionError(f"Adapted physics model differs: {key}")
        compare_states(reference_run["states"], native_run["states"])
        for key, tolerance in (
            ("grasp_compression_pct", 2.0),
            ("lift_min_height_mm", 2.0),
            ("hold_drop_mm", 2.0),
            ("place_center_error_mm", 2.0),
        ):
            if abs(reference_metrics[key] - native_metrics[key]) > tolerance:
                raise AssertionError(f"Adapter metric differs: {key}")
    print(
        "S02 adapter check passed; native temporary data and logs deleted.", flush=True
    )


if __name__ == "__main__":
    main()
