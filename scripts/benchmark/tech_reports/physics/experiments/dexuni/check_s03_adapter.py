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
"""Temporarily check native DexSim against an EmbodiChain S03 run.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/check_s03_adapter.py --reference-run PATH
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


def _state_differences(reference: dict, candidate: dict) -> np.ndarray:
    if not np.array_equal(reference["joint_names"], candidate["joint_names"]):
        raise AssertionError("Joint order differs")
    if reference["particles"].shape[1:] != candidate["particles"].shape[1:]:
        raise AssertionError("Particle topology differs")
    rows = []
    for phase in range(9):
        left = np.flatnonzero(reference["phase"] == phase)
        right = np.flatnonzero(candidate["phase"] == phase)
        if not len(left) or not len(right):
            raise AssertionError(f"Missing phase {phase}")
        li = left[np.rint(np.linspace(0, len(left) - 1, 5)).astype(int)]
        ri = right[np.rint(np.linspace(0, len(right) - 1, 5)).astype(int)]
        delta = reference["particles"][li] - candidate["particles"][ri]
        rms = 1000 * np.sqrt(np.mean(np.sum(delta**2, axis=-1)))
        center = 1000 * np.linalg.norm(delta.mean(axis=1), axis=1).max()
        corner = 1000 * np.max(
            np.abs(reference["corner_height"][li] - candidate["corner_height"][ri])
        )
        if phase == 6:
            corner = 1000 * max(
                abs(
                    reference["corner_height"][left].mean()
                    - candidate["corner_height"][right].mean()
                ),
                abs(
                    reference["corner_height"][left[-1]]
                    - candidate["corner_height"][right[-1]]
                ),
            )
        rows.append((rms, center, corner))
    return np.asarray(rows)


def compare_states(
    reference: dict, native: dict, repeat: dict | list[dict] | None = None
) -> None:
    """Check motion against a same-platform repeat envelope and fixed grasp gate.

    Args:
        reference: Recorded EmbodiChain states.
        native: Ephemeral same-backend native states.
        repeat: Independent EmbodiChain repeat for free-sheet contact variation.
    """
    delta = _state_differences(reference, native)
    limits = np.full_like(delta, 2.0)
    if repeat is not None:
        # Free folds amplify contact-order noise even within the same facade.
        # Use twice the observed repeat deviation plus 1 mm, while requiring
        # the stable grasped material corner to stay within 2 mm during hold and transport.
        repeats = repeat if isinstance(repeat, list) else [repeat]
        baseline = np.maximum.reduce(
            [_state_differences(reference, item) for item in repeats]
        )
        limits = np.maximum(limits, 2 * baseline + 1)
        limits[3:5, 2] = 2.0
    if not np.isfinite(delta).all() or (delta > limits).any():
        phases = np.flatnonzero((delta > limits).any(axis=1)).tolist()
        raise AssertionError(
            f"Adapter motion exceeds within-platform repeat envelope in phases {phases}"
        )


def main() -> None:
    """Run native checking in a temporary directory and delete all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Check stable grasp adaptation without final free-fold acceptance",
    )
    parser.add_argument(
        "--repeat-envelope",
        action="store_true",
        help="Optional historical full-sheet stochastic-envelope diagnostic; not part of the fixed-five protocol",
    )
    parser.add_argument(
        "--diagnose-native",
        action="store_true",
        help="Inspect native model invariants and failed check names without extra facade repeats; all native files remain ephemeral",
    )
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
                str(
                    TECH.parents[2]
                    / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/inputs"
                ),
            )
        )
        .expanduser()
        .resolve()
    )
    env = dict(os.environ, DEXUNI_ALLOW_PILOT="1", DEXUNI_INPUT_ROOT=str(cache))
    spec = importlib.util.spec_from_file_location(
        "s03_metrics", TECH / "reporting/dexuni/s03/analyze.py"
    )
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.INPUTS = cache / "s03"
    reference_run = analyzer.load_run(reference_path)
    reference_metrics = analyzer.task_metrics(reference_run)
    with tempfile.TemporaryDirectory(
        prefix=".s03_adapter_",
        dir=TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni",
    ) as temp:
        temporary = Path(temp)
        repeat_runs = []
        repeat_metrics = []
        for repeat_index in range(
            2
            if args.repeat_envelope
            and not args.repeat_only
            and not args.diagnose_native
            else 0
        ):
            repeat_command = [
                sys.executable,
                str(TECH / "physics/platforms/dexuni/embodichain/s03.py"),
                "--state",
                manifest["state"]["id"],
                "--variant",
                manifest["variant"],
                "--kind",
                "full",
                "--images",
                "none",
                "--out-root",
                str(temporary / f"repeat{repeat_index}"),
            ]
            with (temporary / f"repeat{repeat_index}.log").open("w") as stream:
                subprocess.run(
                    repeat_command,
                    cwd=TECH,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                    check=True,
                )
            repeat_path = next(
                (temporary / f"repeat{repeat_index}/raw_data/embodichain").iterdir()
            )
            repeat_run = analyzer.load_run(repeat_path)
            one_metrics = analyzer.task_metrics(repeat_run)
            if not one_metrics["success"]:
                raise AssertionError(
                    "Within-platform repeat fails task acceptance: "
                    + json.dumps(one_metrics, default=str)
                )
            repeat_runs.append(repeat_run)
            repeat_metrics.append(one_metrics)
        command = [
            sys.executable,
            str(
                TECH
                / f"physics/platforms/dexuni/{'embodichain' if args.repeat_only else 'dexsim'}/s03.py"
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
        # Invariant model arrays prove the facade did not alter physical inputs.
        with (
            np.load(reference_path / "newton_model.npz") as ref_model,
            np.load(native_path / "newton_model.npz") as native_model,
        ):
            for key in (
                "particle_mass",
                "particle_flags",
                "tri_indices",
                "tri_materials",
                "edge_indices",
                "edge_bending_properties",
                "particle_q",
                "particle_qd",
                "particle_radius",
                "particle_ke",
                "particle_kd",
                "particle_mu",
                "particle_colors",
                "tri_poses",
                "tri_areas",
                "edge_rest_angle",
                "edge_rest_length",
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
                # Surface-only Newton models omit tetrahedral arrays entirely.
                if (
                    key.startswith("tet_")
                    and key not in ref_model
                    and key not in native_model
                ):
                    continue
                if not np.allclose(
                    ref_model[key],
                    native_model[key],
                    rtol=1e-6,
                    atol=1e-8,
                    equal_nan=True,
                ):
                    raise AssertionError(f"Adapted physics model differs: {key}")
        if args.diagnose_native:
            print(
                "Native model invariants match; task failed checks: "
                + native_metrics.get("failed_checks", ""),
                flush=True,
            )
            return
        required = (
            (
                "finite",
                "controller_finished",
                "pass_lift",
                "pass_hold",
                "pass_transport",
                "pass_descent",
            )
            if args.performance_only
            else ("success",)
        )
        if not all(reference_metrics[k] and native_metrics[k] for k in required):
            raise AssertionError(
                "One side fails the task acceptance checks: "
                + native_metrics.get("failed_checks", "")
            )
        if args.repeat_envelope:
            compare_states(
                reference_run["states"],
                native_run["states"],
                None if not repeat_runs else [r["states"] for r in repeat_runs],
            )
        else:
            left, right = reference_run["states"], native_run["states"]
            if not np.array_equal(left["time"], right["time"]) or not np.allclose(
                left["target_q"], right["target_q"], atol=1e-7
            ):
                raise AssertionError("Adapter control commands or cadence differ")
            if np.linalg.norm(left["tcp"] - right["tcp"], axis=1).max() > 0.002:
                raise AssertionError("Adapter TCP differs by more than 2 mm")
        for key, tolerance in (
            ("corner_hold_min_mm", 2.0),
            ("transport_corner_min_mm", 2.0),
            ("corner_hold_drop_mm", 2.0),
            ("release_corner_height_mm", 2.0),
        ):
            if args.performance_only and key == "release_corner_height_mm":
                continue
            if repeat_metrics and key == "release_corner_height_mm":
                tolerance = max(
                    tolerance,
                    2
                    * max(
                        abs(reference_metrics[key] - item[key])
                        for item in repeat_metrics
                    )
                    + 1,
                )
            elif key == "release_corner_height_mm" and not args.repeat_envelope:
                # Both must satisfy the same return-to-table task threshold.
                # Post-release free folds are not a deterministic replay.
                tolerance = (
                    1000
                    * reference_run["config"]["acceptance"]["release_corner_height_max"]
                )
            if abs(reference_metrics[key] - native_metrics[key]) > tolerance:
                raise AssertionError(
                    f"Adapter metric differs: {key}; absolute difference "
                    f"{abs(reference_metrics[key] - native_metrics[key]):.3f} mm "
                    f"exceeds {tolerance:.3f} mm"
                )
    print(
        "S03 adapter check passed; native temporary data and logs deleted.", flush=True
    )


if __name__ == "__main__":
    main()
