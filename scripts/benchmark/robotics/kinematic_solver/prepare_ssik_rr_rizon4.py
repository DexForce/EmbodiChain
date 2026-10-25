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

"""Offline build of the licensed Rizon4 RR coefficient bundle used by CUDA.

Example::

    python prepare_ssik_rr_rizon4.py --ssik-source /path/to/pinned/ssik \
        --output /path/to/ssik_rr_rizon4.npz

The pinned source and its build dependencies are needed only by this utility.
Runtime preparation loads the resulting data without SSIK or symbolic algebra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

__all__: list[str] = []

_PIN = "4109147f37a7ca3e2516c83853aca583612783f3"
_SOURCE_SHA256 = "16c1734c249261c179098ada3d36581c730e4deee6e4f78ee4ac8824d7118073"
_ORIGINAL_SHA256 = "a047bb730836954447ad6dc99a1683db5476d48da82dc3093fbcaaec9cfe3066"


def _build(source: Path, output: Path) -> dict[str, object]:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != _PIN:
        raise ValueError(f"SSIK source must be pinned to {_PIN}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "src/ssik"],
        check=True,
        capture_output=True,
    )
    prebuilt = source / "src/ssik/prebuilt/flexiv/rizon4_ik.py"
    original = prebuilt.with_name("rizon4_jl.npz")
    for path, expected in (
        (prebuilt, _SOURCE_SHA256),
        (original, _ORIGINAL_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source integrity mismatch: {path}")
    sys.path.insert(0, str(source / "src"))
    from ssik._native import _derive_rr_tensor
    from ssik.prebuilt.flexiv.rizon4_ik import _KB
    from ssik.refinement import kinbody_fk_jacobian_batch
    from ssik.solvers.ikgeo._raghavan_roth import _fk_dh
    from ssik.solvers.jointlock.seven_r import _lock_joint

    with np.load(original, allow_pickle=False) as archive:
        constants = {key: archive[key] for key in archive.files}
    sample, link = 8, 1
    if int(constants["lock_idx"]) != 2 or constants["q_lock"][sample] != 0:
        raise ValueError(
            "The pinned RR grid does not have its expected zero-lock slice"
        )
    # With original joint3 locked to zero, joints2/4 are anti-parallel.
    # Their DH common normal is the perpendicular component of the sum of
    # intervening POE offsets, not the unstable skew-axis intersection chart.
    axis = constants["axes"][1]
    axis = axis / np.linalg.norm(axis)
    offset = constants["t_left"][2, :3, 3] + constants["t_left"][3, :3, 3]
    corrected_a = float(np.linalg.norm(offset - axis * np.dot(axis, offset)))
    original_a = float(constants["a"][sample, link])
    constants["a"][sample, link] = corrected_a
    tensor = _derive_rr_tensor(
        tuple(constants["alpha"][sample]),
        tuple(constants["a"][sample]),
        tuple(constants["d"][sample]),
        int(constants["linearity_joint"][sample]),
    )
    for key in (
        "linearity_joint",
        "left_bilinear",
        "right_bilinear",
        "drop_joint",
        "p_sin",
        "p_cos",
    ):
        constants[key][sample] = tensor[key]
    constants[f"mono_factors_{sample}"] = tensor["mono_factors"]
    for prefix in ("po", "q"):
        rows, columns, monomials, coefficients = tensor[f"{prefix}_coo"]
        constants[f"{prefix}_rc_{sample}"] = np.column_stack((rows, columns))
        constants[f"{prefix}_mono_{sample}"] = monomials
        constants[f"{prefix}_coeff_{sample}"] = coefficients

    # Certify every chart against the original POE chain, independently of
    # the CUDA RR coefficient evaluator and of the runtime final PK gate.
    rng = np.random.default_rng(20261025)
    q7 = rng.uniform(constants["lo"], constants["hi"], size=(128, 7))
    per_slice_error = []
    for index, angle in enumerate(constants["q_lock"]):
        chain = _lock_joint(_KB, int(constants["lock_idx"]), float(angle))
        q6 = np.delete(q7, int(constants["lock_idx"]), axis=1)
        actual = kinbody_fk_jacobian_batch(chain, q6)[0]
        dh = tuple(constants[key][index] for key in ("alpha", "a", "d"))
        pre = np.linalg.inv(constants["t_pre_inv"][index])
        post = np.linalg.inv(constants["t_post_inv"][index])
        recovered = np.stack(
            [pre @ _fk_dh(q + constants["theta_offset"][index], dh) @ post for q in q6]
        )
        maximum = float(np.linalg.norm(actual - recovered, axis=(1, 2)).max())
        per_slice_error.append(maximum)
        if maximum > 1e-7:
            raise ValueError(
                f"Prepared DH/POE chart {index} failed FK agreement: {maximum}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **constants)
    metadata = {
        "ssik_commit": _PIN,
        "source_module": "ssik.prebuilt.flexiv.rizon4_ik",
        "source_module_sha256": _SOURCE_SHA256,
        "original_sidecar_sha256": _ORIGINAL_SHA256,
        "sidecar_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "license": "BSD-3-Clause; ../ssik_LICENSE.txt",
        "fixed_lock_samples": 16,
        "locked_joint_zero_based": 2,
        "chart_correction": {
            "sample_zero_based": sample,
            "dh_a_zero_based": link,
            "original_m": original_a,
            "corrected_m": corrected_a,
            "derivation": "norm((I - h2*h2.T) @ (p3 + p4)) in normalized original POE coordinates",
            "reason": "At the zero-lock parallel-axis slice the upstream skew-axis DH chart has a 0.88468 mm common-normal error",
        },
        "fk_certificate": {
            "oracle": "Original pinned KinBody POE versus standard DH composition for every chart",
            "seed": 20261025,
            "random_postures_per_slice": 128,
            "maximum_frobenius_error_per_slice": per_slice_error,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssik-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            _build(arguments.ssik_source.resolve(), arguments.output.resolve()),
            indent=2,
        )
    )
