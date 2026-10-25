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


def _build_parameterized(
    source: Path, output: Path, urdf: Path | None = None
) -> dict[str, object]:
    """Derive one POE RR system whose coefficients depend on the lock angle.

    Every joint factor starts with ``Rz(q)``. That is the only factorization
    needed by RR elimination; a distal-DH common normal is unnecessary. The
    original locked rotation appears inside one fixed factor, affinely in its
    sine/cosine, so arbitrary lock coordinates need no chart interpolation.
    """
    import sympy as sp

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
    original = source / "src/ssik/prebuilt/flexiv/rizon4_jl.npz"
    module = original.with_name("rizon4_ik.py")
    if (
        hashlib.sha256(original.read_bytes()).hexdigest() != _ORIGINAL_SHA256
        or hashlib.sha256(module.read_bytes()).hexdigest() != _SOURCE_SHA256
    ):
        raise ValueError("Pinned parameterized RR source integrity mismatch")
    sys.path.insert(0, str(source / "src"))
    from ssik.prebuilt.flexiv.rizon4_ik import _KB
    from ssik.refinement import kinbody_fk_jacobian_batch
    from ssik.solvers.ikgeo._raghavan_roth import _so3_basis

    with np.load(original, allow_pickle=False) as archive:
        original_arrays = {key: archive[key] for key in archive.files}
    source_axes = original_arrays["axes"]
    source_offsets = np.vstack(
        (original_arrays["t_left"][:, :3, 3], original_arrays["t_right"][-1, :3, 3])
    )
    axes = source_axes.copy()
    axes[np.abs(axes) < 5e-9] = 0
    axes[np.abs(axes - 1) < 5e-9] = 1
    axes[np.abs(axes + 1) < 5e-9] = -1
    offsets = np.round(source_offsets, decimals=9)
    home_rotation = original_arrays["t_right"][-1, :3, :3].copy()
    home_rotation[np.abs(home_rotation) < 5e-9] = 0
    home_rotation[np.abs(home_rotation - 1) < 5e-9] = 1
    home_rotation[np.abs(home_rotation + 1) < 5e-9] = -1
    if not np.array_equal(axes @ axes.T == 1, source_axes @ source_axes.T > 1 - 1e-8):
        raise ValueError(
            "Pinned RR axis roundoff canonicalization changed its geometry"
        )

    def embed(rotation):
        return rotation.row_join(sp.zeros(3, 1)).col_join(sp.Matrix([[0, 0, 0, 1]]))

    def translate(vector):
        column = sp.Matrix([sp.Rational(str(float(value))) for value in vector])
        return sp.eye(3).row_join(column).col_join(sp.Matrix([[0, 0, 0, 1]]))

    def rz(sine, cosine):
        return sp.Matrix(
            [[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

    def inverse(transform):
        rotation = transform[:3, :3].T
        return rotation.row_join(-rotation * transform[:3, 3]).col_join(
            sp.Matrix([[0, 0, 0, 1]])
        )

    frames = []
    for axis in axes[[0, 1, 3, 4, 5, 6]]:
        x = np.cross(np.array([0, 1, 0]), axis)
        if np.linalg.norm(x) == 0:
            x = np.array([1, 0, 0])
        x = x / np.linalg.norm(x)
        frames.append(
            sp.Matrix(np.column_stack((x, np.cross(axis, x), axis)).astype(int))
        )
    lock_s, lock_c = sp.symbols("lock_s lock_c", real=True)
    sine, cosine = sp.symbols("s0:6", real=True), sp.symbols("c0:6", real=True)
    factors = [
        embed(frames[0].T) * translate(offsets[1]) * embed(frames[1]),
        embed(frames[1].T)
        * translate(offsets[2])
        * rz(lock_s, lock_c)
        * translate(offsets[3])
        * embed(frames[2]),
        embed(frames[2].T) * translate(offsets[4]) * embed(frames[3]),
        embed(frames[3].T) * translate(offsets[5]) * embed(frames[4]),
        embed(frames[4].T) * translate(offsets[6]) * embed(frames[5]),
        embed(frames[5].T)
        * translate(offsets[7])
        * sp.Matrix(home_rotation.astype(int))
        .row_join(sp.zeros(3, 1))
        .col_join(sp.Matrix([[0, 0, 0, 1]])),
    ]
    forward = [rz(sine[index], cosine[index]) * factors[index] for index in range(6)]
    backward = [
        inverse(factors[index]) * rz(-sine[index], cosine[index]) for index in range(6)
    ]
    target_symbols = sp.symbols("T_:12", real=True)
    target = sp.Matrix(3, 4, target_symbols).col_join(sp.Matrix([[0, 0, 0, 1]]))
    lhs = forward[1] * forward[2] * forward[3]
    rhs = backward[0] * target * backward[5] * backward[4]
    ll, lr, pl, pr = lhs[:3, 2], rhs[:3, 2], lhs[:3, 3], rhs[:3, 3]
    ppl, ppr, lpl, lpr = pl.dot(pl), pr.dot(pr), ll.dot(pl), lr.dot(pr)
    equations = (
        list(ll - lr)
        + list(pl - pr)
        + list(ll.cross(pl) - lr.cross(pr))
        + [ppl - ppr, lpl - lpr]
        + list(ppl * ll - 2 * lpl * pl - ppr * lr + 2 * lpr * pr)
    )
    active = (0, 1, 2, 3, 5)
    active_s = tuple(sine[index] for index in active) + (lock_s,)
    active_c = tuple(cosine[index] for index in active) + (lock_c,)
    so3_groebner = sp.groebner(_so3_basis(target_symbols), *target_symbols)
    ideal = [s * s + c * c - 1 for s, c in zip(active_s, active_c)] + [
        poly.as_expr() for poly in so3_groebner.polys
    ]
    reduced = [
        sp.expand(
            sp.reduced(sp.expand(eq), ideal, *active_s, *active_c, *target_symbols)[1]
        )
        for eq in equations
    ]
    left = [
        sine[2] * sine[3],
        sine[2] * cosine[3],
        cosine[2] * sine[3],
        cosine[2] * cosine[3],
        sine[2],
        cosine[2],
        sine[3],
        cosine[3],
        1,
    ]
    right = [
        sine[0] * sine[5],
        sine[0] * cosine[5],
        cosine[0] * sine[5],
        cosine[0] * cosine[5],
        sine[0],
        cosine[0],
        sine[5],
        cosine[5],
    ]
    ps, pc, po, q = [sp.zeros(14, size) for size in (9, 9, 9, 8)]
    joint_generators = tuple(sine[index] for index in active) + tuple(
        cosine[index] for index in active
    )
    for row, equation in enumerate(reduced):
        polynomial = sp.Poly(equation, *joint_generators)
        for column, monomial in enumerate(left):
            ps[row, column] = polynomial.coeff_monomial(sp.expand(sine[1] * monomial))
            pc[row, column] = polynomial.coeff_monomial(sp.expand(cosine[1] * monomial))
            po[row, column] = polynomial.coeff_monomial(monomial)
        for column, monomial in enumerate(right):
            q[row, column] = -polynomial.coeff_monomial(monomial)
        reconstructed = sum(
            (ps[row, col] * sine[1] + pc[row, col] * cosine[1] + po[row, col])
            * monomial
            for col, monomial in enumerate(left)
        ) - sum(q[row, col] * monomial for col, monomial in enumerate(right))
        if sp.expand(equation - reconstructed) != 0:
            raise ValueError(
                f"Parameterized RR row {row} failed exact symbolic reconstruction"
            )

    p_affine = np.zeros((3, 3, 14, 9))
    for kind, matrix in enumerate((ps, pc, po)):
        for phase, variable in ((1, lock_s), (2, lock_c)):
            coefficient = matrix.diff(variable)
            if any(entry.free_symbols for entry in coefficient):
                raise ValueError(
                    "RR P factors are not affinely separable from the lock angle"
                )
            p_affine[kind, phase] = np.array(coefficient, dtype=np.float64)
        if kind != 2:
            p_affine[kind, 0] = np.array(
                matrix.subs({lock_s: 0, lock_c: 0}), dtype=np.float64
            )
    po_target = po.subs({lock_s: 0, lock_c: 0})
    if any(entry.has(lock_s, lock_c) for entry in q):
        raise ValueError("RR Q unexpectedly depends on the lock angle")
    basis, indices, values = {}, [], []
    for kind, matrix in enumerate((po_target, q)):
        for row in range(matrix.rows):
            for col in range(matrix.cols):
                if matrix[row, col] == 0:
                    continue
                polynomial = sp.Poly(matrix[row, col], *target_symbols)
                for monomial, coefficient in zip(
                    polynomial.monoms(), polynomial.coeffs()
                ):
                    if monomial not in basis:
                        basis[monomial] = len(basis)
                    indices.append((kind, row, col, basis[monomial]))
                    values.append(float(coefficient))
    mono_factors = np.full((len(basis), 4), -1, dtype=np.int32)
    for monomial, index in basis.items():
        entries = [
            variable for variable, power in enumerate(monomial) for _ in range(power)
        ]
        if len(entries) > 4:
            raise ValueError("Parameterized RR target monomial exceeds degree four")
        mono_factors[index, : len(entries)] = entries
    factor_affine = np.zeros((3, 6, 4, 4))
    for index, factor in enumerate(factors):
        factor_affine[0, index] = np.array(
            factor.subs({lock_s: 0, lock_c: 0}), dtype=np.float64
        )
        factor_affine[1, index] = np.array(factor.diff(lock_s), dtype=np.float64)
        factor_affine[2, index] = np.array(factor.diff(lock_c), dtype=np.float64)
    pre = np.array(translate(offsets[0]) * embed(frames[0]), dtype=np.float64)

    def canonical_fk(joints):
        result = pre.copy()
        phase_s, phase_c = np.sin(joints[2]), np.cos(joints[2])
        fi = factor_affine[0] + phase_s * factor_affine[1] + phase_c * factor_affine[2]
        for angle, factor in zip(np.delete(joints, 2), fi):
            sa, ca = np.sin(angle), np.cos(angle)
            rotation = np.array(
                [[ca, -sa, 0, 0], [sa, ca, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            result = result @ rotation @ factor
        return result

    rng = np.random.default_rng(20261026)
    angle_cases = np.r_[
        [-3.0543, -3.0543 + 1e-8, -1e-6, -1e-9, 0, 1e-9, 1e-6, 3.0543 - 1e-8, 3.0543],
        rng.uniform(-3.0543, 3.0543, 119),
    ]
    joints = rng.uniform(
        original_arrays["lo"], original_arrays["hi"], size=(len(angle_cases) * 16, 7)
    )
    joints[:, 2] = np.repeat(angle_cases, 16)
    actual = kinbody_fk_jacobian_batch(_KB, joints)[0]
    reconstructed = np.stack([canonical_fk(qpos) for qpos in joints])
    maximum_poe = float(np.linalg.norm(actual - reconstructed, axis=(1, 2)).max())
    if maximum_poe > 1e-8:
        raise ValueError(
            f"Parameterized original POE FK certificate failed: {maximum_poe}"
        )
    canonical_home = canonical_fk(np.zeros(7))
    pk_certificate = None
    if urdf is not None:
        import pytorch_kinematics as pk
        import torch

        chain = pk.build_serial_chain_from_urdf(
            urdf.read_text(), "flange", "base_link"
        ).to(dtype=torch.float64)
        home = (
            chain.forward_kinematics(torch.zeros((1, 7), dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        correction = np.linalg.solve(home, canonical_home)
        pose = (
            chain.forward_kinematics(torch.from_numpy(joints)).get_matrix().numpy()
            @ correction
        )
        maximum_pk = float(np.linalg.norm(pose - reconstructed, axis=(1, 2)).max())
        if maximum_pk > 2e-6:
            raise ValueError(
                f"Parameterized original PK FK certificate failed: {maximum_pk}"
            )
        pk_certificate = {
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "root": "base_link",
            "end": "flange",
            "joint_order": chain.get_joint_parameter_names(),
            "max_frobenius_error": maximum_pk,
        }
    arrays = dict(
        source_axes=source_axes,
        source_offsets=source_offsets,
        source_limits=np.column_stack((original_arrays["lo"], original_arrays["hi"])),
        canonical_home=canonical_home,
        pre_inverse=np.linalg.inv(pre),
        factor_affine=factor_affine,
        p_affine=p_affine,
        mono_factors=mono_factors,
        coefficient_indices=np.array(indices, dtype=np.int32),
        coefficient_values=np.array(values, dtype=np.float64),
        global_lock_angles=original_arrays["q_lock"],
        local_offsets=np.array(
            [0, -0.001, 0.001, -0.005, 0.005, -0.02, 0.02, 0, 0], dtype=np.float64
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {
        "ssik_commit": _PIN,
        "source_module_sha256": _SOURCE_SHA256,
        "source_sidecar_sha256": _ORIGINAL_SHA256,
        "sidecar_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "license": "BSD-3-Clause; ../ssik_LICENSE.txt",
        "algebra": "Parameterized POE RR, Rz(q_i) F_i(phi), linearity=1, left=(2,3), right=(0,5), drop=4; exact trig and full SO3 Groebner reduction",
        "symbolic_reconstruction_zero_rows": 14,
        "target_max_polynomial_degree": 4,
        "p_lock_dependence": "affine in sine/cosine; no mixed target/lock monomials",
        "q_lock_dependence": "independent",
        "target_coefficient_terms": len(values),
        "max_axis_roundoff_canonicalization": float(
            np.linalg.norm(source_axes - axes, axis=1).max()
        ),
        "max_offset_roundoff_canonicalization": float(
            np.linalg.norm(source_offsets - offsets, axis=1).max()
        ),
        "fk_certificate": {
            "seed": 20261026,
            "lock_angles": angle_cases.tolist(),
            "postures_per_lock_angle": 16,
            "original_poe_max_frobenius_error": maximum_poe,
            "original_pk": pk_certificate,
        },
        "global_samples": 16,
        "local_samples": 9,
        "slots_per_sample": 24,
        "adaptive_local_samples": "last two are redundancy-only predictions from a single original-model tangent, minimizing displacement and centering its hard-limit/step feasible interval; joint results always come from RR roots",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssik-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameterized", "--parametric", action="store_true")
    parser.add_argument("--urdf", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            (
                _build_parameterized(
                    arguments.ssik_source.resolve(),
                    arguments.output.resolve(),
                    arguments.urdf.resolve() if arguments.urdf else None,
                )
                if arguments.parameterized
                else _build(arguments.ssik_source.resolve(), arguments.output.resolve())
            ),
            indent=2,
        )
    )
