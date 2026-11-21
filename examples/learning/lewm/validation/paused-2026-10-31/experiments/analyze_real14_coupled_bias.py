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

"""Independent 14D native-row model, including the documented PGS mimic pass."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from cube_point_kinematics import PointKinematics

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
OUT = ROOT / "alignment/real14-coupled-impulse"
REF = ROOT / "alignment-point-response-pgs"
source = json.loads((OUT / "source.json").read_text())
contract = json.loads((REF / "physical-contract.json").read_text())
from real14_link_bias import LinkBiasKinematics

model = LinkBiasKinematics()
free = json.loads((OUT / "native-none-base-v0.json").read_text())["results"]
free = {(x["tier"], x["name"]): x for x in free}


def native_rows(row, gain_mode):
    q, v = np.array(row["qpos"][:14]), np.array(row["qvel"][:14])
    points = model.evaluate(q, v)
    J, rhs, regularizer = [], [], []
    if row["mode"] in ("point", "both"):
        basis = [model.evaluate(q, eye) for eye in np.eye(14)]
        eq = contract["equality"]
        for column, (index, _) in enumerate(model.points):
            point = points[column]
            rotation = point["rotation_a"].T
            jac = np.array(
                [
                    -rotation
                    @ (
                        b[column]["velocity"]
                        - np.cross(b[column]["omega_a"], b[column]["residual"])
                    )
                    for b in basis
                ]
            ).T
            ids = np.flatnonzero(np.array(row["equality_ids"]) == index)
            np.testing.assert_allclose(
                -rotation @ np.array(row["J"])[ids],
                np.array([rotation @ b[column]["velocity"] for b in basis]).T,
                atol=1e-12,
                rtol=0,
            )
            target = np.zeros(3)
            if gain_mode == "discrete":
                K, B, d = np.array(row["KBIP"])[ids[0], :3]
                R = row["R"][ids[0]]
                k, b = K * d / R, (500 - 0.002 * K * d) / R
                jv = point["velocity"] - np.cross(point["omega_a"], point["residual"])
                target = (
                    rotation
                    @ (jv - 0.002 * B * point["velocity"] - 0.002 * point["Jdotv"])
                    / (1 - 0.002**2 * K * d)
                )
            else:
                tau, ratio = eq["solref"][index]
                d0, dmax = eq["solimp"][index][:2]
                A0 = sum(
                    contract["body_invweight0"][eq[key][index]][0]
                    for key in ("obj1id", "obj2id")
                )
                R = (1 - d0) / d0 * A0
                k, b = d0 / (dmax * tau * ratio) ** 2 / R, 2 / (dmax * tau) / R
            pos = -rotation @ point["residual"]
            goal = -b / (b + 0.002 * k) * target - k / (b + 0.002 * k) * pos
            J.extend(jac)
            rhs.extend(goal)
            regularizer.extend([1 / (0.002 * (b + 0.002 * k))] * 3)
    if row["mode"] in ("mimic", "both"):
        ids = np.flatnonzero(np.array(row["equality_ids"]) == 2)
        jac = np.array(row["J"])[ids[0]]
        response = jac @ np.linalg.solve(np.array(row["mass"]), jac)
        tau, ratio = contract["equality"]["solref"][2]
        d0, dmax = contract["equality"]["solimp"][2][:2]
        omega = d0 / (dmax * tau * ratio * np.sqrt(1 - d0))
        zeta = ratio / np.sqrt(1 - d0)
        k, b = omega**2 / response, 2 * omega * zeta / response
        J.append(jac)
        rhs.append(-k / (b + 0.002 * k) * row["position"][ids[0]])
        regularizer.append(1 / (0.002 * (b + 0.002 * k)))
    return np.array(J), np.array(rhs), np.array(regularizer)


results = []
for path in sorted(OUT.glob("native-*-v[01].json")):
    native = json.loads(path.read_text())
    if native["mode"] == "none":
        continue
    for measured in native["results"]:
        row = next(
            x
            for x in source["rows"]
            if (x["mode"], x["tier"], x["name"])
            == (native["mode"], measured["tier"], measured["name"])
        )
        M, old = np.array(row["mass"]), np.array(row["qvel"][:14])
        J, rhs, R = native_rows(row, native["point_gains"])
        if native["mode"] in ("point", "both"):
            bias = model.point_solver_bias(
                np.array(row["qpos"][:14]), np.array(row["qvel"][:14])
            )
            rotations = [
                x["rotation_a"].T
                for x in model.evaluate(
                    np.array(row["qpos"][:14]), np.array(row["qvel"][:14])
                )
            ]
            rhs[:6] += 0.002 * np.concatenate([r @ b for r, b in zip(rotations, bias)])
        vf = np.array(free[(row["tier"], row["name"])]["v_final"])
        response = np.linalg.solve(M, J.T)
        impulse = np.linalg.solve(np.diag(R) + J @ response, rhs - J @ vf)
        predicted = vf + response @ impulse
        projected = predicted.copy()
        if native["velocity_iterations"] and native["mode"] in ("mimic", "both"):
            # One hard velocity pass of the same row after converged positions.
            j = J[-1]
            r = response[:, -1]
            projected -= r * (j @ projected) / (j @ r)
        actual = np.array(measured["v_final"])
        result = {
            "mode": native["mode"],
            "gains": native["point_gains"],
            "velocity_iterations": native["velocity_iterations"],
            "tier": row["tier"],
            "name": row["name"],
            "native_model_final_error": float(abs(projected - actual).max()),
            "native_model_motion_error": float(
                abs(predicted - measured["v_motion"]).max()
            ),
            "native_model_source_pre_error": float(
                abs(predicted - row["v_pre_implicit"]).max()
            ),
            "constraint_delta_reconstruction_error": float(
                abs(
                    response @ np.linalg.lstsq(response, actual - vf, rcond=1e-10)[0]
                    - (actual - vf)
                ).max()
            ),
            "maximum_inverse_response": float(abs(J @ response).max()),
            "regularizer": R.tolist(),
            "predicted_position_velocity": predicted.tolist(),
            "predicted_final_velocity": projected.tolist(),
            "predicted_impulse": impulse.tolist(),
        }
        results.append(result)
report = {
    "scope": "source M14 and independent URDF native D6 Jacobian; native free-step response; explicit position spring rows plus independently predicted articulated COM/angular bias velocity and one hard mimic velocity projection",
    "rows": results,
}
(OUT / "native-analytic-spatial-bias.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
for key in sorted({(r["mode"], r["gains"], r["velocity_iterations"]) for r in results}):
    group = [
        r for r in results if (r["mode"], r["gains"], r["velocity_iterations"]) == key
    ]
    print(
        key,
        {
            metric: max(r[metric] for r in group)
            for metric in (
                "native_model_final_error",
                "native_model_motion_error",
                "native_model_source_pre_error",
                "constraint_delta_reconstruction_error",
            )
        },
        flush=True,
    )
