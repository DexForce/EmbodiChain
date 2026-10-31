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


from __future__ import annotations
from pathlib import Path
import json, numpy as np
from scipy.spatial.transform import Rotation

p = Path(
    "/root/sources/EmbodiChain/outputs/lewm/official-cube/alignment/point-step-response"
)
ref = json.loads((p / "rotating-jdot-source.json").read_text())
rows = []
h = 0.002


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


for case in ref["rows"]:
    pose = np.array(case["pose"])
    v = np.array(case["velocity_world"]).ravel()
    rot = [Rotation.from_quat(x[3:]).as_matrix() for x in pose]
    anchor = [rot[i] @ case["anchors"][i] for i in range(2)]
    error = pose[1, :3] + anchor[1] - pose[0, :3] - anchor[0]
    J = np.hstack([-np.eye(3), skew(anchor[0]), np.eye(3), -skew(anchor[1])])
    N = np.hstack([-np.eye(3), skew(anchor[0] + error), np.eye(3), -skew(anchor[1])])
    M = np.diag([1] * 3 + [10] * 3 + [0.5] * 3 + [5] * 3)
    # Obtain source K,d,R independently by reusing the source stiffness and
    # implicit damping: h*(Dn+h*Kn)=1/R.
    Kn, Dn = case["gain"]
    R = 1 / (h * (Dn + h * Kn))
    Kd = Kn * R
    vt = rot[0] @ case["target"]
    impulse = np.linalg.solve(
        R * np.eye(3) + N @ M @ N.T, (1 - h * h * Kd) * vt - N @ v - h * Kd * error
    )
    predicted = (v + M @ N.T @ impulse).reshape(2, 6)
    # Source sign is A minus B; the analytic exported drift shares that sign.
    sd = -np.array(case["analytic_Jdotv"])
    # Recover B from the source target relation where needed without dividing
    # velocity components: source solref=.005, dmax=.99.
    B = 2 / (0.99 * 0.005)
    src_imp = np.linalg.solve(
        R * np.eye(3) + J @ M @ J.T, -h * B * (J @ v) - h * Kd * error - h * sd
    )
    source_pred = (v + M @ J.T @ src_imp).reshape(2, 6)
    actual = np.array(
        json.loads((p / f'rotating-jdot-native-case-{case["id"]}.json').read_text())[
            "rows"
        ][0]["velocity"]
    )
    source = np.array(case["after_velocity_world"])
    row = {
        "id": case["id"],
        "source_analytic_velocity_error": float(abs(source_pred - source).max()),
        "native_analytic_velocity_error": float(abs(predicted - actual).max()),
        "native_source_velocity_error": float(abs(actual - source).max()),
        "analytic_native_source_velocity_error": float(
            abs(predicted - source_pred).max()
        ),
        "source_angular0_delta": (source[0, 3:] - v.reshape(2, 6)[0, 3:]).tolist(),
        "native_angular0_delta": (actual[0, 3:] - v.reshape(2, 6)[0, 3:]).tolist(),
    }
    rows.append(row)
    print(row)
(p / "rotating-analytic.json").write_text(
    json.dumps(
        {
            "scope": "independent 12D inverse-mass impulse solve; native uses anchor B for actor A moment arm; source uses anchor A",
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
