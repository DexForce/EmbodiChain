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

"""Validate independent world COM velocities and bias accelerations in MuJoCo."""

from __future__ import annotations
import json
from pathlib import Path
import h5py
import mujoco
import numpy as np
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from real14_link_bias import LinkBiasKinematics

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
model = LinkBiasKinematics()
with h5py.File(ROOT / "alignment/official-queries.h5") as file:
    q, v = file["qpos"][:], file["qvel"][:]
env = CubeEnv(
    env_type="single", ob_type="states", multiview=False, visualize_info=False
)
rows = []
try:
    env.reset(seed=42)
    m, d = env.model, env.data
    m.opt.gravity[:] = 0
    for key in (
        "CONTACT",
        "LIMIT",
        "FRICTIONLOSS",
        "ACTUATION",
        "SPRING",
        "DAMPER",
        "WARMSTART",
    ):
        m.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{key}"))
    for i in range(50):
        env.set_state(q[i], v[i])
        d.eq_active[:] = False
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        for name, (center, lv, av, la, aa) in model.com_states(
            q[i, :14], v[i, :14]
        ).items():
            index = m.body(name).id
            velocity, acceleration = np.empty(6), np.empty(6)
            jp, jr = np.empty((3, m.nv)), np.empty((3, m.nv))
            mujoco.mj_jacBodyCom(m, d, jp, jr, index)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, index, velocity, 0)
            mujoco.mj_objectAcceleration(
                m, d, mujoco.mjtObj.mjOBJ_BODY, index, acceleration, 0
            )
            bias = acceleration - np.r_[jr @ d.qacc, jp @ d.qacc]
            rows.append(
                {
                    "query": i,
                    "body": name,
                    "position_error": float(abs(center - d.xipos[index]).max()),
                    "velocity_error": float(abs(np.r_[av, lv] - velocity).max()),
                    "bias_error": float(abs(np.r_[aa, la] - bias).max()),
                }
            )
    report = {
        "scope": "50 official states x 21 positive bodies; world COM velocity and acceleration bias; independent URDF recursion versus MuJoCo RNE minus body Jacobian*qacc",
        "rows": rows,
        "maximum": {
            key: max(r[key] for r in rows)
            for key in ("position_error", "velocity_error", "bias_error")
        },
    }
    (ROOT / "alignment/real14-coupled-impulse/link-bias-source.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report["maximum"], flush=True)
    assert report["maximum"]["position_error"] < 1e-10
    assert report["maximum"]["velocity_error"] < 1e-10
    assert report["maximum"]["bias_error"] < 1e-9
finally:
    env.close()
