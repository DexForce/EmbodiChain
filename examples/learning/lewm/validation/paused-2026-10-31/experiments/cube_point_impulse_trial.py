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

"""Private point-impulse trial. The accepted task/controller remain unchanged.

URDF bias acceleration was independently checked against all 50 official source
queries. D6 still uses a different actor-A moment arm at nonzero separation, so
this adapter is a diagnostic and cannot establish exact equality parity.
"""

from __future__ import annotations

import numpy as np
import torch

from cube_point_kinematics import PointKinematics
from embodichain_tasks.manipulation.ogbench_cube import OGBenchCubeEnv


class PointImpulseCube(OGBenchCubeEnv):
    def _setup_robot(self, **kwargs):
        result = super()._setup_robot(**kwargs)
        self._point_model = PointKinematics(self.root)
        self.point_trial = {
            "steps": 0,
            "max_residual_m": 0.0,
            "max_Jdotv_m_s2": 0.0,
            "max_target_m_s": 0.0,
        }
        return result

    def _point_update(self, qpos, qvel):
        eq = self.contract["equality"]
        rows = [self._point_model.evaluate(q, v) for q, v in zip(qpos, qvel)]
        for column, (index, _) in enumerate(self._point_model.points):
            tau, ratio = eq["solref"][index][:2]
            d0, dmax, width, midpoint, power = eq["solimp"][index]
            K, B = 1 / (dmax * tau * ratio) ** 2, 2 / (dmax * tau)
            A0 = sum(
                self.contract["body_invweight0"][eq[key][index]][0]
                for key in ("obj1id", "obj2id")
            )
            springs, dampers, targets = [], [], []
            for env in rows:
                point = env[column]
                residual = np.linalg.norm(point["residual"])
                x = min(residual / width, 1.0)
                y = (
                    x**power / midpoint ** (power - 1)
                    if x <= midpoint
                    else 1 - (1 - x) ** power / (1 - midpoint) ** (power - 1)
                )
                d = d0 + y * (dmax - d0)
                R = (1 - d) / d * A0
                # Correct the native velocity Jacobian, while retaining the
                # native force moment arm. This does not compensate the latter.
                native_velocity = point["velocity"] - np.cross(
                    point["omega_a"], point["residual"]
                )
                target = (
                    point["rotation_a"].T
                    @ (
                        native_velocity
                        - 0.002 * B * point["velocity"]
                        - 0.002 * point["Jdotv"]
                    )
                    / (1 - 0.002**2 * K * d)
                )
                springs.append(K * d / R)
                dampers.append((1 / 0.002 - 0.002 * K * d) / R)
                targets.append(target)
                self.point_trial["max_residual_m"] = max(
                    self.point_trial["max_residual_m"], float(residual)
                )
                self.point_trial["max_Jdotv_m_s2"] = max(
                    self.point_trial["max_Jdotv_m_s2"],
                    float(np.linalg.norm(point["Jdotv"])),
                )
                self.point_trial["max_target_m_s"] = max(
                    self.point_trial["max_target_m_s"], float(np.linalg.norm(target))
                )
            self.sim.update_rigid_constraint_linear_drive(
                f"ogbench_connect_{index}", springs, dampers, targets
            )
        self.point_trial["steps"] += 1

    def _advance_physics(self):
        for _ in range(25):
            qpos = self.robot.get_qpos()[:, self._joint_ids].cpu().numpy()
            qvel = self.robot.get_qvel()[:, self._joint_ids].cpu().numpy()
            self._point_update(qpos, qvel)
            force = self._passive_and_tendon_forces(qpos, qvel)
            self.robot.set_qf(
                torch.as_tensor(force, dtype=torch.float32, device=self.device),
                joint_ids=self._joint_ids,
            )
            self.sim.update(0.002, step=1, render_final_step=False)
        if self.cfg.enable_source_state:
            self._state_wrench_valid.fill_(True)
            self._has_source_dynamics = True
