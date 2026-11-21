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

"""Private coupled stage trial; accepted production task is not replaced.

Combines independently qualified source force/integration stages, point-anchor
Jacobians, articulated bias velocity and the compliant mimic target term.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from load_source_servo_archive import install_archived_task

archived = install_archived_task()
from embodichain.compute.dynamics._inertia import _implicit_velocity_step
from embodichain.lab.sim import SimulationManager
from real14_link_bias import LinkBiasKinematics

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")


def impedance(residual, parameters):
    d0, dmax, width, midpoint, power = parameters
    x = min(abs(residual) / width, 1.0)
    y = (
        x**power / midpoint ** (power - 1)
        if x <= midpoint
        else 1 - (1 - x) ** power / (1 - midpoint) ** (power - 1)
    )
    return d0 + (dmax - d0) * y


class CoupledSourceCube(archived.OGBenchCubeEnv):
    def _declare_robot(self, **kwargs):
        original = SimulationManager.add_robot

        def declare(sim, cfg):
            cfg.root_props.min_velocity_iters = 0
            return original(sim, cfg)

        with patch.object(SimulationManager, "add_robot", declare):
            return super()._declare_robot(**kwargs)

    def _prepare_scene(self, **kwargs):
        original = SimulationManager.add_rigid_object

        def declare(sim, cfg):
            if cfg.attrs.rigid_props is not None:
                cfg.attrs.rigid_props.min_velocity_iters = 0
            return original(sim, cfg)

        with patch.object(SimulationManager, "add_rigid_object", declare):
            return super()._prepare_scene(**kwargs)

    def _setup_robot(self, **kwargs):
        original = self.sim.create_rigid_constraint

        def create(cfg, *args, **extra):
            cfg.linear_drive_point_anchors = True
            return original(cfg, *args, **extra)

        with patch.object(self.sim, "create_rigid_constraint", create):
            result = super()._setup_robot(**kwargs)
        self._point_model = LinkBiasKinematics(self.root)
        source = json.loads(
            (ROOT / "alignment/real14-coupled-impulse/source.json").read_text()
        )
        row = next(r for r in source["rows"] if r["mode"] == "mimic")
        d = row["KBIP"][0][2]
        self._mimic_inverse_weight = row["R"][0] * d / (1 - d)
        self._mimic_ids = [
            self._names.index(f"ur5e/robotiq/{side}_driver_joint")
            for side in ("right", "left")
        ]
        self._mimic_jacobian = np.zeros(14)
        self._mimic_jacobian[self._mimic_ids] = [1, -1]
        self.coupled_trial = {
            "substeps": 0,
            "max_point_residual_m": 0.0,
            "max_mimic_velocity_force_Nm": 0.0,
            "max_native_motion_final_difference_rad_s": 0.0,
        }
        return result

    def _update_equalities(self, q, v, mass):
        eq = self.contract["equality"]
        states = [self._point_model.evaluate(qp, qv) for qp, qv in zip(q, v)]
        biases = [
            self._point_model.point_solver_bias(qp, qv, common_point=False)
            for qp, qv in zip(q, v)
        ]
        for column, (index, _) in enumerate(self._point_model.points):
            tau, ratio = eq["solref"][index]
            K, B = 1 / (eq["solimp"][index][1] * tau * ratio) ** 2, 2 / (
                eq["solimp"][index][1] * tau
            )
            A0 = sum(
                self.contract["body_invweight0"][eq[key][index]][0]
                for key in ("obj1id", "obj2id")
            )
            spring, damping, target = [], [], []
            for row, bias in zip(states, biases):
                point = row[column]
                length = float(np.linalg.norm(point["residual"]))
                d = impedance(length, eq["solimp"][index])
                R = (1 - d) / d * A0
                spring.append(K * d / R)
                damping.append((500 - 0.002 * K * d) / R)
                target.append(
                    point["rotation_a"].T
                    @ (
                        (1 - 0.002 * B) * point["velocity"]
                        - 0.002 * point["Jdotv"]
                        + 0.002 * bias[column]
                    )
                    / (1 - 0.002**2 * K * d)
                )
                self.coupled_trial["max_point_residual_m"] = max(
                    self.coupled_trial["max_point_residual_m"], length
                )
            self.sim.update_rigid_constraint_linear_drive(
                f"ogbench_connect_{index}", spring, damping, target
            )
        params = eq["solimp"][2]
        tau, ratio = eq["solref"][2]
        K, B = 1 / (params[1] * tau * ratio) ** 2, 2 / (params[1] * tau)
        jac = self._mimic_jacobian
        extra = np.zeros_like(q)
        for i in range(len(q)):
            residual = float(jac @ q[i])
            speed = float(jac @ v[i])
            d = impedance(residual, params)
            R = (1 - d) / d * self._mimic_inverse_weight
            response = jac @ np.linalg.solve(mass[i], jac)
            omega = np.sqrt(K * d * response / R)
            damping = (500 - 0.002 * K * d) * response / (2 * omega * R)
            self.robot.set_mimic_compliance(float(omega), float(damping), env_ids=[i])
            sigma = (1 - 0.002 * B) * speed / (0.002 * R)
            extra[i] = jac * sigma
            self.coupled_trial["max_mimic_velocity_force_Nm"] = max(
                self.coupled_trial["max_mimic_velocity_force_Nm"], abs(float(sigma))
            )
        return extra

    @torch.no_grad()
    def _advance_physics(self):
        control = torch.as_tensor(
            self._control, dtype=torch.float64, device=self.device
        )
        for substep in range(25):
            q = self.robot.get_qpos()[:, self._joint_ids].to(torch.float64)
            v = self.robot.get_qvel()[:, self._joint_ids].to(torch.float64)
            pose = self.robot.body_data.body_link_pose
            mass = self._source_dynamics.mass_matrix(pose)
            force, derivative = self._source_dynamics.forces_and_derivative(
                q, v, control
            )
            extra = self._update_equalities(
                q.cpu().numpy(), v.cpu().numpy(), mass.cpu().numpy()
            )
            applied = force + torch.as_tensor(
                extra, dtype=force.dtype, device=self.device
            )
            if substep == 24:
                self._source_control_pose = pose.clone()
            self.robot.set_qf(applied.to(torch.float32), joint_ids=self._joint_ids)
            self.sim.update(0.002, step=1, render_final_step=False)
            physical_q = self.robot.get_qpos()[:, self._joint_ids].to(torch.float64)
            physical_v = self.robot.get_qvel()[:, self._joint_ids].to(torch.float64)
            difference = float(((physical_q - q) / 0.002 - physical_v).abs().max())
            self.coupled_trial["max_native_motion_final_difference_rad_s"] = max(
                self.coupled_trial["max_native_motion_final_difference_rad_s"],
                difference,
            )
            if substep == 24 and self.cfg.enable_source_state:
                self._capture_source_wrench(self._source_control_pose)
            # The added equality target force is a constraint contribution and
            # is deliberately excluded from the source smooth-force derivative.
            position, velocity = _implicit_velocity_step(
                q, v, physical_v, mass, derivative, 0.002
            )
            self.robot.set_state(
                qpos=position,
                qvel=velocity,
                joint_ids=self._joint_ids,
                clear_dynamics=False,
                clamp_qpos=False,
            )
            self.coupled_trial["substeps"] += 1
        self._source_step_valid.fill_(True)
        if self.cfg.enable_source_state:
            self._state_wrench_valid.fill_(True)
            self._has_source_dynamics = True
