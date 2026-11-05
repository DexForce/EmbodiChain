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

"""Check references and acceptance calculations independently of Isaac Lab."""

from __future__ import annotations

import os

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from scripts.benchmark.tech_reports.reporting.p01_analysis import (
    measure,
    periods,
    reference,
)

CFG = json.loads(
    (
        Path(__file__).resolve().parents[4]
        / "scripts/benchmark/tech_reports/physics/experiments/p01_configs/isaaclab.json"
    ).read_text()
)


class AnalysisTests(unittest.TestCase):
    def test_freefall_reference_and_threshold(self) -> None:
        t = np.arange(1001) * 0.01
        ref = np.stack((1.0 - 9.81 * t * t / 2, -9.81 * t), axis=-1)
        exact = measure("freefall", t, ref, ref, CFG)
        self.assertTrue(exact["candidate_pass"])
        self.assertEqual(exact["position_max_error"], 0.0)
        perturbed = ref.copy()
        perturbed[100, 0] += 0.006
        failed = measure("freefall", t, perturbed, ref, CFG)
        self.assertFalse(failed["candidate_pass"])
        self.assertAlmostEqual(failed["first_pointwise_threshold_failure_s"], 1.0)

    def test_rk4_against_dop853_and_exact_period(self) -> None:
        a = 9.81 * 0.5 / (0.5**2 + 0.4 * 0.025**2)
        theta0 = 0.18
        t = np.arange(4001) * 0.0025
        actual = reference(np.array([theta0]), 10.0, 0.0001, 0.0025, a)[:, 0]
        check = solve_ivp(
            lambda t, y: [y[1], -a * np.sin(y[0])],
            [0, 10],
            [theta0, 0.0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            t_eval=t,
        )
        np.testing.assert_allclose(actual, check.y.T, rtol=0, atol=1e-9)
        exact_period = 4 / np.sqrt(a) * ellipk(np.sin(theta0 / 2) ** 2)
        self.assertLess(abs(periods(t, actual[:, 0]).mean() / exact_period - 1), 1e-6)
        self.assertTrue(measure("pendulum", t, actual, actual, CFG)["candidate_pass"])

    def test_energy_failure_survives_recovery(self) -> None:
        t = np.arange(1001) * 0.01
        state = np.stack(
            (0.18 * np.cos(4.42 * t), -0.18 * 4.42 * np.sin(4.42 * t)), axis=-1
        )
        modified = state.copy()
        modified[100, 1] += 1.0
        result = measure("pendulum", t, modified, state, CFG)
        self.assertFalse(result["candidate_pass"])
        self.assertGreater(result["energy_max_relative_drift"], 0.02)
        self.assertAlmostEqual(result["first_pointwise_threshold_failure_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
