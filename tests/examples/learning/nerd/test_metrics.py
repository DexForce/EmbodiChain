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

"""Independent geometry and simulator-free contracts for rollout metrics."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import torch

from examples.learning.nerd._support._evaluation import (
    _error_values,
    _safe_number,
    _teacher_row_sha256,
)


def test_metrics_import_without_any_simulation_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from examples.learning.nerd._support._evaluation import _error_values; assert not any(k == 'newton' or k == 'dexsim' or k.startswith('embodichain.lab.sim') for k in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_continuous_angle_and_bounded_joint_errors_have_distinct_meaning():
    cart = torch.tensor([[0.0, 2 * math.pi]], dtype=torch.float64)
    error = _error_values(
        "cartpole",
        cart,
        torch.zeros_like(cart),
        torch.zeros_like(cart),
        torch.zeros_like(cart),
    )
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-12, rtol=0)
    q = torch.zeros(1, 15, dtype=torch.float64)
    q[:, 6] = 1
    target = q.clone()
    q[:, 7] = 2 * math.pi
    v = torch.zeros(1, 14, dtype=torch.float64)
    error = _error_values("ant", q, v, target, v)
    assert error[0, 0].item() == math.pi / 4


def test_orientation_metric_is_geodesic_and_quaternion_sign_invariant():
    q = torch.zeros(2, 15, dtype=torch.float64)
    q[:, 6] = 1
    target = q.clone()
    q[0, 5:7] = math.sqrt(0.5)
    q[1, 6] = -1
    v = torch.zeros(2, 14, dtype=torch.float64)
    error = _error_values("ant", q, v, target, v)
    torch.testing.assert_close(
        error[:, 3],
        torch.tensor([math.pi / 2, 0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert _safe_number(float("nan")) is None
    assert _safe_number(float("inf")) is None
    assert _safe_number(0.0) == 0.0


def test_teacher_content_hash_changes_with_values_and_identity():
    row = {"id": "seed:1", "metadata": {"sampling_seed": 12}}
    values = [
        np.zeros((3, 2), np.float32),
        np.zeros((3, 2), np.float32),
        np.zeros((2, 2), np.float32),
    ]
    expected = _teacher_row_sha256(row, values)
    assert expected == _teacher_row_sha256(
        {**row, "file": "different.h5"}, [v.copy() for v in values]
    )
    assert expected != _teacher_row_sha256({**row, "id": "seed:2"}, values)
    values[-1][0, 0] = 1
    assert expected != _teacher_row_sha256(row, values)
