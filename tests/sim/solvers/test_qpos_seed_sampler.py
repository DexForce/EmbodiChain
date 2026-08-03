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

import torch

from embodichain.lab.sim.solvers.qpos_seed_sampler import QposSeedSampler


def _sample(batch_size: int = 3) -> torch.Tensor:
    sampler = QposSeedSampler(
        num_samples=5,
        dof=2,
        device=torch.device("cpu"),
    )
    qpos_seed = torch.zeros(batch_size, 2)
    lower = torch.full((2,), -1.0)
    upper = torch.full((2,), 1.0)
    return sampler.sample(qpos_seed, lower, upper, batch_size).reshape(
        batch_size,
        5,
        2,
    )


def test_random_restarts_are_independent_between_targets() -> None:
    torch.manual_seed(7)

    samples = _sample()

    torch.testing.assert_close(samples[:, 0], torch.zeros(3, 2))
    assert not torch.equal(samples[0, 1:], samples[1, 1:])
    assert not torch.equal(samples[1, 1:], samples[2, 1:])


def test_random_restarts_follow_the_torch_seed() -> None:
    torch.manual_seed(11)
    first = _sample()
    torch.manual_seed(11)
    second = _sample()

    torch.testing.assert_close(first, second)
