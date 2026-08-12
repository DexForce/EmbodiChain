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


def test_random_ik_seeds_are_independent_across_batch_rows() -> None:
    torch.manual_seed(7)
    sampler = QposSeedSampler(num_samples=4, dof=3, device=torch.device("cpu"))

    sampled = sampler.sample(
        qpos_seed=torch.zeros(2, 3),
        lower_limits=-torch.ones(3),
        upper_limits=torch.ones(3),
        batch_size=2,
    ).reshape(2, 4, 3)

    assert torch.equal(sampled[:, 0], torch.zeros(2, 3))
    assert not torch.equal(sampled[0, 1:], sampled[1, 1:])
