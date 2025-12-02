# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from embodichain.lab.sim.utility.workspace_analyzer.samplers.base_sampler import (
    BaseSampler,
    ISampler,
)

from embodichain.lab.sim.utility.workspace_analyzer.samplers.iniform_sampler import (
    UniformSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.random_sampler import (
    RandomSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.halton_sampler import (
    HaltonSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.sobol_sampler import (
    SobolSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers.lhs_sampler import (
    LatinHypercubeSampler,
)

from embodichain.lab.sim.utility.workspace_analyzer.samplers.sampler_factory import (
    SamplerFactory,
    create_sampler,
)

# Note: Geometric constraints are temporarily disabled

# Note: PlaneSampler is intentionally not included here as it's a high-level
# specialized sampler, not a base sampler. Import it directly when needed:
# from embodichain.lab.sim.utility.workspace_analyzer.samplers.plane_sampler import PlaneSampler

__all__ = [
    # Base samplers and interfaces
    "BaseSampler",
    "ISampler",
    # Concrete base samplers
    "UniformSampler",
    "RandomSampler",
    "HaltonSampler",
    "SobolSampler",
    "LatinHypercubeSampler",
    # Note: Geometric constraints temporarily disabled
    # Factory for creating base samplers
    "SamplerFactory",
    "create_sampler",
    # Note: Constraint-based convenience functions temporarily disabled
]
