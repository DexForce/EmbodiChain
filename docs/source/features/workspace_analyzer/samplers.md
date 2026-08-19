# Workspace Analyzer Samplers

Workspace analysis supports regular grids, pseudo-random sampling,
low-discrepancy sequences, stratified sampling, and targeted distributions.
All samplers return a tensor with shape `(num_samples, dimensions)` (a uniform
grid may return a nearby grid-sized count).

## Quick start

```python
import torch

from embodichain.lab.sim.workspace.samplers import UniformSampler

bounds = torch.tensor(
    [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0.0, 2.0],
    ],
    dtype=torch.float32,
)

sampler = UniformSampler(samples_per_dim=10, seed=42)
samples = sampler.sample(num_samples=1000, bounds=bounds)
print(samples.shape)
```

Use the keyword arguments `num_samples` and `bounds` as shown. This form works
consistently across all built-in samplers.

## Available strategies

| Strategy | Class | Best suited to | Notes |
|----------|-------|----------------|-------|
| `uniform` | `UniformSampler` | Systematic low-dimensional coverage | Creates a regular grid; count is `samples_per_dim ** dimensions` when explicitly set. |
| `random` | `RandomSampler` | Fast baselines and high-dimensional spaces | Independent uniform samples inside each bound. |
| `halton` | `HaltonSampler` | Low- to medium-dimensional quasi-Monte Carlo | Deterministic low-discrepancy sequence with an optional initial skip. |
| `sobol` | `SobolSampler` | Higher-dimensional quasi-Monte Carlo | Uses SciPy when available and otherwise falls back to the built-in implementation. |
| `lhs` | `LatinHypercubeSampler` | Experimental design and sensitivity analysis | SciPy enables optimized Latin-hypercube layouts. |
| `gaussian` | `GaussianSampler` | Local exploration around a target | Available through the factory or its concrete submodule; clips to bounds by default. |
| `importance` | `ImportanceSampler` | Concentrating samples in task-relevant regions | Requires a `weight_fn` when constructed. |

`SamplingStrategy.SPHERE` is currently a compatibility alias for uniform
sampling; it does not apply a spherical geometric constraint.

## Factory usage

Use `create_sampler` when the strategy comes from configuration:

```python
from embodichain.lab.sim.workspace.configs import SamplingStrategy
from embodichain.lab.sim.workspace.samplers import create_sampler

sampler = create_sampler(
    SamplingStrategy.SOBOL,
    seed=42,
    scramble=True,
)
samples = sampler.sample(num_samples=1000, bounds=bounds)
```

The factory accepts either a `SamplingStrategy` value or its string value:

```python
random_sampler = create_sampler("random", seed=42)
halton_sampler = create_sampler("halton", seed=42, skip=100)
gaussian_sampler = create_sampler("gaussian", seed=42, std=0.2)
```

Importance sampling additionally needs a non-negative weighting function:

```python
def center_weight(points: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.linalg.vector_norm(points, dim=1))


importance_sampler = create_sampler(
    SamplingStrategy.IMPORTANCE,
    seed=42,
    weight_fn=center_weight,
)
samples = importance_sampler.sample(num_samples=1000, bounds=bounds)
```

## Workspace Analyzer integration

Select a strategy through `SamplingConfig`; `WorkspaceAnalyzer` creates the
matching sampler and uses it for joint- and Cartesian-space sampling:

```python
from embodichain.lab.sim.workspace.configs import (
    SamplingConfig,
    SamplingStrategy,
)

sampling = SamplingConfig(
    strategy=SamplingStrategy.HALTON,
    num_samples=1000,
    seed=42,
)
```

For importance sampling, construct the sampler directly with its `weight_fn`;
`SamplingConfig` does not currently forward that callable to the factory.

## Choosing a sampler

- Start with `random` for a fast baseline.
- Use `uniform` when complete grid coverage is practical.
- Prefer `halton`, `sobol`, or `lhs` when coverage quality matters more than
  strict randomness.
- Use `gaussian` or `importance` only when you intentionally want a biased
  sampling distribution.
