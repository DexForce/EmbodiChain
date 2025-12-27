# Workspace Analyzer Samplers

The samplers module provides various sampling strategies for workspace analysis, from uniform grids to quasi-random sequences.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Available Samplers](#available-samplers)
  - [1. UniformSampler](#1-uniformsampler)
  - [2. RandomSampler](#2-randomsampler)
  - [3. GaussianSampler](#3-gaussiansampler-)
- [Factory Pattern Usage](#factory-pattern-usage)
- [Integration with Workspace Analyzer](#integration-with-workspace-analyzer)
- [Quick Reference](#quick-reference)

## Quick Start

```python
import torch
from embodichain.lab.devices.workspace_analyzer.samplers import UniformSampler

# Define sampling bounds
bounds = torch.tensor([[0.0, 1.0],    # x-axis range
                       [0.0, 1.0]])   # y-axis range

# Create uniform sampler and generate 1000 samples
sampler = UniformSampler(samples_per_dim=10)
samples = sampler.sample(1000, bounds)  # Note: (num_samples, bounds) order
```

## Overview

Available sampling strategies:

- **UniformSampler**: Grid-based regular sampling ✅
- **RandomSampler**: Pure random sampling ✅
- **GaussianSampler**: Normal distribution sampling ✅

## Basic Usage

```python
import torch
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
    UniformSampler, RandomSampler, GaussianSampler
)

# Define bounds: [[min, max], [min, max], ...]
bounds = torch.tensor([[-1, 1], [-1, 1], [0, 2]], dtype=torch.float32)

# Generate samples with implemented samplers
samples = UniformSampler(samples_per_dim=10).sample(1000, bounds)
print(f"Generated {samples.shape[0]} samples")
```

## Available Samplers

### 1. UniformSampler

Grid-based sampling with regular spacing.

```python
import torch
from embodichain.lab.sim.utility.workspace_analyzer.samplers import UniformSampler

bounds = torch.tensor([[-1, 1], [-1, 1], [0, 2]], dtype=torch.float32)
# Create uniform sampler
uniform_sampler = UniformSampler(seed=42, samples_per_dim=10)
samples = uniform_sampler.sample(1000, bounds)
```

**Best for**: Low-dimensional spaces (2-4D), systematic exploration

### 2. RandomSampler

Pure random sampling with uniform distribution.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import RandomSampler

random_sampler = RandomSampler(seed=42)
samples = random_sampler.sample(1000, bounds)
```

**Best for**: High-dimensional spaces, baseline comparisons

### 3. GaussianSampler ✅

Gaussian distribution sampling around specified mean.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import GaussianSampler

gaussian_sampler = GaussianSampler(seed=42, std=0.2)
samples = gaussian_sampler.sample(1000, bounds)
```

**Best for**: Uncertainty analysis, robustness studies around workspace center

## Factory Pattern Usage

Use the factory to create samplers by strategy (only implemented samplers):

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import create_sampler
from embodichain.lab.sim.utility.workspace_analyzer.configs import SamplingStrategy

# Create implemented samplers by strategy
uniform_sampler = create_sampler(
    SamplingStrategy.UNIFORM, 
    seed=42, 
    samples_per_dim=10
)

random_sampler = create_sampler(
    SamplingStrategy.RANDOM,
    seed=42
)

gaussian_sampler = create_sampler(
    SamplingStrategy.GAUSSIAN,
    seed=42,
    std=0.2
)
```

## Integration with Workspace Analyzer

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import SamplingConfig, SamplingStrategy

# Configure sampling in analyzer
sampling_config = SamplingConfig(
    strategy=SamplingStrategy.UNIFORM,
    num_samples=1000,
    seed=42
)

# Analyzer will use the specified sampler automatically
```

## Quick Reference

**Available Samplers**:

- **UniformSampler**: Use for 2-4D systematic exploration
- **RandomSampler**: Baseline for any dimensional spaces  
- **GaussianSampler**: Uncertainty and robustness analysis

**Sample Size Guidelines**:

- **Uniform**: `samples_per_dim^n_dims` (exponential growth)
- **Random**: Any size (flexible)
- **Gaussian**: Any size (flexible)
