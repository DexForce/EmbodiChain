# Workspace Analyzer Samplers

The **Workspace Analyzer Samplers** module provides a comprehensive collection of sampling strategies for robotic workspace analysis. This module implements various statistical and quasi-random sampling methods to efficiently explore the robot's configuration and Cartesian spaces.

## Overview

Sampling is a fundamental component of workspace analysis, as it determines how the analysis algorithm explores the robot's workspace. Different sampling strategies offer trade-offs between coverage quality, computational efficiency, and statistical properties. The samplers module provides a unified interface through the factory pattern, making it easy to switch between different sampling methods.

## Core Architecture

### ISampler Interface

All samplers implement the `ISampler` protocol, which defines the contract for sampling operations:

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import ISampler

class ISampler(Protocol):
    def sample(
        self, bounds: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        """Generate samples within the given bounds."""
        ...
        
    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy."""
        ...
```

### BaseSampler Abstract Class

The `BaseSampler` class provides common functionality for all concrete samplers:

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import BaseSampler

class BaseSampler(ABC):
    def __init__(self, seed: int = 42, device: Optional[torch.device] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.device = device or torch.device("cpu")
```

## Available Samplers

### 1. UniformSampler

**Grid-based uniform sampling** that generates samples on a regular grid within specified bounds.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import UniformSampler

# Create uniform sampler with fixed samples per dimension
uniform_sampler = UniformSampler(seed=42, samples_per_dim=10)

# Generate samples
bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32)
samples = uniform_sampler.sample(bounds, num_samples=100)
print(f"Generated {samples.shape[0]} samples")  # Actual: 10^2 = 100 samples
```

**Characteristics:**

- **Advantages:** Perfect coverage, deterministic, good for low-dimensional spaces
- **Disadvantages:** Exponential growth with dimensions (curse of dimensionality)
- **Best for:** 2-4 dimensional analysis, boundary detection, systematic exploration

**Use Cases:**

```python
# Perfect for 2D workspace visualization
sampler_2d = UniformSampler(samples_per_dim=50)
bounds_2d = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
grid_samples = sampler_2d.sample(bounds_2d, num_samples=2500)

# Systematic joint space exploration
joint_sampler = UniformSampler(samples_per_dim=5)
joint_bounds = torch.tensor([[-np.pi, np.pi]] * 7)  # 7-DOF robot
joint_samples = joint_sampler.sample(joint_bounds, num_samples=78125)  # 5^7
```

### 2. RandomSampler

**Pure random sampling** using uniform random distribution within bounds.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import RandomSampler

# Create random sampler
random_sampler = RandomSampler(seed=42)

# Generate random samples
bounds = torch.tensor([[-2, 2], [-2, 2], [0, 3]], dtype=torch.float32)
samples = random_sampler.sample(bounds, num_samples=1000)
```

**Characteristics:**

- **Advantages:** Simple, works in any dimension, no memory requirements
- **Disadvantages:** Poor coverage, clustering, slow convergence
- **Best for:** High-dimensional spaces, baseline comparisons, Monte Carlo methods

**Use Cases:**

```python
# High-dimensional joint space sampling
high_dim_sampler = RandomSampler(seed=123)
bounds_14d = torch.tensor([[-np.pi, np.pi]] * 14)  # Dual-arm robot
samples_14d = high_dim_sampler.sample(bounds_14d, num_samples=10000)

# Quick exploration with minimal setup
quick_sampler = RandomSampler()
workspace_samples = quick_sampler.sample(workspace_bounds, num_samples=500)
```

### 3. HaltonSampler

**Quasi-random low-discrepancy sequence** sampler using Halton sequences with prime bases.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import HaltonSampler

# Create Halton sampler with default settings
halton_sampler = HaltonSampler(seed=42)

# Create with custom bases and skip initial samples
halton_custom = HaltonSampler(
    seed=42,
    bases=[2, 3, 5, 7, 11, 13],  # Prime bases for each dimension
    skip=100  # Skip first 100 samples to reduce correlation
)

# Generate quasi-random samples
bounds = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.float32)
samples = halton_sampler.sample(bounds, num_samples=1000)
```

**Characteristics:**

- **Advantages:** Better uniformity than random, deterministic, fast convergence
- **Disadvantages:** Degrades in high dimensions (>10), potential correlation
- **Best for:** 2-10 dimensional spaces, Monte Carlo integration, parameter studies

**Use Cases:**

```python
# Cartesian workspace analysis
cartesian_sampler = HaltonSampler(skip=50)
cartesian_bounds = torch.tensor([
    [-0.8, 0.8],  # X range
    [-0.8, 0.8],  # Y range  
    [0.1, 1.2]    # Z range
])
cartesian_samples = cartesian_sampler.sample(cartesian_bounds, num_samples=2000)

# Parameter sensitivity analysis
param_sampler = HaltonSampler(bases=[2, 3, 5, 7])
param_bounds = torch.tensor([
    [0.1, 0.3],    # Link length variation
    [0.8, 1.2],    # Mass scaling
    [0.5, 2.0],    # Damping coefficient
    [0.1, 0.5]     # Friction coefficient
])
param_samples = param_sampler.sample(param_bounds, num_samples=1500)
```

### 4. SobolSampler

**Sobol sequence sampler** using scrambled Sobol quasi-random sequences.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import SobolSampler

# Create Sobol sampler with scrambling for better uniformity
sobol_sampler = SobolSampler(
    seed=42,
    scramble=True,  # Improves uniformity
    optimization=None  # No additional optimization
)

# Generate Sobol samples
bounds = torch.tensor([[-2, 2], [-2, 2], [-1, 1]], dtype=torch.float32)
samples = sobol_sampler.sample(bounds, num_samples=1024)  # Powers of 2 work best
```

**Characteristics:**

- **Advantages:** Excellent uniformity, works well up to ~40 dimensions, fast generation
- **Disadvantages:** Optimal for powers-of-2 sample sizes, requires scipy
- **Best for:** Medium to high-dimensional spaces, optimization studies, robust analysis

**Use Cases:**

```python
# High-quality workspace sampling
workspace_sampler = SobolSampler(scramble=True)
bounds_3d = torch.tensor([[-1, 1], [-1, 1], [0, 2]])
samples_1024 = workspace_sampler.sample(bounds_3d, num_samples=1024)

# Multi-robot configuration analysis
multi_robot_sampler = SobolSampler(seed=456, scramble=True)
dual_arm_bounds = torch.tensor([[-np.pi, np.pi]] * 14)  # 2x 7-DOF arms
dual_arm_samples = multi_robot_sampler.sample(dual_arm_bounds, num_samples=2048)
```

### 5. LatinHypercubeSampler (LHS)

**Latin Hypercube Sampling** for stratified sampling with guaranteed coverage in each dimension.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import LatinHypercubeSampler

# Create LHS sampler with optimization
lhs_sampler = LatinHypercubeSampler(
    seed=42,
    strength=1,  # Standard LHS
    optimization="random-cd"  # Random coordinate descent optimization
)

# Create optimized LHS for better quality
lhs_optimized = LatinHypercubeSampler(
    seed=42,
    strength=2,  # Strength-2 LHS (better uniformity)
    optimization="lloyd"  # Lloyd's optimization (slower but better)
)

# Generate LHS samples
bounds = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.float32)
samples = lhs_sampler.sample(bounds, num_samples=100)
```

**Characteristics:**

- **Advantages:** Excellent coverage with small samples, no clustering, works in high dimensions
- **Disadvantages:** May have correlation (unless optimized), not deterministic across sizes
- **Best for:** Experimental design, sensitivity analysis, efficient exploration

**Use Cases:**

```python
# Efficient workspace characterization
efficient_sampler = LatinHypercubeSampler(
    strength=2, 
    optimization="random-cd"
)
workspace_bounds = torch.tensor([
    [-0.6, 0.6], [-0.6, 0.6], [0.2, 1.0]
])
efficient_samples = efficient_sampler.sample(workspace_bounds, num_samples=200)

# Design of experiments for robot parameters
doe_sampler = LatinHypercubeSampler(optimization="lloyd")
design_bounds = torch.tensor([
    [0.2, 0.4],   # Link 1 length
    [0.2, 0.4],   # Link 2 length  
    [0.1, 0.3],   # Link 3 length
    [1.0, 5.0],   # Payload mass
    [0.5, 2.0]    # Speed scaling
])
design_samples = doe_sampler.sample(design_bounds, num_samples=50)
```

### 6. ImportanceSampler

**Adaptive importance sampling** that focuses on regions of high interest.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import ImportanceSampler

# Create importance sampler with adaptive refinement
importance_sampler = ImportanceSampler(
    seed=42,
    base_strategy=SamplingStrategy.HALTON,  # Base sampling method
    adaptation_rate=0.1,  # Learning rate for adaptation
    refinement_threshold=0.8  # Threshold for region refinement
)

# Generate samples that adapt to workspace characteristics
bounds = torch.tensor([[-1, 1], [-1, 1], [0, 2]], dtype=torch.float32)
samples = importance_sampler.sample(bounds, num_samples=1000)
```

**Characteristics:**

- **Advantages:** Focuses on important regions, adaptive, efficient for complex workspaces
- **Disadvantages:** Requires multiple iterations, more complex setup
- **Best for:** Complex workspace shapes, optimization-focused analysis, adaptive exploration

### 7. GaussianSampler

**Multi-variate Gaussian sampling** around specified mean with covariance control.

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import GaussianSampler

# Create Gaussian sampler around robot's nominal configuration
nominal_config = torch.tensor([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
config_std = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

gaussian_sampler = GaussianSampler(
    seed=42,
    mean=nominal_config,
    std=config_std
)

# Generate samples around the nominal configuration
joint_bounds = torch.tensor([[-np.pi, np.pi]] * 7)
samples = gaussian_sampler.sample(joint_bounds, num_samples=500)
```

**Characteristics:**

- **Advantages:** Natural for uncertainty analysis, controllable distribution, physically meaningful
- **Disadvantages:** May miss boundary regions, assumes Gaussian uncertainty
- **Best for:** Uncertainty propagation, robustness analysis, local exploration

## Factory Pattern Usage

The `SamplerFactory` provides a unified way to create and manage samplers:

```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
    SamplerFactory, 
    create_sampler
)
from embodichain.lab.sim.utility.workspace_analyzer.configs import SamplingStrategy

# Using the factory
factory = SamplerFactory()

# Create samplers by strategy
uniform_sampler = factory.create_sampler(
    SamplingStrategy.UNIFORM, 
    seed=42, 
    samples_per_dim=10
)

halton_sampler = factory.create_sampler(
    SamplingStrategy.HALTON,
    seed=42,
    skip=100
)

lhs_sampler = factory.create_sampler(
    SamplingStrategy.LHS,
    seed=42,
    strength=2,
    optimization="lloyd"
)

# Using the convenience function
sobol_sampler = create_sampler(
    SamplingStrategy.SOBOL,
    seed=42,
    scramble=True
)
```

## Advanced Usage Patterns

### 1. Multi-Stage Sampling

Combine different samplers for comprehensive analysis:

```python
def multi_stage_workspace_analysis(robot, sim_manager, bounds, total_samples=2000):
    """Multi-stage sampling for comprehensive workspace analysis."""
    
    # Stage 1: Coarse uniform grid for overall structure
    uniform_sampler = UniformSampler(samples_per_dim=8)  # 8^3 = 512 samples
    coarse_samples = uniform_sampler.sample(bounds, num_samples=512)
    
    # Stage 2: LHS for efficient space-filling
    lhs_sampler = LatinHypercubeSampler(optimization="random-cd")
    lhs_samples = lhs_sampler.sample(bounds, num_samples=744)  # 744 + 512 = 1256
    
    # Stage 3: Halton for low-discrepancy refinement
    halton_sampler = HaltonSampler(skip=50)
    halton_samples = halton_sampler.sample(bounds, num_samples=744)  # 744 + 1256 = 2000
    
    # Combine all samples
    all_samples = torch.cat([coarse_samples, lhs_samples, halton_samples], dim=0)
    
    return all_samples
```

### 2. Adaptive Sampling Density

Adjust sampling density based on workspace characteristics:

```python
def adaptive_density_sampling(workspace_bounds, complexity_measure, total_samples=1000):
    """Adapt sampling density based on workspace complexity."""
    
    if complexity_measure < 0.3:
        # Simple workspace - use uniform sampling
        sampler = UniformSampler(samples_per_dim=10)
    elif complexity_measure < 0.7:
        # Moderate complexity - use LHS
        sampler = LatinHypercubeSampler(optimization="random-cd")
    else:
        # Complex workspace - use importance sampling
        sampler = ImportanceSampler(
            base_strategy=SamplingStrategy.SOBOL,
            adaptation_rate=0.15
        )
    
    return sampler.sample(workspace_bounds, total_samples)
```

### 3. Stratified Sampling by Region

Divide workspace into regions with different sampling strategies:

```python
def stratified_workspace_sampling(workspace_bounds, num_samples=1000):
    """Sample different workspace regions with appropriate strategies."""
    
    # Define regions
    central_region = workspace_bounds * 0.5  # Central 50%
    peripheral_region = workspace_bounds     # Full workspace
    
    # Central region: High-density LHS for detailed analysis
    central_sampler = LatinHypercubeSampler(optimization="lloyd")
    central_samples = central_sampler.sample(central_region, num_samples // 2)
    
    # Peripheral region: Halton for boundary exploration
    peripheral_sampler = HaltonSampler(skip=100)
    peripheral_samples = peripheral_sampler.sample(peripheral_region, num_samples // 2)
    
    # Filter peripheral samples to exclude central region
    peripheral_mask = torch.any(
        torch.abs(peripheral_samples) > torch.abs(central_region[:, 1]) * 0.5,
        dim=1
    )
    peripheral_filtered = peripheral_samples[peripheral_mask]
    
    return torch.cat([central_samples, peripheral_filtered], dim=0)
```

## Performance Considerations

### Sample Size Guidelines

Different samplers perform optimally with different sample sizes:

```python
# Optimal sample sizes for different samplers
OPTIMAL_SAMPLE_SIZES = {
    SamplingStrategy.UNIFORM: "samples_per_dim^n_dims",  # Grid-based
    SamplingStrategy.RANDOM: "Any size",                 # No constraints  
    SamplingStrategy.HALTON: "500-5000",                # Medium sizes work best
    SamplingStrategy.SOBOL: "2^n (powers of 2)",        # 64, 128, 256, 512, 1024, 2048
    SamplingStrategy.LHS: "50-1000",                     # Efficient with small sizes
    SamplingStrategy.IMPORTANCE: "1000+",               # Needs adaptation samples
    SamplingStrategy.GAUSSIAN: "Any size"               # Flexible
}

def get_recommended_sample_size(strategy, n_dims, quality_level="medium"):
    """Get recommended sample size for given strategy and dimensions."""
    
    if strategy == SamplingStrategy.UNIFORM:
        if quality_level == "low":
            return 5 ** n_dims
        elif quality_level == "medium":
            return 10 ** n_dims
        else:
            return 20 ** n_dims
    
    elif strategy == SamplingStrategy.SOBOL:
        base_sizes = {"low": 256, "medium": 1024, "high": 4096}
        return base_sizes[quality_level]
    
    elif strategy == SamplingStrategy.LHS:
        base_sizes = {"low": 50, "medium": 200, "high": 500}
        return base_sizes[quality_level]
    
    else:
        base_sizes = {"low": 500, "medium": 1000, "high": 2000}
        return base_sizes[quality_level]
```

### Memory and Computational Efficiency

```python
def create_memory_efficient_sampler(n_dims, available_memory_gb=8):
    """Create sampler based on available memory and dimensions."""
    
    # Estimate memory requirements
    bytes_per_sample = n_dims * 4  # Float32
    max_samples = int(available_memory_gb * 1e9 * 0.1 / bytes_per_sample)  # Use 10% of memory
    
    if n_dims <= 4:
        # Low dimensions: uniform sampling is feasible
        samples_per_dim = int(max_samples ** (1/n_dims))
        return UniformSampler(samples_per_dim=samples_per_dim)
    
    elif n_dims <= 10:
        # Medium dimensions: quasi-random sequences
        return HaltonSampler(skip=50)
    
    else:
        # High dimensions: LHS or random
        if max_samples >= 1000:
            return LatinHypercubeSampler(optimization="random-cd")
        else:
            return RandomSampler()
```

## Integration with Workspace Analyzer

The samplers integrate seamlessly with the main workspace analyzer:

```python
from embodichain.lab.sim.utility.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig
)

# Configure sampling strategy in analyzer config
config = WorkspaceAnalyzerConfig(
    sampling_strategy=SamplingStrategy.LHS,
    sampling_params={
        "strength": 2,
        "optimization": "lloyd"
    },
    num_samples=1000
)

# Create analyzer with custom sampling
analyzer = WorkspaceAnalyzer(
    robot=robot,
    config=config,
    sim_manager=sim_manager
)

# Analysis will use the specified sampler
results = analyzer.analyze(visualize=True)
```

## Best Practices

### 1. Strategy Selection

```python
def select_optimal_sampler(analysis_type, n_dims, num_samples, quality_requirements):
    """Select the optimal sampling strategy based on analysis requirements."""
    
    if analysis_type == "boundary_detection":
        # Uniform sampling for precise boundary detection
        return SamplingStrategy.UNIFORM
    
    elif analysis_type == "volume_estimation":
        if n_dims <= 6 and num_samples >= 1000:
            return SamplingStrategy.SOBOL  # Best for integration
        else:
            return SamplingStrategy.LHS    # Efficient alternative
    
    elif analysis_type == "optimization":
        return SamplingStrategy.IMPORTANCE  # Adaptive exploration
    
    elif analysis_type == "uncertainty_analysis":
        return SamplingStrategy.GAUSSIAN   # Natural for uncertainty
    
    elif analysis_type == "general_exploration":
        if n_dims <= 8:
            return SamplingStrategy.HALTON  # Good uniformity
        else:
            return SamplingStrategy.LHS     # Scales well
    
    else:
        return SamplingStrategy.RANDOM     # Fallback option
```

### 2. Quality Assessment

```python
def assess_sampling_quality(samples, bounds):
    """Assess the quality of generated samples."""
    
    # Check coverage uniformity
    n_dims = samples.shape[1]
    grid_size = int(np.ceil(samples.shape[0] ** (1/n_dims)))
    
    # Discretize space and count samples per cell
    discretized = torch.floor(
        (samples - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) * grid_size
    ).clamp(0, grid_size-1)
    
    # Calculate uniformity metrics
    unique_cells = len(torch.unique(discretized, dim=0))
    total_cells = grid_size ** n_dims
    coverage_ratio = unique_cells / total_cells
    
    return {
        "coverage_ratio": coverage_ratio,
        "empty_cells": total_cells - unique_cells,
        "uniformity_score": 1.0 - np.std([len(discretized[discretized[:, i] == j]) 
                                         for i in range(n_dims) 
                                         for j in range(grid_size)])
    }
```

### 3. Error Handling

```python
def robust_sampler_creation(strategy, **kwargs):
    """Robustly create samplers with fallback options."""
    
    try:
        return create_sampler(strategy, **kwargs)
    
    except ImportError as e:
        if "scipy" in str(e) and strategy in [SamplingStrategy.LHS, SamplingStrategy.SOBOL]:
            logger.log_warning(f"SciPy not available for {strategy}, falling back to Halton")
            return create_sampler(SamplingStrategy.HALTON, **kwargs)
        else:
            raise
    
    except Exception as e:
        logger.log_warning(f"Failed to create {strategy} sampler: {e}, using Random sampler")
        return create_sampler(SamplingStrategy.RANDOM, **kwargs)
```
