# Workspace Analyzer Configurations

The **Workspace Analyzer Configs** module provides a comprehensive configuration system for robotic workspace analysis. This module offers type-safe, well-documented configuration classes that control every aspect of workspace analysis, from sampling strategies to visualization options.

Based on the `WorkspaceAnalyzerConfig` dataclass and its sub-configurations, this system provides modular, extensible configuration management for all aspects of workspace analysis.

## Table of Contents

- [Overview](#overview)
- [Configuration Types](#configuration-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The configs module enables fine-grained control over:

- **Sampling Strategies**: Control how joint space is sampled
- **Caching Behavior**: Configure result caching and storage
- **Metric Calculation**: Define which metrics to compute and how
- **Visualization Options**: Customize visualization appearance and behavior
- **Dimensional Constraints**: Set workspace bounds and exclusion zones

### Key Features

- **Type Safety**: All configurations use dataclasses with type hints
- **Validation**: Built-in validation and sensible defaults
- **Extensibility**: Easy to extend with custom configuration options
- **Documentation**: Comprehensive docstrings for all parameters
- **Modularity**: Separate configs for different analysis aspects

## Configuration Types

### 1. Sampling Configuration

Controls how the robot's joint space is sampled for workspace analysis.

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    SamplingConfig, SamplingStrategy
)

# Basic uniform sampling
config = SamplingConfig(
    strategy=SamplingStrategy.UNIFORM,
    num_samples=10000,
    grid_resolution=20,
    seed=42
)

# Random sampling with larger batch size
random_config = SamplingConfig(
    strategy=SamplingStrategy.RANDOM,
    num_samples=50000,
    batch_size=5000,
    seed=123
)
```

#### Available Sampling Strategies

- **UNIFORM**: Uniform grid sampling across joint space
- **RANDOM**: Pseudo-random sampling with specified seed
- **HALTON**: Quasi-random Halton sequence (better coverage)
- **SOBOL**: Quasi-random Sobol sequence (low-discrepancy)
- **LATIN_HYPERCUBE**: Latin Hypercube Sampling (space-filling)
- **IMPORTANCE**: Importance sampling with custom weight function
- **GAUSSIAN**: Gaussian distribution around specified mean

#### Advanced Sampling Examples

```python
# Quasi-random sampling for better coverage
halton_config = SamplingConfig(
    strategy=SamplingStrategy.HALTON,
    num_samples=25000,
    batch_size=2500
)

# Gaussian sampling around robot's nominal pose
gaussian_config = SamplingConfig(
    strategy=SamplingStrategy.GAUSSIAN,
    num_samples=15000,
    gaussian_mean=0.0,  # Center of joint range
    gaussian_std=0.3    # 30% of range as std dev
)

# Importance sampling with custom weight function
def weight_function(joint_values):
    # Higher weight for configurations closer to zero
    return np.exp(-np.sum(joint_values**2))

importance_config = SamplingConfig(
    strategy=SamplingStrategy.IMPORTANCE,
    num_samples=20000,
    importance_weight_func=weight_function
)
```

### 2. Cache Configuration

Manages caching of analysis results for improved performance.

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import CacheConfig
from pathlib import Path

# Basic caching configuration
cache_config = CacheConfig(
    enabled=True,
    cache_dir=Path("./workspace_cache"),
    use_hash=True,
    compression=True
)

# High-performance caching setup
performance_cache = CacheConfig(
    enabled=True,
    cache_dir=Path("/fast_storage/workspace_cache"),
    use_hash=True,
    compression=False,  # Faster access, more storage
    max_cache_size_mb=5000,
    cache_format="npz"
)
```

#### Cache Configuration Options

- **enabled**: Enable/disable caching entirely
- **cache_dir**: Custom cache directory (default: system cache)
- **use_hash**: Hash-based cache keys for consistency
- **compression**: Compress cache files to save space
- **max_cache_size_mb**: Automatic cleanup of old cache files
- **cache_format**: Storage format ('npz', 'pkl', 'h5')

### 3. Metric Configuration

Defines which workspace metrics to compute and their parameters.

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    MetricConfig, MetricType, ReachabilityConfig, 
    ManipulabilityConfig, DensityConfig
)

# Comprehensive metric analysis
metric_config = MetricConfig(
    enabled_metrics=[MetricType.ALL],
    reachability=ReachabilityConfig(
        voxel_size=0.005,  # High resolution
        min_points_per_voxel=2,
        compute_coverage=True
    ),
    manipulability=ManipulabilityConfig(
        jacobian_threshold=0.001,
        compute_isotropy=True,
        compute_heatmap=True
    ),
    density=DensityConfig(
        radius=0.03,
        k_neighbors=50,
        compute_distribution=True
    ),
    save_results=True,
    output_format="json"
)
```

#### Metric Types

**Reachability Metrics**:

- Workspace volume calculation
- Coverage percentage
- Reachable point density

```python
reachability_config = ReachabilityConfig(
    voxel_size=0.01,           # Voxel resolution for volume
    min_points_per_voxel=1,    # Occupancy threshold
    compute_coverage=True      # Coverage vs bounding box
)
```

**Manipulability Metrics**:

- Manipulability index throughout workspace
- Isotropy analysis
- Dexterity heatmaps

```python
manipulability_config = ManipulabilityConfig(
    jacobian_threshold=0.01,   # Minimum valid manipulability
    compute_isotropy=True,     # Condition number analysis
    compute_heatmap=False      # Generate spatial heatmap
)
```

**Density Metrics**:

- Local point density
- Distribution statistics
- Clustering analysis

```python
density_config = DensityConfig(
    radius=0.05,              # Local neighborhood radius
    k_neighbors=30,           # Number of neighbors to consider
    compute_distribution=True  # Statistical distribution
)
```

### 4. Visualization Configuration

Controls the appearance and behavior of workspace visualizations.

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    VisualizationConfig, VisualizationType
)

# High-quality point cloud visualization
viz_config = VisualizationConfig(
    enabled=True,
    vis_type=VisualizationType.POINT_CLOUD,
    point_size=3.0,
    alpha=0.8,
    color_by_distance=True,
    voxel_size=0.02,
    is_voxel_down=True,
    show_unreachable_points=True
)

# Sphere visualization for presentations
sphere_viz_config = VisualizationConfig(
    vis_type=VisualizationType.SPHERE,
    sphere_radius=0.008,
    sphere_resolution=15,
    alpha=0.6,
    show_unreachable_points=False
)
```

#### Visualization Options

- **vis_type**: Visualization method (POINT_CLOUD, VOXEL, SPHERE, MESH, HEATMAP)
- **point_size**: Size of rendered points
- **alpha**: Transparency level (0.0-1.0)
- **color_by_distance**: Color-code by distance from base
- **voxel_size**: Downsampling resolution
- **show_unreachable_points**: Display unreachable regions

### 5. Dimension Constraints

Defines workspace bounds, exclusion zones, and physical constraints.

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import DimensionConstraint
import numpy as np

# Basic workspace bounds
dimension_config = DimensionConstraint(
    min_bounds=np.array([-1.0, -1.0, 0.0]),  # [x_min, y_min, z_min]
    max_bounds=np.array([1.0, 1.0, 2.0]),    # [x_max, y_max, z_max]
    joint_limits_scale=0.95,  # Use 95% of joint range
    ground_height=0.0
)

# Complex workspace with exclusion zones
complex_constraints = DimensionConstraint(
    min_bounds=np.array([-2.0, -1.5, -0.5]),
    max_bounds=np.array([2.0, 1.5, 2.5]),
    joint_limits_scale=0.9,
    exclude_zones=[
        # Obstacle 1: table
        (np.array([0.3, -0.5, 0.0]), np.array([1.2, 0.5, 0.8])),
        # Obstacle 2: wall
        (np.array([1.8, -1.5, 0.0]), np.array([2.0, 1.5, 2.5]))
    ],
    ground_height=-0.1,
    enforce_collision_free=True,
    self_collision_check=True
)
```

#### Constraint Options

- **min_bounds/max_bounds**: Cartesian workspace limits
- **joint_limits_scale**: Scale factor for joint range usage
- **exclude_zones**: List of forbidden regions
- **ground_height**: Floor level for filtering
- **enforce_collision_free**: Enable collision checking
- **self_collision_check**: Check robot self-collisions

## Usage Examples

### Basic Workspace Analysis Setup

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import *

# Complete configuration for basic analysis
sampling_config = SamplingConfig(
    strategy=SamplingStrategy.UNIFORM,
    num_samples=15000,
    grid_resolution=25,
    batch_size=1500
)

cache_config = CacheConfig(
    enabled=True,
    compression=True,
    max_cache_size_mb=2000
)

metric_config = MetricConfig(
    enabled_metrics=[MetricType.REACHABILITY, MetricType.DENSITY],
    save_results=True
)

viz_config = VisualizationConfig(
    vis_type=VisualizationType.POINT_CLOUD,
    point_size=2.5,
    color_by_distance=True
)

constraints = DimensionConstraint(
    min_bounds=np.array([-1.2, -1.2, 0.0]),
    max_bounds=np.array([1.2, 1.2, 1.8]),
    joint_limits_scale=0.9
)
```

### High-Performance Configuration

```python
# Configuration optimized for speed and large datasets
fast_sampling = SamplingConfig(
    strategy=SamplingStrategy.HALTON,  # Better than random
    num_samples=100000,                # Large dataset
    batch_size=10000,                  # Large batches
    seed=42
)

fast_cache = CacheConfig(
    enabled=True,
    compression=False,                 # Faster access
    cache_format="npz",               # Efficient format
    max_cache_size_mb=10000           # Large cache
)

minimal_metrics = MetricConfig(
    enabled_metrics=[MetricType.REACHABILITY],  # Only essential
    reachability=ReachabilityConfig(
        voxel_size=0.02,              # Lower resolution
        compute_coverage=False         # Skip expensive computation
    ),
    save_results=False                # Don't save to disk
)

efficient_viz = VisualizationConfig(
    vis_type=VisualizationType.VOXEL, # Faster than spheres
    voxel_size=0.03,                  # Aggressive downsampling
    is_voxel_down=True,
    show_unreachable_points=False     # Reduce complexity
)
```

### Research Configuration

```python
# Configuration for detailed research analysis
research_sampling = SamplingConfig(
    strategy=SamplingStrategy.SOBOL,  # Low-discrepancy sequence
    num_samples=50000,
    batch_size=2500
)

persistent_cache = CacheConfig(
    enabled=True,
    cache_dir=Path("./research_cache"),
    use_hash=True,
    compression=True,
    cache_format="h5"                 # Good for large datasets
)

comprehensive_metrics = MetricConfig(
    enabled_metrics=[MetricType.ALL],
    reachability=ReachabilityConfig(
        voxel_size=0.005,             # High resolution
        compute_coverage=True
    ),
    manipulability=ManipulabilityConfig(
        jacobian_threshold=0.001,
        compute_isotropy=True,
        compute_heatmap=True          # Generate heatmaps
    ),
    density=DensityConfig(
        radius=0.02,                  # Fine-grained density
        k_neighbors=50,
        compute_distribution=True
    ),
    save_results=True,
    output_format="json"
)

publication_viz = VisualizationConfig(
    vis_type=VisualizationType.SPHERE,
    sphere_radius=0.006,
    sphere_resolution=20,             # High quality spheres
    alpha=0.7,
    color_by_distance=True,
    show_unreachable_points=True
)

detailed_constraints = DimensionConstraint(
    min_bounds=np.array([-1.5, -1.5, -0.2]),
    max_bounds=np.array([1.5, 1.5, 2.0]),
    joint_limits_scale=1.0,           # Full joint range
    exclude_zones=[],                 # No exclusions for research
    enforce_collision_free=True,
    self_collision_check=True
)
```

### Configuration Composition

```python
# Create configurations programmatically
def create_analysis_config(
    resolution: str = "medium",
    enable_cache: bool = True,
    visualization_type: str = "point_cloud"
) -> tuple:
    """Create configuration set based on analysis requirements."""
    
    # Resolution-dependent settings
    resolution_params = {
        "low": {"samples": 5000, "voxel": 0.05, "batch": 1000},
        "medium": {"samples": 20000, "voxel": 0.02, "batch": 2000},
        "high": {"samples": 80000, "voxel": 0.01, "batch": 5000}
    }
    params = resolution_params[resolution]
    
    sampling_config = SamplingConfig(
        strategy=SamplingStrategy.HALTON,
        num_samples=params["samples"],
        batch_size=params["batch"]
    )
    
    cache_config = CacheConfig(enabled=enable_cache)
    
    viz_config = VisualizationConfig(
        vis_type=getattr(VisualizationType, visualization_type.upper()),
        voxel_size=params["voxel"]
    )
    
    return sampling_config, cache_config, viz_config

# Usage
low_res_configs = create_analysis_config("low", True, "voxel")
high_res_configs = create_analysis_config("high", False, "sphere")
```

## Best Practices

### Configuration Management

1. **Use Type Hints**: Leverage the type safety provided by dataclasses
2. **Validate Parameters**: Check parameter ranges before analysis
3. **Document Choices**: Comment configuration choices for reproducibility
4. **Version Configs**: Save configuration alongside results
5. **Test Different Settings**: Experiment with sampling strategies

### Performance Optimization

```python
# Performance tips based on use case
def optimize_for_speed():
    return SamplingConfig(
        strategy=SamplingStrategy.RANDOM,  # Fastest generation
        batch_size=10000,                  # Large batches
        num_samples=20000                  # Moderate resolution
    )

def optimize_for_accuracy():
    return SamplingConfig(
        strategy=SamplingStrategy.SOBOL,   # Better coverage
        batch_size=2000,                   # Moderate batches
        num_samples=100000                 # High resolution
    )

def optimize_for_memory():
    return SamplingConfig(
        batch_size=500,                    # Small batches
        num_samples=15000                  # Moderate total
    )
```

### Configuration Validation

```python
def validate_config(config):
    """Validate configuration parameters."""
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    if config.batch_size > config.num_samples:
        raise ValueError("batch_size cannot exceed num_samples")
    
    if hasattr(config, 'voxel_size') and config.voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

# Usage
try:
    validate_config(my_config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## API Reference

### SamplingConfig

Configuration for joint space sampling strategies.

#### Parameters

- **strategy**: `SamplingStrategy` - Sampling method to use
- **num_samples**: `int` - Total number of samples to generate (default: 1000)
- **grid_resolution**: `int` - Grid resolution for uniform sampling (default: 10)
- **batch_size**: `int` - Batch size for processing (default: 1000)
- **seed**: `int` - Random seed for reproducibility (default: 42)
- **importance_weight_func**: `Optional[Callable]` - Weight function for importance sampling
- **gaussian_mean**: `Optional[float]` - Mean for Gaussian sampling
- **gaussian_std**: `Optional[float]` - Standard deviation for Gaussian sampling

### CacheConfig

Configuration for result caching and storage.

#### CacheConfig Parameters

- **enabled**: `bool` - Enable caching (default: True)
- **cache_dir**: `Optional[Path]` - Cache directory path
- **use_hash**: `bool` - Use hash-based cache keys (default: True)
- **compression**: `bool` - Compress cache files (default: True)
- **max_cache_size_mb**: `int` - Maximum cache size in MB (default: 1000)
- **cache_format**: `str` - Cache file format: 'npz', 'pkl', 'h5' (default: 'npz')

### MetricConfig

Configuration for workspace analysis metrics.

#### MetricConfig Parameters

- **enabled_metrics**: `List[MetricType]` - List of metrics to compute
- **reachability**: `ReachabilityConfig` - Reachability analysis settings
- **manipulability**: `ManipulabilityConfig` - Manipulability analysis settings
- **density**: `DensityConfig` - Density analysis settings
- **save_results**: `bool` - Save results to file (default: True)
- **output_format**: `str` - Output format: 'json', 'yaml', 'pkl' (default: 'json')

### VisualizationConfig

Configuration for visualization appearance and behavior.

#### VisualizationConfig Parameters

- **enabled**: `bool` - Enable visualization (default: True)
- **vis_type**: `VisualizationType` - Visualization type (default: POINT_CLOUD)
- **voxel_size**: `float` - Voxel downsampling size (default: 0.05)
- **point_size**: `float` - Point size in visualization (default: 4.0)
- **alpha**: `float` - Transparency level 0.0-1.0 (default: 0.5)
- **color_by_distance**: `bool` - Color by distance from base (default: True)
- **sphere_radius**: `float` - Sphere radius for sphere visualization (default: 0.005)
- **show_unreachable_points**: `bool` - Show unreachable points (default: False)

### DimensionConstraint

Configuration for workspace bounds and constraints.

#### DimensionConstraint Parameters

- **min_bounds**: `Optional[np.ndarray]` - Minimum workspace bounds [x, y, z]
- **max_bounds**: `Optional[np.ndarray]` - Maximum workspace bounds [x, y, z]
- **joint_limits_scale**: `float` - Joint range scale factor (default: 1.0)
- **exclude_zones**: `List[Tuple[np.ndarray, np.ndarray]]` - Exclusion zones
- **ground_height**: `float` - Ground plane height (default: 0.0)
- **enforce_collision_free**: `bool` - Enable collision checking (default: False)
- **self_collision_check**: `bool` - Check self-collisions (default: False)

## Configuration Examples

### Minimal Configuration

```python
# Simplest possible configuration
from embodichain.lab.sim.utility.workspace_analyzer.configs import *

config = SamplingConfig()  # Uses defaults
# Result: uniform sampling, 1000 samples, grid resolution 10
```

### Production Configuration

```python
# Production-ready configuration
production_sampling = SamplingConfig(
    strategy=SamplingStrategy.HALTON,
    num_samples=25000,
    batch_size=2500,
    seed=42
)

production_cache = CacheConfig(
    enabled=True,
    cache_dir=Path("/app/cache"),
    compression=True,
    max_cache_size_mb=5000
)

production_metrics = MetricConfig(
    enabled_metrics=[MetricType.REACHABILITY, MetricType.MANIPULABILITY],
    save_results=True,
    output_format="json"
)
```

### Debug Configuration

```python
# Configuration for debugging and development
debug_config = SamplingConfig(
    strategy=SamplingStrategy.UNIFORM,
    num_samples=500,        # Small for fast testing
    batch_size=100,         # Small batches
    grid_resolution=8       # Low resolution
)

debug_viz = VisualizationConfig(
    vis_type=VisualizationType.POINT_CLOUD,
    point_size=5.0,         # Large points for visibility
    show_unreachable_points=True,  # Show all data
    alpha=0.9               # High opacity
)
```
