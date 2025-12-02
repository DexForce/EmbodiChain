# Workspace Analyzer Metrics

The **Workspace Analyzer Metrics** module provides comprehensive metrics and analysis tools for evaluating robotic workspace characteristics. This module implements various quantitative measures to assess workspace quality, coverage, manipulability, and performance.

## Overview

This module provides three main workspace analysis metrics:

- **ReachabilityMetric**: Voxel-based volume and coverage analysis ✅
- **ManipulabilityMetric**: Dexterity analysis (heuristic method or with Jacobians) ✅
- **DensityMetric**: Point distribution and density analysis ✅

**Note**: Some advanced features like spatial heatmaps, clustering analysis, and robot-specific helpers are not yet implemented.

## Table of Contents

- [Metric Types](#metric-types)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Quick Reference](#quick-reference)

## Metric Types

### 1. ReachabilityMetric

Computes workspace volume and coverage using voxel-based analysis.

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics import ReachabilityMetric

# Basic usage
workspace_points = np.random.uniform(-1, 1, size=(1000, 3))
reachability = ReachabilityMetric()
results = reachability.compute(workspace_points)

print(f"Volume: {results['volume']:.4f} m³")
print(f"Coverage: {results['coverage']:.1f}%")
```

**Returns**: volume, coverage percentage, voxel count, bounding box, centroid

### 2. ManipulabilityMetric

Analyzes dexterity using distance-based heuristic or Yoshikawa index (with Jacobians).

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics import ManipulabilityMetric

# Basic usage (heuristic method)
workspace_points = np.random.uniform(-1, 1, size=(1000, 3))
manipulability = ManipulabilityMetric()
results = manipulability.compute(workspace_points)

print(f"Mean manipulability: {results['mean_manipulability']:.3f}")
```

**Returns**: mean, std, min, max manipulability; condition numbers (if Jacobians provided)

### 3. DensityMetric

Computes local point density using radius-based neighborhood analysis.

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics import DensityMetric

# Basic usage
workspace_points = np.random.uniform(-1, 1, size=(1000, 3))
density = DensityMetric()
results = density.compute(workspace_points)

print(f"Mean density: {results['mean_density']:.2f}")
```

**Returns**: mean, std, min, max density; distribution histogram (if enabled)

## Usage Examples

### Basic Usage

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics import (
    ReachabilityMetric, ManipulabilityMetric, DensityMetric
)

# Generate sample data
workspace_points = np.random.uniform(-1, 1, size=(5000, 3))

# Compute all metrics
reach_results = ReachabilityMetric().compute(workspace_points)
manip_results = ManipulabilityMetric().compute(workspace_points)
density_results = DensityMetric().compute(workspace_points)

print(f"Volume: {reach_results['volume']:.4f} m³")
print(f"Mean manipulability: {manip_results['mean_manipulability']:.3f}")
print(f"Mean density: {density_results['mean_density']:.2f}")
```

### With Custom Configuration

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    ReachabilityConfig, DensityConfig
)

# Custom configuration
reach_config = ReachabilityConfig(voxel_size=0.02, compute_coverage=True)
density_config = DensityConfig(radius=0.1, compute_distribution=True)

reachability = ReachabilityMetric(reach_config)
density = DensityMetric(density_config)

reach_results = reachability.compute(workspace_points)
density_results = density.compute(workspace_points)
```

## Configuration

All metrics can be customized using configuration classes:

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    ReachabilityConfig, ManipulabilityConfig, DensityConfig
)

# Reachability configuration
reach_config = ReachabilityConfig(
    voxel_size=0.01,          # Voxel size in meters
    min_points_per_voxel=1,   # Minimum points for occupancy
    compute_coverage=True     # Enable coverage calculation
)

# Manipulability configuration  
manip_config = ManipulabilityConfig(
    jacobian_threshold=0.01,  # Minimum valid manipulability
    compute_isotropy=True     # Enable condition number analysis
)

# Density configuration
density_config = DensityConfig(
    radius=0.05,              # Neighborhood radius
    compute_distribution=True # Enable histogram computation
)
```

## API Reference

**ReachabilityMetric**: Returns volume, coverage, num_voxels, bounding_box, centroid

**ManipulabilityMetric**: Returns mean/std/min/max manipulability, num_valid_points, condition numbers (if available)

**DensityMetric**: Returns mean/std/min/max density, distribution histogram (if enabled)

## Quick Reference

```python
# Default usage
reachability = ReachabilityMetric()
manipulability = ManipulabilityMetric() 
density = DensityMetric()

# Custom configuration
from embodichain.lab.sim.utility.workspace_analyzer.configs import ReachabilityConfig
reach_config = ReachabilityConfig(voxel_size=0.02, compute_coverage=True)
reachability = ReachabilityMetric(reach_config)
```
