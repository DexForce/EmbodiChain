# Workspace Analyzer Visualizers

The visualizers module provides visualization tools for analyzing robotic workspace data in 3D space.

## Table of Contents

- [Overview](#overview)
- [Visualization Types](#visualization-types)
- [Usage Examples](#usage-examples)
- [Backend Support](#backend-support)
- [Quick Reference](#quick-reference)

## Overview

The visualizers module enables:

- **Workspace reachability visualization** with multiple rendering styles
- **3D point cloud, voxel, and sphere representations**
- **Multiple backends**: Open3D, Matplotlib, and simulation environments
- **Factory pattern** for easy visualizer creation

## Visualization Types

### 1. Point Cloud Visualizer ✅

Fast rendering for large datasets.

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import PointCloudVisualizer

visualizer = PointCloudVisualizer(backend='open3d', point_size=2.0)
```

**Best for**: Large point sets (>10k points), fast rendering

### 2. Voxel Visualizer ✅

Volumetric representation for occupancy maps.

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VoxelVisualizer

visualizer = VoxelVisualizer(backend='open3d', voxel_size=0.01)
```

**Best for**: Occupancy grids, collision detection

### 3. Sphere Visualizer ✅

Smooth visualization for reachability zones.

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import SphereVisualizer

visualizer = SphereVisualizer(backend='open3d', sphere_radius=0.005)
```

**Best for**: Publication-quality figures, smooth appearance

## Usage Examples

### Basic Usage

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import create_visualizer

# Generate workspace data
points = np.random.rand(1000, 3) * 2 - 1  # Random points in [-1, 1]³
colors = np.random.rand(1000, 3)  # Random colors

# Create and use visualizer
visualizer = create_visualizer('point_cloud', backend='open3d', point_size=3.0)
result = visualizer.visualize(points, colors)
visualizer.show()
```

### Factory Pattern

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizerFactory

factory = VisualizerFactory()
visualizer = factory.create_visualizer('voxel', backend='open3d', voxel_size=0.02)
```

## Backend Support

### Available Backends

- **`open3d`**: Interactive 3D visualization (requires `pip install open3d`)
- **`matplotlib`**: Static figures and plots (requires `pip install matplotlib`)
- **`sim_manager`**: Integration with simulation environment
- **`data`**: Returns processed data without visualization (headless mode)

## Quick Reference

**Available Visualizers**:

- **PointCloudVisualizer**: Fast rendering for large datasets
- **VoxelVisualizer**: Volumetric representation for occupancy maps
- **SphereVisualizer**: Smooth visualization for publication figures

**Common Parameters**:

- **PointCloud**: `backend`, `point_size`
- **Voxel**: `backend`, `voxel_size`
- **Sphere**: `backend`, `sphere_radius`, `sphere_resolution`

**Quick Creation**:

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import create_visualizer

# Create any visualizer
visualizer = create_visualizer('point_cloud', backend='open3d', point_size=2.0)
visualizer = create_visualizer('voxel', backend='open3d', voxel_size=0.01)
visualizer = create_visualizer('sphere', backend='open3d', sphere_radius=0.005)
```
