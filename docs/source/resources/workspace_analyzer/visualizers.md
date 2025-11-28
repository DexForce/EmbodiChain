# Workspace Analyzer Visualizers

The **Workspace Analyzer Visualizers** module provides a comprehensive set of visualization tools for analyzing and displaying robotic workspace data in 3D space. This module is part of the EmbodiChain framework and offers multiple visualization backends and rendering strategies.

## Table of Contents

- [Overview](#overview)
- [Visualization Types](#visualization-types)
- [Factory Pattern](#factory-pattern)
- [Usage Examples](#usage-examples)
- [Backend Support](#backend-support)
- [Configuration](#configuration)
- [API Reference](#api-reference)

## Overview

The visualizers module enables researchers and developers to:

- **Analyze robotic workspace reachability** with different visualization styles
- **Compare workspace coverage** across different robot configurations
- **Visualize collision-free regions** and obstacles
- **Generate publication-quality figures** for research papers
- **Interactive exploration** of 3D workspace data

### Key Features

- **Multiple Visualization Types**: Point clouds, voxel grids, and sphere representations
- **Backend Flexibility**: Support for Open3D, Matplotlib, and simulation environments
- **Factory Pattern**: Easy creation and registration of custom visualizers
- **Performance Optimized**: Efficient rendering for large datasets
- **Extensible Design**: Simple interface for adding new visualization methods

## Visualization Types

### 1. Point Cloud Visualizer

**Best for**: Large datasets, fast rendering, memory efficiency

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import PointCloudVisualizer

visualizer = PointCloudVisualizer(
    backend='open3d',
    point_size=2.0
)
```

**Advantages**:

- Fast rendering for large point sets (>100k points)
- Memory efficient
- Clear spatial representation
- Interactive viewing with Open3D

**Use Cases**:

- Dense workspace sampling visualization
- Real-time workspace updates
- Large-scale reachability analysis

### 2. Voxel Visualizer

**Best for**: Occupancy maps, collision detection, discrete analysis

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VoxelVisualizer

visualizer = VoxelVisualizer(
    backend='open3d',
    voxel_size=0.01
)
```

**Advantages**:

- Clear volumetric representation
- Good for occupancy mapping
- Uniform spatial discretization
- Efficient for collision checking

**Considerations**:

- Memory intensive for high resolution
- May lose fine details
- Resolution-dependent accuracy

**Use Cases**:

- Occupancy grid visualization
- Collision-free space analysis
- Workspace volume calculations

### 3. Sphere Visualizer

**Best for**: Smooth visualization, reachability zones, uncertainty representation

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import SphereVisualizer

visualizer = SphereVisualizer(
    backend='open3d',
    sphere_radius=0.005,
    sphere_resolution=10
)
```

**Advantages**:

- Smooth visual appearance
- Excellent for showing reachability regions
- Intuitive spatial understanding
- Can represent uncertainty/tolerance zones

**Considerations**:

- More computationally expensive
- Higher memory usage
- Can become cluttered with many points

**Use Cases**:

- Reachability envelope visualization
- End-effector workspace boundaries
- Uncertainty visualization
- Publication-quality figures

## Factory Pattern

The module uses a factory pattern for easy visualizer creation and management:

### Basic Usage

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import create_visualizer, VisualizationType

# Quick creation
visualizer = create_visualizer(VisualizationType.POINT_CLOUD, backend='open3d')

# Using factory directly
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizerFactory

factory = VisualizerFactory()
visualizer = factory.create_visualizer('voxel', voxel_size=0.02)
```

### Custom Visualizer Registration

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizerFactory
from embodichain.lab.sim.utility.workspace_analyzer.visualizers.base_visualizer import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    # Implementation here
    pass

# Register custom visualizer
factory = VisualizerFactory()
factory.register_visualizer("custom", CustomVisualizer)

# Use custom visualizer
custom_viz = factory.create_visualizer("custom")
```

## Backend Support

### 1. Open3D Backend (`backend='open3d'`)

**Features**:

- Interactive 3D visualization
- High-quality rendering
- Built-in camera controls
- Export capabilities

**Requirements**: `pip install open3d`

### 2. Matplotlib Backend (`backend='matplotlib'`)

**Features**:

- Publication-quality figures
- 2D and 3D plotting
- Extensive customization
- Easy figure saving

**Requirements**: `pip install matplotlib`

### 3. Simulation Manager Backend (`backend='sim_manager'`)

**Features**:

- Integration with simulation environment
- Real-time visualization updates
- Robot context awareness
- Scene integration

### 4. Data Backend (`backend='data'`)

**Features**:

- Returns processed data without visualization
- Useful for batch processing
- Data export and analysis
- Headless operation

## Usage Examples

### Basic Workspace Visualization

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import create_visualizer

# Generate sample workspace data
points = np.random.rand(1000, 3) * 2 - 1  # Random points in [-1, 1]³
colors = np.random.rand(1000, 3)  # Random colors

# Create and use visualizer
visualizer = create_visualizer('point_cloud', backend='open3d', point_size=3.0)
pcd = visualizer.visualize(points, colors)
visualizer.show()
```

### Comparative Visualization

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizerFactory

# Create multiple visualizers for comparison
factory = VisualizerFactory()

point_viz = factory.create_visualizer('point_cloud', backend='open3d')
voxel_viz = factory.create_visualizer('voxel', backend='open3d', voxel_size=0.05)
sphere_viz = factory.create_visualizer('sphere', backend='open3d', sphere_radius=0.02)

# Visualize same data with different methods
workspace_data = generate_workspace_data()  # Your data generation function

point_result = point_viz.visualize(workspace_data)
voxel_result = voxel_viz.visualize(workspace_data)
sphere_result = sphere_viz.visualize(workspace_data)
```

### Batch Processing with Data Backend

```python
# Process multiple datasets without visualization
visualizer = create_visualizer('voxel', backend='data', voxel_size=0.01)

results = []
for dataset in datasets:
    processed_data = visualizer.visualize(dataset)
    results.append(processed_data)
    
# Analyze results
analyze_workspace_coverage(results)
```

### Integration with Simulation Environment

```python
# Using with simulation manager
from embodichain.lab.sim import SimulationManager

sim_manager = SimulationManager()
visualizer = create_visualizer(
    'sphere',
    backend='sim_manager',
    sim_manager=sim_manager,
    control_part_name='robot_arm'
)

# Visualize in simulation context
workspace_points = robot.calculate_workspace()
visualizer.visualize(workspace_points)
```

## Configuration

### Visualization Configuration

Each visualizer accepts configuration parameters:

```python
# Point cloud configuration
point_config = {
    'backend': 'open3d',
    'point_size': 2.5,
    'color_map': 'viridis'
}

# Voxel configuration
voxel_config = {
    'backend': 'open3d',
    'voxel_size': 0.015,
    'color_scheme': 'height_based'
}

# Sphere configuration
sphere_config = {
    'backend': 'open3d',
    'sphere_radius': 0.008,
    'sphere_resolution': 12,
    'material': 'glossy'
}
```

### Backend-Specific Options

```python
# Open3D specific options
open3d_config = {
    'window_name': 'Workspace Analysis',
    'width': 1024,
    'height': 768,
    'background_color': [0, 0, 0]
}

# Matplotlib specific options
mpl_config = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn'
}
```

## API Reference

### BaseVisualizer

Abstract base class for all visualizers.

#### Methods

- `visualize(points, colors=None, **kwargs) -> Any`: Main visualization method
- `show(**kwargs) -> None`: Display the visualization
- `save(filename, **kwargs) -> None`: Save visualization to file
- `clear() -> None`: Clear current visualization
- `set_camera_position(position, target, up) -> None`: Set camera view

### PointCloudVisualizer

Renders points as a point cloud.

#### Constructor Parameters

- `backend: str`: Visualization backend ('open3d', 'matplotlib', 'sim_manager', 'data')
- `point_size: float`: Size of points (default: 2.0)
- `config: Dict[str, Any]`: Additional configuration options

### VoxelVisualizer

Renders points as voxel grid.

#### VoxelVisualizer Parameters

- `backend: str`: Visualization backend
- `voxel_size: float`: Size of each voxel (default: 0.01)
- `config: Dict[str, Any]`: Additional configuration options

### SphereVisualizer

Renders points as spheres.

#### SphereVisualizer Parameters

- `backend: str`: Visualization backend
- `sphere_radius: float`: Radius of spheres (default: 0.005)
- `sphere_resolution: int`: Mesh resolution (default: 10)
- `config: Dict[str, Any]`: Additional configuration options

### VisualizerFactory

Singleton factory for creating visualizers.

#### Factory Methods

- `create_visualizer(viz_type, **kwargs) -> BaseVisualizer`: Create visualizer instance
- `register_visualizer(name, visualizer_class) -> None`: Register custom visualizer
- `list_available_types() -> List[str]`: List registered types
- `is_registered(viz_type) -> bool`: Check if type is registered

### Utility Functions

#### create_visualizer()

Convenience function for quick visualizer creation.

```python
def create_visualizer(
    viz_type: Optional[VisualizationType | str] = None,
    **kwargs: Any
) -> BaseVisualizer:
```

## Performance Tips

### Memory Optimization

- Use `backend='data'` for batch processing without visualization
- Reduce `voxel_size` for VoxelVisualizer with large datasets
- Use PointCloudVisualizer for datasets > 50k points

### Rendering Performance

- Decrease `sphere_resolution` for SphereVisualizer with many points
- Use `backend='matplotlib'` for static figures
- Enable GPU acceleration in Open3D when available

### Best Practices

1. **Choose appropriate visualizer type** based on data size and analysis goals
2. **Use factory pattern** for consistent visualizer creation
3. **Configure backends** based on output requirements (interactive vs. static)
4. **Batch process** large datasets using data backend
5. **Cache visualizations** when possible to avoid recomputation

## Troubleshooting

### Common Issues

1. **ImportError for Open3D**: Install with `pip install open3d`
2. **Memory errors with large datasets**: Reduce voxel size or use point cloud visualization
3. **Slow rendering**: Decrease sphere resolution or switch to point cloud
4. **Empty visualizations**: Check data format and coordinate systems

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('embodichain.lab.sim.utility.workspace_analyzer').setLevel(logging.DEBUG)
```

## Contributing

To add a new visualizer:

1. Inherit from `BaseVisualizer`
2. Implement required abstract methods
3. Register with factory using `register_visualizer()`
4. Add comprehensive tests and documentation

Example template:

```python
from embodichain.lab.sim.utility.workspace_analyzer.visualizers.base_visualizer import BaseVisualizer

class MyCustomVisualizer(BaseVisualizer):
    def __init__(self, backend='open3d', **kwargs):
        super().__init__(backend)
        # Custom initialization
        
    def visualize(self, points, colors=None, **kwargs):
        # Implementation
        pass
        
    def show(self, **kwargs):
        # Display implementation
        pass
```
