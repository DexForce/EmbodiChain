# Workspace Analyzer Constraints

The **Workspace Analyzer Constraints** module provides comprehensive constraint checking and validation for robotic workspace analysis. This module ensures that workspace analysis respects physical limitations, safety boundaries, and operational constraints.

## Table of Contents

- [Overview](#overview)
- [Constraint Types](#constraint-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The constraints module provides basic constraint checking for workspace analysis:

- **Spatial Boundaries**: Define workspace limits using min/max bounds
- **Exclusion Zones**: Avoid specified rectangular obstacle regions  
- **Ground Constraints**: Handle floor plane height limits
- **Configuration-based Setup**: Easy configuration through DimensionConstraint

### Key Features

- **Simple Boundary Checking**: Min/max bounds validation
- **Basic Obstacle Avoidance**: Rectangular exclusion zone filtering
- **Multi-Backend Support**: Works with both NumPy arrays and PyTorch tensors
- **Configuration Integration**: Uses DimensionConstraint config objects

### TODO - Not Yet Implemented

- **Joint Limit Constraints**: Physical joint limitation checking
- **Advanced Collision Detection**: Robot mesh-based collision checking with environment
- **Self-Collision Detection**: Robot self-collision validation  
- **Complex Safety Margins**: Advanced safety margin calculations
- **Dynamic Constraint Updates**: Real-time constraint modification
- **Performance Statistics**: Constraint checking metrics

**Note**: The current `check_collision()` method only performs simple rectangular exclusion zone checking, not true geometric collision detection.

## Constraint Types

### 1. Spatial Boundary Constraints

Define workspace boundaries using min/max bounds:

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Basic workspace boundaries
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.0, -1.0, 0.0]),  # [x_min, y_min, z_min]
    max_bounds=np.array([1.0, 1.0, 2.0]),    # [x_max, y_max, z_max]
    ground_height=0.0
)
```

### 2. Exclusion Zone Constraints

Avoid specified rectangular obstacle regions (simple collision avoidance):

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Workspace with exclusion zones
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-2.0, -1.5, 0.0]),
    max_bounds=np.array([2.0, 1.5, 2.0]),
    exclude_zones=[
        # Table obstacle
        (np.array([0.3, -0.5, 0.0]), np.array([1.2, 0.5, 0.8])),
        # Wall section
        (np.array([1.8, -1.5, 0.0]), np.array([2.0, 1.5, 2.0]))
    ]
)
```

### 3. Configuration-based Setup (Recommended)

Use DimensionConstraint for easy configuration:

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import DimensionConstraint
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Create configuration
config = DimensionConstraint(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 2.0]),
    ground_height=0.0,
    exclude_zones=[
        (np.array([0.2, 0.2, 0.0]), np.array([0.4, 0.4, 0.2]))
    ]
)

# Create checker from config
checker = WorkspaceConstraintChecker.from_config(config)
```

### TODO - Planned Features

#### Advanced Constraint Features (Not Implemented)

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import DimensionConstraint

# TODO: These parameters exist in DimensionConstraint but are not implemented
config = DimensionConstraint(
    joint_limits_scale=0.95,        # TODO: Joint limit checking not implemented
    enforce_collision_free=True,    # TODO: Mesh-based collision not implemented  
    self_collision_check=True       # TODO: Self-collision checking not implemented
)
```

**Current Implementation Status**:

- ✅ **Simple exclusion zones**: Rectangular bounding box obstacle avoidance
- ❌ **Joint limits**: Physical joint limitation checking not implemented
- ❌ **Mesh collision**: Advanced geometric collision detection not implemented
- ❌ **Self-collision**: Robot self-collision validation not implemented

**Planned Features**:

- Enforce mechanical joint limits and prevent over-extension
- True mesh-based collision detection with environment geometry
- Self-collision checking between robot links
- Singular configuration avoidance

## Usage Examples

### Basic Constraint Checking

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Create constraint checker
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 1.5]),
    ground_height=0.0
)

# Test points
test_points = np.array([
    [0.5, 0.5, 0.5],    # Valid point
    [2.0, 0.0, 0.5],    # Outside x boundary
    [0.0, 0.0, -0.1],   # Below ground
    [0.8, 0.8, 1.2]     # Valid point
])

# Check boundaries
valid_bounds = checker.check_bounds(test_points)
print(f"Valid bounds: {valid_bounds}")  # [True, False, False, True]

# Filter valid points
valid_points = checker.filter_points(test_points)
print(f"Filtered points shape: {valid_points.shape}")  # (2, 3)
```

### Workspace with Obstacles

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Workspace with exclusion zones
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.5, -1.5, 0.0]),
    max_bounds=np.array([1.5, 1.5, 2.0]),
    exclude_zones=[
        # Table obstacle
        (np.array([0.2, -0.4, 0.0]), np.array([0.8, 0.4, 0.8])),
    ]
)

# Test points
test_points = np.array([
    [0.0, 0.0, 0.5],    # Valid point
    [0.5, 0.0, 0.4],    # Inside table (excluded)
    [1.0, 1.0, 1.0]     # Valid point
])

# Check collision (simple exclusion zone checking)
valid_collision = checker.check_collision(test_points)
print(f"Collision-free: {valid_collision}")  # [True, False, True]

# Comprehensive filtering (bounds + collision)
safe_points = checker.filter_points(test_points)
print(f"Safe points: {len(safe_points)} out of {len(test_points)}")
```

### Configuration-based Setup

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import DimensionConstraint
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Create configuration object
config = DimensionConstraint(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 1.5]),
    ground_height=0.0,
    exclude_zones=[
        # Add table obstacle
        (np.array([0.2, 0.2, 0.0]), np.array([0.6, 0.6, 0.8]))
    ]
)

# Create checker from config
checker = WorkspaceConstraintChecker.from_config(config)

# Add more obstacles dynamically
checker.add_exclude_zone(
    min_bounds=np.array([-0.3, 0.7, 0.0]),
    max_bounds=np.array([0.3, 1.0, 1.2])
)

print(f"Total exclusion zones: {checker.get_num_exclude_zones()}")
```

### Dynamic Obstacle Management

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Add and remove obstacles dynamically
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 1.5])
)

# Add obstacles
checker.add_exclude_zone(
    min_bounds=np.array([0.2, 0.2, 0.0]),
    max_bounds=np.array([0.4, 0.4, 0.8])
)

print(f"Exclusion zones: {checker.get_num_exclude_zones()}")

# Clear all obstacles
checker.clear_exclude_zones()
print(f"After clearing: {checker.get_num_exclude_zones()} zones")
```

### Integration with Workspace Analysis

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import DimensionConstraint
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Define workspace constraints
constraints = DimensionConstraint(
    min_bounds=np.array([-0.8, -0.6, 0.0]),
    max_bounds=np.array([0.8, 0.6, 1.2]),
    exclude_zones=[
        # Table obstacle
        (np.array([0.2, 0.1, 0.0]), np.array([0.6, 0.4, 0.8]))
    ]
)

# Create checker
checker = WorkspaceConstraintChecker.from_config(constraints)

# Filter workspace points
workspace_points = np.random.uniform(-1, 1, size=(10000, 3))
valid_points = checker.filter_points(workspace_points)

print(f"Valid workspace points: {len(valid_points)}/{len(workspace_points)}")
print(f"Coverage: {len(valid_points)/len(workspace_points)*100:.1f}%")
```

## Best Practices

### Choosing Bounds

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Conservative workspace bounds
def create_safe_bounds(robot_reach, safety_margin=0.1):
    """Create workspace bounds with safety margins."""
    return {
        'min_bounds': np.array([-robot_reach + safety_margin] * 3),
        'max_bounds': np.array([robot_reach - safety_margin] * 3),
        'ground_height': safety_margin
    }

# Example usage
bounds = create_safe_bounds(robot_reach=1.2, safety_margin=0.1)
checker = WorkspaceConstraintChecker(**bounds)
```

### Validation

```python
import numpy as np

def validate_bounds(min_bounds, max_bounds):
    """Validate constraint bounds."""
    if min_bounds is not None and max_bounds is not None:
        if not np.all(min_bounds < max_bounds):
            raise ValueError("min_bounds must be less than max_bounds")
    
    for bounds in [min_bounds, max_bounds]:
        if bounds is not None and len(bounds) != 3:
            raise ValueError("Bounds must have exactly 3 dimensions")

# Use validation
min_bounds = np.array([-1, -1, 0])
max_bounds = np.array([1, 1, 2])
validate_bounds(min_bounds, max_bounds)
```

## API Reference

### WorkspaceConstraintChecker

Main constraint checker implementation.

#### Constructor Parameters

- **min_bounds**: `Optional[np.ndarray]` - Minimum bounds [x, y, z]
- **max_bounds**: `Optional[np.ndarray]` - Maximum bounds [x, y, z]
- **ground_height**: `float` - Ground plane height (default: 0.0)
- **exclude_zones**: `List[Tuple[np.ndarray, np.ndarray]]` - Exclusion zones
- **device**: `Optional[torch.device]` - PyTorch device

#### Methods

- **check_bounds(points)**: Check if points are within bounds
- **check_collision(points)**: Check if points avoid rectangular exclusion zones (simple collision)
- **filter_points(points)**: Filter points by all constraints
- **add_exclude_zone(min_bounds, max_bounds)**: Add exclusion zone
- **clear_exclude_zones()**: Remove all exclusion zones
- **get_num_exclude_zones()**: Get number of exclusion zones

#### Class Methods

- **from_config(config)**: Create checker from DimensionConstraint

### DimensionConstraint

Configuration dataclass for constraints.

#### Parameters

- **min_bounds**: `Optional[np.ndarray]` - Workspace minimum bounds
- **max_bounds**: `Optional[np.ndarray]` - Workspace maximum bounds
- **exclude_zones**: `List[Tuple[np.ndarray, np.ndarray]]` - Exclusion zones
- **ground_height**: `float` - Ground plane height (default: 0.0)
- **joint_limits_scale**: `float` - Joint limits scaling (TODO: not implemented)
- **enforce_collision_free**: `bool` - Advanced collision checking (TODO: not implemented)
- **self_collision_check**: `bool` - Self-collision check (TODO: not implemented)

**Note**: These advanced collision detection features are defined in the config but not yet used by WorkspaceConstraintChecker.

## Quick Start

### Basic Setup

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Simple boundary constraints
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1, -1, 0]),
    max_bounds=np.array([1, 1, 1])
)
```

### With Obstacles

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Workspace with exclusion zones
checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.2, -1.2, 0.05]),
    max_bounds=np.array([1.2, 1.2, 1.8]),
    exclude_zones=[
        # Table obstacle
        (np.array([0.3, 0.2, 0.0]), np.array([0.8, 0.7, 0.8]))
    ]
)
```

### Using Configuration (Recommended)

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.configs.dimension_constraint import DimensionConstraint
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker

# Create config
config = DimensionConstraint(
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 1.5]),
    exclude_zones=[
        (np.array([0.2, 0.2, 0.0]), np.array([0.4, 0.4, 0.8]))
    ]
)

# Create checker
checker = WorkspaceConstraintChecker.from_config(config)
```
