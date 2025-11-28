# Workspace Analyzer Constraints

The **Workspace Analyzer Constraints** module provides comprehensive constraint checking and validation for robotic workspace analysis. This module ensures that workspace analysis respects physical limitations, safety boundaries, and operational constraints.

## Table of Contents

- [Overview](#overview)
- [Constraint Types](#constraint-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The constraints module enables robust workspace analysis by enforcing:

- **Spatial Boundaries**: Define workspace limits and exclusion zones
- **Physical Constraints**: Respect robot joint limits and reachability
- **Safety Zones**: Implement collision avoidance and safety margins
- **Ground Constraints**: Handle floor planes and elevation limits
- **Custom Constraints**: Extensible framework for application-specific limits

### Key Features

- **Flexible Constraint Definition**: Support for complex geometric constraints
- **High Performance**: Vectorized operations for large point sets
- **Multi-Backend Support**: Works with both NumPy arrays and PyTorch tensors
- **Extensible Design**: Easy to add custom constraint types
- **Safety First**: Built-in safety margins and validation
- **Real-time Filtering**: Efficient point filtering for interactive applications

## Constraint Types

### 1. Spatial Boundary Constraints

Define the operational workspace boundaries for the robot.

```python
from embodichain.lab.sim.utility.workspace_analyzer.constraints import WorkspaceConstraintChecker
import numpy as np

# Basic workspace boundaries
constraint_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.0, -1.0, 0.0]),  # [x_min, y_min, z_min]
    max_bounds=np.array([1.0, 1.0, 2.0]),    # [x_max, y_max, z_max]
    ground_height=0.0
)
```

**Applications**:

- Define robot's maximum reach envelope
- Set table/workspace surface limits
- Implement safety perimeters
- Restrict vertical working range

### 2. Exclusion Zone Constraints

Specify regions that the robot must avoid.

```python
# Workspace with exclusion zones
constraint_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-2.0, -1.5, 0.0]),
    max_bounds=np.array([2.0, 1.5, 2.0]),
    exclude_zones=[
        # Obstacle 1: Table
        (np.array([0.3, -0.5, 0.0]), np.array([1.2, 0.5, 0.8])),
        # Obstacle 2: Wall section
        (np.array([1.8, -1.5, 0.0]), np.array([2.0, 1.5, 2.0]))
    ]
)
```

**Applications**:

- Avoid static obstacles
- Implement safety zones around humans
- Protect sensitive equipment
- Define no-go areas

### 3. Ground Plane Constraints

Handle floor constraints and elevation limits.

```python
# Elevated workspace with ground constraints
constraint_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.5, -1.5, 0.1]),  # 10cm above ground
    max_bounds=np.array([1.5, 1.5, 2.5]),
    ground_height=0.0,
    enforce_ground_clearance=True
)
```

**Applications**:

- Prevent ground collisions
- Maintain minimum clearance
- Handle elevated platforms
- Implement floor safety margins

### 4. Joint Limit Constraints

Respect robot's physical joint limitations.

```python
# Joint-aware constraint checking
constraint_checker = WorkspaceConstraintChecker(
    joint_limits_scale=0.95,        # Use 95% of joint range
    enforce_joint_limits=True,
    self_collision_check=True
)
```

**Applications**:

- Enforce mechanical joint limits
- Prevent over-extension
- Avoid singular configurations
- Check self-collision potential

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

# Generate test points
test_points = np.array([
    [0.5, 0.5, 0.5],    # Valid point
    [2.0, 0.0, 0.5],    # Outside x boundary
    [0.0, 0.0, -0.1],   # Below ground
    [0.8, 0.8, 1.2]     # Valid point
])

# Check which points satisfy constraints
valid_mask = checker.check_bounds(test_points)
print(f"Valid points: {valid_mask}")  # [True, False, False, True]

# Filter to keep only valid points
valid_points = checker.filter_points(test_points)
print(f"Filtered points shape: {valid_points.shape}")  # (2, 3)
```

### Complex Workspace with Obstacles

```python
# Define complex workspace environment
complex_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-2.0, -1.5, 0.05]),
    max_bounds=np.array([2.0, 1.5, 2.5]),
    ground_height=0.0,
    exclude_zones=[
        # Central table
        (np.array([-0.5, -0.8, 0.0]), np.array([0.5, 0.8, 0.9])),
        # Left wall segment
        (np.array([-2.0, 1.3, 0.0]), np.array([-1.7, 1.5, 2.5])),
        # Overhead beam
        (np.array([-1.0, -0.2, 2.2]), np.array([1.0, 0.2, 2.5]))
    ]
)

# Test large point cloud
num_points = 100000
random_points = np.random.uniform(
    low=[-2.5, -2.0, -0.5], 
    high=[2.5, 2.0, 3.0], 
    size=(num_points, 3)
)

# Filter points efficiently
start_time = time.time()
valid_points = complex_checker.filter_points(random_points)
filter_time = time.time() - start_time

print(f"Filtered {num_points} points in {filter_time:.3f}s")
print(f"Valid points: {len(valid_points)}/{num_points} ({100*len(valid_points)/num_points:.1f}%)")
```

### Dynamic Constraint Updates

```python
# Adaptive constraint checking for changing environments
class DynamicWorkspaceChecker:
    def __init__(self, base_bounds):
        self.base_checker = WorkspaceConstraintChecker(
            min_bounds=base_bounds[0],
            max_bounds=base_bounds[1]
        )
        self.dynamic_obstacles = []
    
    def add_temporary_obstacle(self, obstacle_bounds, duration):
        """Add temporary obstacle that expires after duration."""
        obstacle = {
            'bounds': obstacle_bounds,
            'expires': time.time() + duration
        }
        self.dynamic_obstacles.append(obstacle)
        self.update_constraints()
    
    def update_constraints(self):
        """Update constraints based on current dynamic obstacles."""
        # Remove expired obstacles
        current_time = time.time()
        self.dynamic_obstacles = [
            obs for obs in self.dynamic_obstacles 
            if obs['expires'] > current_time
        ]
        
        # Update exclusion zones
        exclude_zones = [obs['bounds'] for obs in self.dynamic_obstacles]
        
        # Create new checker with updated constraints
        self.base_checker = WorkspaceConstraintChecker(
            min_bounds=self.base_checker.min_bounds,
            max_bounds=self.base_checker.max_bounds,
            exclude_zones=exclude_zones
        )
    
    def check_points(self, points):
        self.update_constraints()  # Ensure constraints are current
        return self.base_checker.filter_points(points)

# Usage example
dynamic_checker = DynamicWorkspaceChecker([
    np.array([-1.5, -1.5, 0.0]),
    np.array([1.5, 1.5, 2.0])
])

# Add temporary human safety zone
human_position = np.array([0.5, 0.0, 0.0])
safety_radius = 0.4
human_bounds = (
    human_position - safety_radius,
    human_position + safety_radius
)
dynamic_checker.add_temporary_obstacle(human_bounds, duration=30.0)  # 30 seconds

# Check points with dynamic constraints
workspace_points = generate_workspace_points()
safe_points = dynamic_checker.check_points(workspace_points)
```

### Performance Optimization

```python
# High-performance constraint checking for large datasets
class OptimizedConstraintChecker:
    def __init__(self, constraints_config):
        self.checker = WorkspaceConstraintChecker(**constraints_config)
        self.batch_size = 50000  # Process in batches
        
    def check_large_dataset(self, points, show_progress=True):
        """Efficiently check constraints for very large point sets."""
        total_points = len(points)
        valid_points_list = []
        
        for i in range(0, total_points, self.batch_size):
            batch = points[i:i + self.batch_size]
            
            # Process batch
            valid_batch = self.checker.filter_points(batch)
            valid_points_list.append(valid_batch)
            
            if show_progress and (i // self.batch_size) % 10 == 0:
                progress = (i + len(batch)) / total_points * 100
                print(f"Progress: {progress:.1f}%")
        
        # Combine results
        if valid_points_list:
            return np.vstack(valid_points_list)
        else:
            return np.empty((0, 3))

# Example with 1 million points
optimizer = OptimizedConstraintChecker({
    'min_bounds': np.array([-2.0, -2.0, 0.0]),
    'max_bounds': np.array([2.0, 2.0, 2.0]),
    'ground_height': 0.0
})

# Generate large dataset
large_dataset = np.random.uniform(-3, 3, size=(1000000, 3))

# Process efficiently
start_time = time.time()
filtered_results = optimizer.check_large_dataset(large_dataset)
process_time = time.time() - start_time

print(f"Processed 1M points in {process_time:.2f}s")
print(f"Rate: {1000000/process_time:.0f} points/second")
```

### Integration with Workspace Analysis

```python
# Complete workspace analysis with constraints
def analyze_constrained_workspace(robot, constraint_config, sampling_config):
    """Perform workspace analysis respecting all constraints."""
    
    # Create constraint checker
    checker = WorkspaceConstraintChecker(**constraint_config)
    
    # Generate workspace samples
    joint_samples = generate_joint_samples(robot, sampling_config)
    
    # Convert to Cartesian poses
    poses = []
    for joint_config in joint_samples:
        try:
            pose = robot.forward_kinematics(joint_config)
            poses.append(pose[:3, 3])  # Extract position
        except:
            continue  # Skip invalid configurations
    
    poses = np.array(poses)
    
    # Apply constraints
    valid_poses = checker.filter_points(poses)
    
    # Calculate metrics
    total_samples = len(poses)
    valid_samples = len(valid_poses)
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    # Calculate workspace volume (approximate)
    if len(valid_poses) > 100:
        # Use convex hull for volume estimation
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(valid_poses)
            workspace_volume = hull.volume
        except:
            workspace_volume = 0
    else:
        workspace_volume = 0
    
    return {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'coverage_ratio': coverage,
        'workspace_volume': workspace_volume,
        'valid_poses': valid_poses,
        'constraint_stats': checker.get_statistics()
    }

# Usage
constraint_config = {
    'min_bounds': np.array([-1.2, -1.2, 0.1]),
    'max_bounds': np.array([1.2, 1.2, 1.8]),
    'exclude_zones': [
        (np.array([0.3, -0.4, 0.0]), np.array([0.8, 0.4, 0.7]))
    ],
    'ground_height': 0.0
}

sampling_config = {
    'num_samples': 50000,
    'strategy': 'uniform'
}

results = analyze_constrained_workspace(my_robot, constraint_config, sampling_config)
print(f"Workspace coverage: {results['coverage_ratio']:.3f}")
print(f"Estimated volume: {results['workspace_volume']:.4f} m³")
```

## Best Practices

### Constraint Design Guidelines

```python
# Guidelines for effective constraint design
class ConstraintDesignGuidelines:
    @staticmethod
    def create_conservative_bounds(robot_reach, safety_margin=0.1):
        """Create conservative workspace bounds with safety margins."""
        return {
            'min_bounds': np.array([-robot_reach + safety_margin] * 3),
            'max_bounds': np.array([robot_reach - safety_margin] * 3),
            'ground_height': safety_margin
        }
    
    @staticmethod
    def validate_constraint_config(config):
        """Validate constraint configuration for common issues."""
        issues = []
        
        if 'min_bounds' in config and 'max_bounds' in config:
            if not np.all(config['min_bounds'] < config['max_bounds']):
                issues.append("min_bounds should be less than max_bounds")
        
        if 'exclude_zones' in config:
            for i, (zone_min, zone_max) in enumerate(config['exclude_zones']):
                if not np.all(zone_min < zone_max):
                    issues.append(f"Exclude zone {i} has invalid bounds")
        
        return issues
    
    @staticmethod
    def optimize_for_performance(config, expected_points):
        """Optimize configuration for expected point count."""
        optimized = config.copy()
        
        if expected_points > 100000:
            # Use simpler constraints for large datasets
            optimized['fast_mode'] = True
        
        return optimized
```

### Error Handling and Validation

```python
# Robust constraint checking with error handling
class RobustConstraintChecker:
    def __init__(self, config):
        self.config = self.validate_and_sanitize_config(config)
        self.checker = WorkspaceConstraintChecker(**self.config)
        
    def validate_and_sanitize_config(self, config):
        """Validate and sanitize configuration parameters."""
        sanitized = config.copy()
        
        # Ensure bounds are numpy arrays
        if 'min_bounds' in config and config['min_bounds'] is not None:
            sanitized['min_bounds'] = np.array(config['min_bounds'])
        
        if 'max_bounds' in config and config['max_bounds'] is not None:
            sanitized['max_bounds'] = np.array(config['max_bounds'])
        
        # Validate dimensions
        for key in ['min_bounds', 'max_bounds']:
            if key in sanitized and sanitized[key] is not None:
                if len(sanitized[key]) != 3:
                    raise ValueError(f"{key} must have exactly 3 dimensions")
        
        return sanitized
    
    def safe_filter_points(self, points):
        """Safely filter points with comprehensive error handling."""
        try:
            # Validate input
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("Points must be Nx3 array")
            
            # Check for invalid values
            if not np.isfinite(points).all():
                logger.log_warning("Removing non-finite points")
                finite_mask = np.isfinite(points).all(axis=1)
                points = points[finite_mask]
            
            # Apply constraints
            return self.checker.filter_points(points)
            
        except Exception as e:
            logger.log_error(f"Constraint checking failed: {e}")
            return np.empty((0, 3))  # Return empty array on failure
```

## API Reference

### IConstraintChecker

Interface protocol defining the constraint checker contract.

#### Methods

- **check_bounds(points) -> bool array**: Check which points satisfy bounds
- **filter_points(points) -> filtered array**: Keep only constraint-satisfying points

### BaseConstraintChecker

Abstract base class for constraint implementations.

#### BaseConstraintChecker Parameters

- **min_bounds**: `Optional[np.ndarray]` - Minimum workspace bounds [x, y, z]
- **max_bounds**: `Optional[np.ndarray]` - Maximum workspace bounds [x, y, z]  
- **ground_height**: `float` - Ground plane elevation (default: 0.0)
- **device**: `Optional[torch.device]` - PyTorch device for computations

#### BaseConstraintChecker Methods

- **check_bounds(points) -> Union[torch.Tensor, np.ndarray]**: Bounds checking
- **filter_points(points) -> Union[torch.Tensor, np.ndarray]**: Point filtering
- **validate_bounds() -> bool**: Validate constraint configuration

### WorkspaceConstraintChecker

Concrete implementation of workspace constraints.

#### WorkspaceConstraintChecker Parameters

- **min_bounds**: `Optional[np.ndarray]` - Workspace minimum bounds
- **max_bounds**: `Optional[np.ndarray]` - Workspace maximum bounds
- **exclude_zones**: `List[Tuple[np.ndarray, np.ndarray]]` - Exclusion zone list
- **ground_height**: `float` - Ground plane height (default: 0.0)
- **joint_limits_scale**: `float` - Joint range scaling factor (default: 1.0)
- **enforce_collision_free**: `bool` - Enable collision checking (default: False)
- **self_collision_check**: `bool` - Check self-collisions (default: False)

#### WorkspaceConstraintChecker Methods

- **add_exclusion_zone(min_bounds, max_bounds) -> None**: Add new exclusion zone
- **remove_exclusion_zone(index) -> None**: Remove exclusion zone by index
- **get_statistics() -> Dict**: Get constraint checking statistics
- **set_safety_margins(margins) -> None**: Configure safety margins

#### Performance Characteristics

- **Bounds Check**: O(1) per point for simple bounds
- **Exclusion Zones**: O(k) per point where k is number of zones
- **Memory Usage**: Constant overhead + input array size
- **Batch Processing**: Vectorized operations for efficiency

## Configuration Examples

### Minimal Configuration

```python
# Simplest constraint setup
simple_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1, -1, 0]),
    max_bounds=np.array([1, 1, 1])
)
```

### Production Configuration

```python
# Production environment with safety margins
production_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-1.2, -1.2, 0.05]),    # 5cm ground clearance
    max_bounds=np.array([1.2, 1.2, 1.8]),       # Conservative height limit
    exclude_zones=[
        # Safety zone around operator position
        (np.array([-0.3, 0.8, 0.0]), np.array([0.3, 1.2, 2.0]))
    ],
    ground_height=0.0,
    joint_limits_scale=0.95                       # 95% of joint range
)
```

### Research Configuration

```python
# Detailed research setup with comprehensive constraints
research_checker = WorkspaceConstraintChecker(
    min_bounds=np.array([-2.0, -2.0, -0.1]),
    max_bounds=np.array([2.0, 2.0, 2.5]),
    exclude_zones=[
        # Multiple obstacles for complex environment
        (np.array([0.5, -0.5, 0.0]), np.array([1.0, 0.5, 1.0])),
        (np.array([-1.0, 1.0, 0.0]), np.array([-0.5, 1.5, 1.5])),
    ],
    ground_height=-0.05,
    enforce_collision_free=True,
    self_collision_check=True
)
```
