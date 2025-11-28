# Workspace Analyzer Metrics

The **Workspace Analyzer Metrics** module provides comprehensive metrics and analysis tools for evaluating robotic workspace characteristics. This module implements various quantitative measures to assess workspace quality, coverage, manipulability, and performance.

## Table of Contents

- [Overview](#overview)
- [Metric Types](#metric-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The metrics module enables quantitative workspace analysis through:

- **Reachability Analysis**: Measure workspace coverage and volume
- **Manipulability Assessment**: Evaluate dexterity throughout the workspace
- **Density Analysis**: Analyze point distribution and clustering
- **Performance Metrics**: Quantify workspace quality and efficiency
- **Comparative Analysis**: Compare different robot configurations

### Key Features

- **Comprehensive Metrics**: Multiple analysis dimensions for complete assessment
- **Scalable Computation**: Efficient algorithms for large datasets
- **Configurable Analysis**: Flexible parameters for different analysis needs
- **Statistical Insights**: Rich statistical analysis and reporting
- **Visualization Support**: Integration with visualization tools
- **Export Capabilities**: Multiple output formats for results

## Metric Types

### 1. Reachability Metrics

Analyze workspace coverage, volume, and reachability characteristics.

```python
from embodichain.lab.sim.utility.workspace_analyzer.metrics import ReachabilityMetric
from embodichain.lab.sim.utility.workspace_analyzer.configs import ReachabilityConfig

# Configure reachability analysis
config = ReachabilityConfig(
    voxel_size=0.01,           # 1cm resolution for volume calculation
    min_points_per_voxel=1,    # Minimum occupancy threshold
    compute_coverage=True      # Calculate coverage statistics
)

# Create metric instance
reachability = ReachabilityMetric(config)

# Analyze workspace points
results = reachability.compute(
    workspace_points=workspace_data,
    joint_configurations=joint_data
)
```

**Computed Metrics**:

- **Workspace Volume**: Total reachable volume in cubic meters
- **Coverage Ratio**: Percentage of bounding box that is reachable
- **Boundary Analysis**: Surface area and shape characteristics
- **Voxel Statistics**: Occupancy distribution and density
- **Reachability Zones**: Classification of workspace regions

**Applications**:

- Robot selection and comparison
- Workspace layout optimization
- Task feasibility analysis
- Performance benchmarking

### 2. Manipulability Metrics

Evaluate robot dexterity and manipulation capabilities throughout the workspace.

```python
from embodichain.lab.sim.utility.workspace_analyzer.metrics import ManipulabilityMetric
from embodichain.lab.sim.utility.workspace_analyzer.configs import ManipulabilityConfig

# Configure manipulability analysis
config = ManipulabilityConfig(
    jacobian_threshold=0.01,   # Minimum valid manipulability
    compute_isotropy=True,     # Calculate isotropy index
    compute_heatmap=True       # Generate spatial heatmap
)

# Create metric instance
manipulability = ManipulabilityMetric(config)

# Analyze with joint configurations
results = manipulability.compute(
    workspace_points=cartesian_points,
    joint_configurations=joint_configs,
    jacobians=jacobian_matrices  # Optional: pre-computed Jacobians
)
```

**Computed Metrics**:

- **Manipulability Index**: Measure of dexterous capability
- **Isotropy Index**: Uniformity of manipulation in all directions
- **Condition Number**: Jacobian conditioning and singularity proximity
- **Dexterity Distribution**: Spatial variation of manipulation capability
- **Singular Region Analysis**: Identification of poor dexterity areas

**Applications**:

- Task-specific robot evaluation
- Optimal pose selection
- Trajectory planning
- Dexterity optimization

### 3. Density Metrics

Analyze spatial distribution and clustering characteristics of workspace points.

```python
from embodichain.lab.sim.utility.workspace_analyzer.metrics import DensityMetric
from embodichain.lab.sim.utility.workspace_analyzer.configs import DensityConfig

# Configure density analysis
config = DensityConfig(
    radius=0.05,              # 5cm neighborhood radius
    k_neighbors=30,           # Consider 30 nearest neighbors
    compute_distribution=True  # Full distribution statistics
)

# Create metric instance
density = DensityMetric(config)

# Analyze point distribution
results = density.compute(
    workspace_points=workspace_points,
    weights=point_weights      # Optional: weighted analysis
)
```

**Computed Metrics**:

- **Local Density**: Point density in local neighborhoods
- **Global Distribution**: Overall spatial distribution patterns
- **Clustering Analysis**: Identification of dense and sparse regions
- **Uniformity Index**: Measure of distribution uniformity
- **Coverage Gaps**: Detection of under-sampled regions

**Applications**:

- Sampling strategy evaluation
- Workspace uniformity assessment
- Coverage gap analysis
- Adaptive sampling guidance

## Usage Examples

### Basic Metric Computation

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.metrics import *
from embodichain.lab.sim.utility.workspace_analyzer.configs import *

# Generate sample workspace data
workspace_points = np.random.uniform(-1, 1, size=(10000, 3))
joint_configs = np.random.uniform(-np.pi, np.pi, size=(10000, 6))

# Compute reachability metrics
reachability = ReachabilityMetric(ReachabilityConfig(voxel_size=0.02))
reach_results = reachability.compute(workspace_points)

print(f"Workspace volume: {reach_results['volume']:.4f} m³")
print(f"Coverage ratio: {reach_results['coverage_ratio']:.3f}")

# Compute density metrics
density = DensityMetric(DensityConfig(radius=0.1, k_neighbors=50))
density_results = density.compute(workspace_points)

print(f"Mean density: {density_results['mean_density']:.2f}")
print(f"Uniformity index: {density_results['uniformity_index']:.3f}")
```

### Comprehensive Workspace Analysis

```python
def comprehensive_workspace_analysis(robot, sampling_config):
    """Perform complete workspace analysis with all metrics."""
    
    # Generate workspace data
    joint_samples = generate_joint_samples(robot, sampling_config)
    workspace_points = []
    jacobians = []
    
    for joints in joint_samples:
        try:
            # Forward kinematics
            pose = robot.forward_kinematics(joints)
            workspace_points.append(pose[:3, 3])
            
            # Jacobian computation
            J = robot.compute_jacobian(joints)
            jacobians.append(J)
        except:
            continue
    
    workspace_points = np.array(workspace_points)
    jacobians = np.array(jacobians)
    
    # Initialize all metrics
    metrics = {
        'reachability': ReachabilityMetric(ReachabilityConfig(
            voxel_size=0.015,
            compute_coverage=True
        )),
        'manipulability': ManipulabilityMetric(ManipulabilityConfig(
            jacobian_threshold=0.001,
            compute_isotropy=True,
            compute_heatmap=True
        )),
        'density': DensityMetric(DensityConfig(
            radius=0.08,
            k_neighbors=40,
            compute_distribution=True
        ))
    }
    
    # Compute all metrics
    results = {}
    for name, metric in metrics.items():
        try:
            if name == 'manipulability':
                results[name] = metric.compute(
                    workspace_points=workspace_points,
                    joint_configurations=joint_samples,
                    jacobians=jacobians
                )
            else:
                results[name] = metric.compute(workspace_points=workspace_points)
        except Exception as e:
            print(f"Failed to compute {name}: {e}")
            results[name] = None
    
    # Generate summary report
    summary = generate_analysis_summary(results)
    
    return {
        'detailed_results': results,
        'summary': summary,
        'workspace_points': workspace_points,
        'joint_configurations': joint_samples
    }

def generate_analysis_summary(results):
    """Generate human-readable analysis summary."""
    summary = {}
    
    if results['reachability']:
        r = results['reachability']
        summary['workspace_volume'] = f"{r['volume']:.4f} m³"
        summary['coverage_percentage'] = f"{r['coverage_ratio']*100:.1f}%"
    
    if results['manipulability']:
        m = results['manipulability']
        summary['avg_manipulability'] = f"{m['mean_manipulability']:.3f}"
        summary['dexterity_uniformity'] = f"{m['isotropy_index']:.3f}"
    
    if results['density']:
        d = results['density']
        summary['point_density'] = f"{d['mean_density']:.2f}"
        summary['distribution_uniformity'] = f"{d['uniformity_index']:.3f}"
    
    return summary

# Usage
analysis_results = comprehensive_workspace_analysis(my_robot, sampling_config)
print("Workspace Analysis Summary:")
for key, value in analysis_results['summary'].items():
    print(f"  {key}: {value}")
```

### Comparative Analysis

```python
def compare_robot_workspaces(robots, configs, metric_types=['all']):
    """Compare workspace characteristics across multiple robots."""
    
    comparison_results = {}
    
    for robot_name, robot in robots.items():
        print(f"Analyzing {robot_name}...")
        
        # Generate workspace data for this robot
        workspace_data = generate_workspace_data(robot, configs[robot_name])
        
        # Compute requested metrics
        robot_metrics = {}
        
        if 'reachability' in metric_types or 'all' in metric_types:
            reachability = ReachabilityMetric(ReachabilityConfig())
            robot_metrics['reachability'] = reachability.compute(
                workspace_data['points']
            )
        
        if 'manipulability' in metric_types or 'all' in metric_types:
            manipulability = ManipulabilityMetric(ManipulabilityConfig())
            robot_metrics['manipulability'] = manipulability.compute(
                workspace_data['points'],
                workspace_data['joints'],
                workspace_data['jacobians']
            )
        
        if 'density' in metric_types or 'all' in metric_types:
            density = DensityMetric(DensityConfig())
            robot_metrics['density'] = density.compute(
                workspace_data['points']
            )
        
        comparison_results[robot_name] = robot_metrics
    
    # Generate comparison table
    comparison_table = create_comparison_table(comparison_results)
    
    return comparison_table

def create_comparison_table(results):
    """Create formatted comparison table."""
    table_data = []
    
    for robot_name, metrics in results.items():
        row = {'Robot': robot_name}
        
        if 'reachability' in metrics:
            r = metrics['reachability']
            row['Volume (m³)'] = f"{r['volume']:.4f}"
            row['Coverage (%)'] = f"{r['coverage_ratio']*100:.1f}"
        
        if 'manipulability' in metrics:
            m = metrics['manipulability']
            row['Avg Manipulability'] = f"{m['mean_manipulability']:.3f}"
            row['Isotropy Index'] = f"{m['isotropy_index']:.3f}"
        
        if 'density' in metrics:
            d = metrics['density']
            row['Point Density'] = f"{d['mean_density']:.2f}"
            row['Uniformity'] = f"{d['uniformity_index']:.3f}"
        
        table_data.append(row)
    
    return table_data

# Usage example
robots = {
    'UR5': ur5_robot,
    'UR10': ur10_robot,
    'Franka': franka_robot
}

configs = {
    'UR5': standard_sampling_config,
    'UR10': high_resolution_config,
    'Franka': standard_sampling_config
}

comparison = compare_robot_workspaces(robots, configs)
print("Robot Workspace Comparison:")
for row in comparison:
    print(f"{row['Robot']:>10}: Vol={row['Volume (m³)']:>8}, "
          f"Manip={row['Avg Manipulability']:>6}, "
          f"Uniform={row['Uniformity']:>6}")
```

### Performance Monitoring and Optimization

```python
class MetricsProfiler:
    """Profile metric computation performance."""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
    
    def profile_metric(self, metric, workspace_points, **kwargs):
        """Profile a single metric computation."""
        import time
        import tracemalloc
        
        metric_name = metric.__class__.__name__
        
        # Start monitoring
        tracemalloc.start()
        start_time = time.time()
        
        # Compute metric
        try:
            results = metric.compute(workspace_points, **kwargs)
            success = True
        except Exception as e:
            results = None
            success = False
            error = str(e)
        
        # Record performance
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.timing_data[metric_name] = end_time - start_time
        self.memory_data[metric_name] = peak / 1024 / 1024  # MB
        
        return {
            'results': results,
            'success': success,
            'computation_time': self.timing_data[metric_name],
            'peak_memory_mb': self.memory_data[metric_name],
            'error': error if not success else None
        }
    
    def profile_all_metrics(self, workspace_points, joint_configs=None):
        """Profile all available metrics."""
        metrics = [
            ReachabilityMetric(),
            ManipulabilityMetric(),
            DensityMetric()
        ]
        
        profile_results = {}
        
        for metric in metrics:
            kwargs = {}
            if isinstance(metric, ManipulabilityMetric) and joint_configs is not None:
                kwargs['joint_configurations'] = joint_configs
            
            profile_results[metric.__class__.__name__] = self.profile_metric(
                metric, workspace_points, **kwargs
            )
        
        return profile_results
    
    def generate_performance_report(self, profile_results):
        """Generate performance analysis report."""
        report = []
        report.append("Metrics Performance Report")
        report.append("=" * 50)
        
        for metric_name, data in profile_results.items():
            if data['success']:
                report.append(f"{metric_name}:")
                report.append(f"  Computation time: {data['computation_time']:.3f}s")
                report.append(f"  Peak memory: {data['peak_memory_mb']:.1f} MB")
            else:
                report.append(f"{metric_name}: FAILED - {data['error']}")
        
        return "\n".join(report)

# Usage
profiler = MetricsProfiler()

# Profile with different dataset sizes
dataset_sizes = [1000, 10000, 50000, 100000]
performance_data = {}

for size in dataset_sizes:
    test_points = np.random.uniform(-1, 1, size=(size, 3))
    test_joints = np.random.uniform(-np.pi, np.pi, size=(size, 6))
    
    profile_results = profiler.profile_all_metrics(test_points, test_joints)
    performance_data[size] = profile_results
    
    print(f"\nDataset size: {size}")
    print(profiler.generate_performance_report(profile_results))
```

## Best Practices

### Metric Selection and Configuration

```python
def choose_optimal_metrics(analysis_goal, dataset_size, computational_budget):
    """Choose appropriate metrics based on analysis requirements."""
    
    recommendations = {
        'configs': {},
        'metrics': []
    }
    
    # Base recommendations by analysis goal
    if analysis_goal == 'robot_comparison':
        recommendations['metrics'] = ['reachability', 'manipulability']
        recommendations['configs']['reachability'] = ReachabilityConfig(
            voxel_size=0.02,  # Moderate resolution for comparison
            compute_coverage=True
        )
    
    elif analysis_goal == 'task_feasibility':
        recommendations['metrics'] = ['reachability', 'density']
        recommendations['configs']['density'] = DensityConfig(
            radius=0.1,       # Task-relevant neighborhood
            compute_distribution=True
        )
    
    elif analysis_goal == 'sampling_optimization':
        recommendations['metrics'] = ['density']
        recommendations['configs']['density'] = DensityConfig(
            radius=0.05,      # Fine-grained analysis
            k_neighbors=50,
            compute_distribution=True
        )
    
    # Adjust for dataset size
    if dataset_size > 100000:
        # Use coarser settings for large datasets
        if 'reachability' in recommendations['configs']:
            recommendations['configs']['reachability'].voxel_size *= 1.5
        if 'density' in recommendations['configs']:
            recommendations['configs']['density'].radius *= 1.2
    
    # Adjust for computational budget
    if computational_budget == 'low':
        # Remove expensive computations
        for config in recommendations['configs'].values():
            if hasattr(config, 'compute_heatmap'):
                config.compute_heatmap = False
            if hasattr(config, 'compute_distribution'):
                config.compute_distribution = False
    
    return recommendations
```

### Error Handling and Validation

```python
class RobustMetricsComputation:
    """Robust metric computation with comprehensive error handling."""
    
    @staticmethod
    def validate_input_data(workspace_points, joint_configs=None):
        """Validate input data quality and format."""
        issues = []
        
        # Check workspace points
        if not isinstance(workspace_points, np.ndarray):
            issues.append("workspace_points must be numpy array")
        elif workspace_points.ndim != 2 or workspace_points.shape[1] != 3:
            issues.append("workspace_points must be Nx3 array")
        elif not np.isfinite(workspace_points).all():
            issues.append("workspace_points contains non-finite values")
        
        # Check joint configurations if provided
        if joint_configs is not None:
            if not isinstance(joint_configs, np.ndarray):
                issues.append("joint_configurations must be numpy array")
            elif joint_configs.shape[0] != workspace_points.shape[0]:
                issues.append("joint_configurations must match workspace_points length")
        
        return issues
    
    @staticmethod
    def safe_compute_metrics(metric, workspace_points, **kwargs):
        """Safely compute metrics with fallback strategies."""
        try:
            # Validate inputs
            issues = RobustMetricsComputation.validate_input_data(
                workspace_points, 
                kwargs.get('joint_configurations')
            )
            
            if issues:
                return {'success': False, 'errors': issues, 'results': None}
            
            # Attempt computation
            results = metric.compute(workspace_points, **kwargs)
            return {'success': True, 'errors': None, 'results': results}
            
        except MemoryError:
            # Handle memory issues with batch processing
            return RobustMetricsComputation.compute_with_batching(
                metric, workspace_points, **kwargs
            )
            
        except Exception as e:
            return {
                'success': False, 
                'errors': [f"Computation failed: {str(e)}"], 
                'results': None
            }
    
    @staticmethod
    def compute_with_batching(metric, workspace_points, batch_size=10000, **kwargs):
        """Compute metrics using batch processing for memory efficiency."""
        try:
            num_points = len(workspace_points)
            batch_results = []
            
            for i in range(0, num_points, batch_size):
                batch_points = workspace_points[i:i+batch_size]
                
                # Prepare batch kwargs
                batch_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, np.ndarray) and len(value) == num_points:
                        batch_kwargs[key] = value[i:i+batch_size]
                    else:
                        batch_kwargs[key] = value
                
                # Compute batch
                batch_result = metric.compute(batch_points, **batch_kwargs)
                batch_results.append(batch_result)
            
            # Combine batch results
            combined_results = RobustMetricsComputation.combine_batch_results(
                batch_results, metric.__class__.__name__
            )
            
            return {'success': True, 'errors': None, 'results': combined_results}
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Batch computation failed: {str(e)}"],
                'results': None
            }
```

## API Reference

### BaseMetric

Abstract base class for all workspace metrics.

#### BaseMetric Parameters

- **config**: `Optional[Any]` - Configuration object for the metric

#### BaseMetric Methods

- **compute(workspace_points, joint_configurations=None, **kwargs) -> Dict[str, Any]**: Compute metric
- **reset() -> None**: Reset metric results
- **get_results() -> Dict[str, Any]**: Get computed results

### ReachabilityMetric

Workspace reachability and coverage analysis.

#### Computed Results

- **volume**: `float` - Total workspace volume in cubic meters
- **coverage_ratio**: `float` - Ratio of occupied to total bounding box volume
- **voxel_count**: `int` - Number of occupied voxels
- **surface_area**: `float` - Approximate workspace surface area
- **bounding_box**: `Dict` - Workspace bounding box dimensions

### ManipulabilityMetric

Robot manipulability and dexterity analysis.

#### ManipulabilityMetric Results

- **mean_manipulability**: `float` - Average manipulability index
- **manipulability_std**: `float` - Standard deviation of manipulability
- **isotropy_index**: `float` - Measure of directional uniformity
- **condition_numbers**: `np.ndarray` - Jacobian condition numbers
- **singular_regions**: `List` - Regions with poor manipulability

### DensityMetric

Spatial distribution and density analysis.

#### DensityMetric Results

- **mean_density**: `float` - Average local point density
- **density_std**: `float` - Standard deviation of density
- **uniformity_index**: `float` - Distribution uniformity measure
- **cluster_analysis**: `Dict` - Clustering statistics
- **coverage_gaps**: `List` - Under-sampled regions

## Configuration Examples

### Quick Analysis Configuration

```python
# Fast analysis for initial evaluation
quick_reachability = ReachabilityMetric(ReachabilityConfig(
    voxel_size=0.05,          # Coarse resolution
    compute_coverage=False     # Skip expensive computations
))

quick_density = DensityMetric(DensityConfig(
    radius=0.1,               # Large neighborhoods
    k_neighbors=20,           # Fewer neighbors
    compute_distribution=False # Skip detailed statistics
))
```

### Detailed Research Configuration

```python
# Comprehensive analysis for research
research_reachability = ReachabilityMetric(ReachabilityConfig(
    voxel_size=0.005,         # High resolution
    min_points_per_voxel=2,   # Stricter occupancy
    compute_coverage=True      # Full analysis
))

research_manipulability = ManipulabilityMetric(ManipulabilityConfig(
    jacobian_threshold=0.001,  # Strict singularity threshold
    compute_isotropy=True,     # Detailed dexterity analysis
    compute_heatmap=True       # Spatial visualization data
))
```
