# Workspace Analyzer Caches

The **Workspace Analyzer Caches** module provides efficient caching mechanisms for workspace analysis data. This module implements various caching strategies to optimize performance during large-scale workspace sampling and analysis operations.

## Table of Contents

- [Overview](#overview)
- [Cache Architecture](#cache-architecture)
- [Cache Types](#cache-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The caches module addresses the performance challenges of workspace analysis by providing:

- **Memory Management**: Efficient handling of large pose datasets
- **Performance Optimization**: Reduce computation time through intelligent caching
- **Storage Flexibility**: Multiple storage backends (memory, disk)
- **Batch Processing**: Optimized batch operations for large datasets
- **Resource Control**: Configurable memory usage and cleanup strategies

### Key Features

- **Multiple Cache Types**: Memory and disk-based caching strategies
- **Automatic Management**: Smart cache management with configurable thresholds
- **Batch Operations**: Efficient batch processing for large datasets
- **Memory Optimization**: Built-in garbage collection and memory management
- **Persistent Storage**: Disk caching for long-term storage and reuse
- **Flexible Interface**: Uniform API across different cache implementations

## Cache Architecture

The module follows a clean architectural design:

```text
caches/
├── __init__.py           # Module exports
├── base_cache.py         # Abstract base cache interface
├── memory_cache.py       # In-memory caching implementation
├── disk_cache.py         # Persistent disk caching
├── cache_manager.py      # High-level cache management
└── cache_utils.py        # Utility functions and helpers
```

### Cache Hierarchy

```text
BaseCache (Abstract Base Class)
├── MemoryCache           # Fast in-memory storage
├── DiskCache            # Persistent disk storage
└── CacheManager         # Orchestrates multiple cache types
```

## Cache Types

### 1. Memory Cache

**Best for**: Fast access, moderate datasets, temporary storage

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache

cache = MemoryCache(
    batch_size=5000,        # Process in batches of 5000
    save_threshold=1000000  # Trigger GC every 1M samples
)
```

**Advantages**:

- Fastest access times
- No I/O overhead
- Simple implementation
- Automatic garbage collection

**Considerations**:

- Memory usage grows with dataset size
- Data lost on process termination
- RAM limitations for very large datasets

**Use Cases**:

- Interactive analysis sessions
- Moderate-sized workspace datasets
- Temporary computation results
- Real-time analysis updates

### 2. Disk Cache

**Best for**: Large datasets, persistent storage, memory constraints

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import DiskCache
from pathlib import Path

cache = DiskCache(
    cache_dir=Path("./workspace_cache"),
    batch_size=10000,
    compression=True,
    max_cache_size_mb=5000
)
```

**Advantages**:

- Persistent across sessions
- Handles very large datasets
- Configurable compression
- Automatic size management

**Considerations**:

- Slower access than memory
- Disk I/O overhead
- Storage space requirements

**Use Cases**:

- Large-scale workspace analysis
- Long-term result storage
- Memory-constrained environments
- Batch processing workflows

### 3. Cache Manager

**Best for**: Hybrid strategies, automatic optimization, complex workflows

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import CacheManager

manager = CacheManager(
    use_memory_cache=True,
    use_disk_cache=True,
    memory_limit_mb=2000,
    disk_cache_dir=Path("./cache")
)
```

**Advantages**:

- Combines multiple cache types
- Automatic cache selection
- Memory overflow handling
- Intelligent data placement

**Use Cases**:

- Production environments
- Variable dataset sizes
- Automatic optimization
- Complex analysis pipelines

## Usage Examples

### Basic Memory Caching

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache

# Create memory cache
cache = MemoryCache(batch_size=1000)

# Generate sample poses
poses = []
for i in range(5000):
    pose = np.eye(4)
    pose[:3, 3] = np.random.rand(3)  # Random translation
    poses.append(pose)

# Add poses to cache in batches
batch_size = 1000
for i in range(0, len(poses), batch_size):
    batch = poses[i:i + batch_size]
    cache.add(batch)

# Retrieve all cached data
all_poses = cache.get_all()
print(f"Cached {len(all_poses)} poses")

# Clear cache when done
cache.clear()
```

### Persistent Disk Caching

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import DiskCache
from pathlib import Path
import numpy as np

# Create disk cache with compression
cache = DiskCache(
    cache_dir=Path("./analysis_cache"),
    batch_size=2000,
    compression=True,
    max_cache_size_mb=1000
)

# Process large dataset in batches
def process_workspace_data(robot_config):
    cache_key = f"workspace_{robot_config.name}_{robot_config.hash}"
    
    # Check if results already cached
    if cache.has_key(cache_key):
        print("Loading from cache...")
        return cache.get(cache_key)
    
    # Generate new workspace data
    poses = generate_workspace_poses(robot_config)
    
    # Cache results for future use
    cache.add(poses, key=cache_key)
    cache.flush()
    
    return poses

# Usage
robot_configs = [config1, config2, config3]
results = []
for config in robot_configs:
    workspace_data = process_workspace_data(config)
    results.append(workspace_data)
```

### Advanced Cache Management

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import CacheManager
from pathlib import Path

# Create hybrid cache manager
manager = CacheManager(
    use_memory_cache=True,
    use_disk_cache=True,
    memory_limit_mb=1500,        # 1.5GB memory limit
    disk_cache_dir=Path("./cache"),
    auto_cleanup=True
)

# Configure cache behavior
manager.set_memory_threshold(0.8)    # Switch to disk at 80% memory
manager.set_compression(True)        # Enable compression for disk cache
manager.set_cleanup_interval(3600)   # Cleanup every hour

# Use manager for large-scale analysis
def analyze_robot_workspace(robot, sampling_config):
    cache_key = manager.generate_key(robot, sampling_config)
    
    # Check cache first
    if manager.has_cached(cache_key):
        return manager.get_cached(cache_key)
    
    # Generate new data
    poses = sample_robot_workspace(robot, sampling_config)
    
    # Cache with automatic storage selection
    manager.cache_data(cache_key, poses)
    
    return poses

# Batch processing with automatic cache management
robots = [robot1, robot2, robot3, robot4]
configs = [config_low, config_medium, config_high]

results = {}
for robot in robots:
    for config in configs:
        key = f"{robot.name}_{config.resolution}"
        results[key] = analyze_robot_workspace(robot, config)

# Manager automatically handles memory/disk decisions
print(f"Memory usage: {manager.get_memory_usage_mb():.1f} MB")
print(f"Disk usage: {manager.get_disk_usage_mb():.1f} MB")
```

### Cache Performance Monitoring

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache
import time

# Create cache with monitoring
cache = MemoryCache(batch_size=5000)

# Performance measurement
start_time = time.time()
total_poses = 0

# Add data and measure performance
for batch_num in range(100):  # 100 batches
    # Generate batch of poses
    batch = [np.random.rand(4, 4) for _ in range(1000)]
    
    # Measure cache operation time
    batch_start = time.time()
    cache.add(batch)
    batch_time = time.time() - batch_start
    
    total_poses += len(batch)
    
    if batch_num % 10 == 0:
        print(f"Batch {batch_num}: {len(batch)} poses in {batch_time:.3f}s")

# Final statistics
total_time = time.time() - start_time
print(f"\nTotal: {total_poses} poses in {total_time:.2f}s")
print(f"Rate: {total_poses / total_time:.0f} poses/second")
print(f"Memory usage: {cache.total_processed} poses")
```

## Best Practices

### Cache Selection Guidelines

```python
def choose_cache_strategy(dataset_size, memory_available, persistence_needed):
    """Choose optimal cache strategy based on requirements."""
    
    if dataset_size < 50000 and memory_available > 2000:  # Small dataset, plenty RAM
        return "memory"
    elif persistence_needed or dataset_size > 200000:      # Large or persistent
        return "disk"
    else:                                                   # Hybrid approach
        return "managed"

# Example usage
strategy = choose_cache_strategy(
    dataset_size=100000,
    memory_available=4000,  # MB
    persistence_needed=True
)
```

### Memory Management

```python
# Memory-conscious caching
class MemoryAwareCaching:
    def __init__(self, max_memory_mb=1000):
        self.max_memory_mb = max_memory_mb
        self.cache = MemoryCache(batch_size=2000)
        
    def add_with_monitoring(self, poses):
        # Check memory usage before adding
        current_memory = self.get_memory_usage()
        
        if current_memory + self.estimate_pose_memory(poses) > self.max_memory_mb:
            # Flush to disk or clear old data
            self.cache.flush()
            if hasattr(self.cache, 'clear_oldest'):
                self.cache.clear_oldest(0.3)  # Clear 30% of oldest data
        
        self.cache.add(poses)
    
    def get_memory_usage(self):
        # Estimate current memory usage
        return len(self.cache) * 64 * 4 / (1024 * 1024)  # Rough estimate
    
    def estimate_pose_memory(self, poses):
        return len(poses) * 64 * 4 / (1024 * 1024)  # 4x4 float64 matrix
```

### Cache Invalidation

```python
# Smart cache invalidation
class SmartCache:
    def __init__(self):
        self.cache = {}
        self.dependencies = {}
        self.timestamps = {}
    
    def cache_with_dependencies(self, key, data, depends_on=None):
        self.cache[key] = data
        self.dependencies[key] = depends_on or []
        self.timestamps[key] = time.time()
    
    def invalidate_dependent(self, changed_key):
        """Invalidate all caches that depend on changed_key."""
        to_invalidate = []
        for key, deps in self.dependencies.items():
            if changed_key in deps:
                to_invalidate.append(key)
        
        for key in to_invalidate:
            self.invalidate(key)
    
    def invalidate(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.dependencies[key]
            del self.timestamps[key]
```

### Performance Optimization

```python
# Batch optimization strategies
def optimize_batch_size(cache_type, data_characteristics):
    """Determine optimal batch size based on cache type and data."""
    
    base_sizes = {
        "memory": 5000,
        "disk": 10000,
        "hybrid": 7500
    }
    
    # Adjust based on data characteristics
    multiplier = 1.0
    
    if data_characteristics.get("pose_complexity", "simple") == "complex":
        multiplier *= 0.7  # Smaller batches for complex poses
    
    if data_characteristics.get("memory_pressure", "low") == "high":
        multiplier *= 0.5  # Much smaller batches under memory pressure
    
    return int(base_sizes[cache_type] * multiplier)

# Usage
batch_size = optimize_batch_size(
    "memory",
    {"pose_complexity": "simple", "memory_pressure": "low"}
)
```

## API Reference

### BaseCache

Abstract base class defining the cache interface.

#### Constructor Parameters

- **batch_size**: `int` - Number of samples per batch (default: 5000)
- **save_threshold**: `int` - Threshold for triggering save/flush operations (default: 10000000)

#### Methods

- **add(poses: List[np.ndarray]) -> None**: Add pose samples to cache
- **flush() -> None**: Flush any pending data
- **get_all() -> Optional[List[np.ndarray]]**: Retrieve all cached poses
- **clear() -> None**: Clear all cached data
- **total_processed** (property): Total number of processed poses

### MemoryCache

In-memory caching implementation.

#### MemoryCache Parameters

- **batch_size**: `int` - Batch size for processing (default: 5000)
- **save_threshold**: `int` - GC trigger threshold (default: 10000000)

#### Additional Methods

- **\_\_len\_\_() -> int**: Returns number of cached poses

#### Performance Characteristics

- **Access Time**: O(1) for retrieval, O(1) for addition
- **Memory Usage**: Linear with number of poses
- **Persistence**: None (data lost on process exit)

### DiskCache

Persistent disk-based caching implementation.

#### DiskCache Parameters

- **cache_dir**: `Path` - Directory for cache files
- **batch_size**: `int` - Batch size for I/O operations (default: 10000)
- **compression**: `bool` - Enable file compression (default: True)
- **max_cache_size_mb**: `int` - Maximum cache directory size (default: 5000)

#### DiskCache Methods

- **has_key(key: str) -> bool**: Check if key exists in cache
- **get(key: str) -> Optional[List[np.ndarray]]**: Retrieve data by key
- **set_compression(enabled: bool) -> None**: Enable/disable compression

#### DiskCache Performance

- **Access Time**: O(log n) for retrieval, O(1) amortized for addition
- **Memory Usage**: Constant (configurable buffer size)
- **Persistence**: Full (survives process restarts)

### CacheManager

High-level cache management and orchestration.

#### CacheManager Parameters

- **use_memory_cache**: `bool` - Enable memory caching (default: True)
- **use_disk_cache**: `bool` - Enable disk caching (default: False)
- **memory_limit_mb**: `int` - Memory usage limit (default: 1000)
- **disk_cache_dir**: `Path` - Disk cache directory

#### Management Methods

- **set_memory_threshold(threshold: float) -> None**: Set memory usage threshold
- **set_cleanup_interval(seconds: int) -> None**: Configure automatic cleanup
- **get_memory_usage_mb() -> float**: Current memory usage
- **get_disk_usage_mb() -> float**: Current disk usage
- **cleanup_old_data(age_hours: int) -> None**: Remove old cached data

## Cache Configuration Examples

### Development Configuration

```python
# Fast development setup
dev_cache = MemoryCache(
    batch_size=1000,      # Small batches for quick feedback
    save_threshold=50000  # Low threshold for frequent cleanup
)
```

### Production Configuration

```python
# Production-optimized setup
prod_manager = CacheManager(
    use_memory_cache=True,
    use_disk_cache=True,
    memory_limit_mb=4000,              # 4GB memory limit
    disk_cache_dir=Path("/fast_ssd/cache"),
    auto_cleanup=True
)

prod_manager.set_memory_threshold(0.85)  # Use 85% of memory before disk
prod_manager.set_cleanup_interval(7200)  # Cleanup every 2 hours
```

### Research Configuration

```python
# Long-term research storage
research_cache = DiskCache(
    cache_dir=Path("./research_results"),
    batch_size=20000,              # Large batches for efficiency
    compression=True,              # Save storage space
    max_cache_size_mb=50000       # 50GB cache limit
)
```
