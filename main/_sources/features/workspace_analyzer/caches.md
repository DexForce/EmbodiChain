# Workspace Analyzer Caches

The Workspace Analyzer Caches module provides simple and efficient caching mechanisms for storing and reusing workspace analysis data.

## Table of Contents

- [Overview](#overview)
- [Cache Types](#cache-types)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

The caches module provides simple caching mechanisms for workspace analysis data to improve performance:

- **Memory Cache**: Fast in-memory storage for small to medium datasets
- **Disk Cache**: Persistent storage for large datasets or long-term use
- **Simple API**: Easy-to-use interface for both cache types

### When to Use

- **Memory Cache**: When you have sufficient RAM and need fast access
- **Disk Cache**: When working with large datasets or need to persist results between sessions


## Cache Types

### Memory Cache

Fast in-memory caching for medium-sized datasets:

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache

# Create memory cache
cache = MemoryCache()

# Add data
positions = [...]  # Your pose data
cache.add(positions)

# Get all data
all_data = cache.get_all()
```

### Disk Cache

Persistent disk caching for large datasets:

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import DiskCache

# Create disk cache
cache = DiskCache(save_dir="./my_cache")

# Add data
positions = [...]  # Your pose data
cache.add(positions)
cache.flush()  # Save to disk

# Get data
all_data = cache.get_all()
```

### Cache Manager

Simplified factory pattern for creating caches:

```python
from embodichain.lab.sim.utility.workspace_analyzer.caches import CacheManager

# Create memory cache
memory_cache = CacheManager.create_cache("memory")

# Create disk cache
disk_cache = CacheManager.create_cache("disk", save_dir="./cache")
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache, DiskCache

# Memory cache example
memory_cache = MemoryCache()

# Add pose data
poses = [np.eye(4) for _ in range(1000)]  # Example data
memory_cache.add(poses)

# Get data
cached_poses = memory_cache.get_all()
print(f"Cached {len(cached_poses)} poses")

# Disk cache example
disk_cache = DiskCache(save_dir="./workspace_cache")
disk_cache.add(poses)
disk_cache.flush()  # Save to disk

# Reload data
reloaded_poses = disk_cache.get_all()
```

### Real-world Usage

```python
# Cache usage in workspace analysis
def analyze_workspace_with_cache(robot_config, use_cache=True):
    if use_cache:
        # Use disk cache to save results
        cache = DiskCache(save_dir=f"./cache_{robot_config.name}")
        
        # Check if cached data exists
        if cache.get_batch_count() > 0:
            print("Loading data from cache...")
            return cache.get_all()
    
    # Generate new workspace data
    print("Generating new workspace data...")
    poses = generate_workspace_poses(robot_config)
    
    if use_cache:
        # Save results
        cache.add(poses)
        cache.flush()
        print(f"Cached {len(poses)} poses to disk")
    
    return poses
```

## Best Practices

### Choosing Cache Type

- **Small datasets (< 100k poses)**: Use `MemoryCache`
- **Large datasets (> 100k poses)**: Use `DiskCache`  
- **Need persistence**: Use `DiskCache`
- **Temporary computation**: Use `MemoryCache`

### Simple Selection Function

```python
def choose_cache(data_size, need_persistence=False):
    """Choose cache type based on data size and requirements"""
    if need_persistence or data_size > 100000:
        return DiskCache()
    else:
        return MemoryCache()

# Usage example
cache = choose_cache(data_size=50000, need_persistence=True)
```



## API Reference

### Basic Methods

All cache classes support these basic operations:

```python
# Add data
cache.add(poses_list)

# Get all data
all_data = cache.get_all()

# Clear cache
cache.clear()

# Flush to disk (DiskCache only)
cache.flush()
```

### Creating Caches

```python
# Memory cache
memory_cache = MemoryCache()

# Disk cache
disk_cache = DiskCache(save_dir="./my_cache")

# Using factory method
cache = CacheManager.create_cache("memory")  # or "disk"
```



## Quick Start

```python
# Simplest usage
from embodichain.lab.sim.utility.workspace_analyzer.caches import MemoryCache, DiskCache

# For small datasets
cache = MemoryCache()

# For large datasets or persistence needed
cache = DiskCache()

# Add data
cache.add(your_poses)

# Get data
result = cache.get_all()
```
