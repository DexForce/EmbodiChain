# URDF Convex Decomposition Tool

The URDF Convex Decomposition Tool is a utility within EmbodiChain designed to automatically process URDF models for simulation. It handles the decomposition of complex visual meshes into convex collision geometries and provides capabilities for model scaling and inertia recomputation.

## Key Features

- **Automated Convex Decomposition**: Uses the CoACD algorithm to decompose concave meshes into multiple convex hulls, essential for stable physics simulation.
- **URDF Modification**: Automatically generates a new URDF file linking to the newly created convex collision meshes.
- **Inertia Handling**: Supports recomputing inertial properties (mass, center of mass, inertia tensor) based on the geometry.
- **Model Scaling**: Allows for scaling the entire robot model (geometry, joints, origins) by specified factors.

## Method 1: Python API Usage

The tool provides a high-level function `generate_urdf_collision_convexes` for programmatic access. This is recommended for integrating the decomposition process into larger pipelines or scripts.

**Parameters:**

- `urdf_path`: Path to the input URDF file.
- `output_urdf_name`: Filename for the output URDF.
- `max_convex_hull_num`: Maximum number of convex hulls to generate per mesh (default: 16).
- `recompute_inertia`: Whether to recalculate inertial properties (default: False).
- `scale`: Optional numpy array `[x, y, z]` to scale the model.

```python
from embodichain.toolkits.acd.urdf_modifider import generate_urdf_collision_convexes
import numpy as np

# Example: Decompose and Scale
generate_urdf_collision_convexes(
    urdf_path="./assets/robot.urdf",
    output_urdf_name="robot_processed.urdf",
    max_convex_hull_num=16,
    recompute_inertia=True,
    scale=np.array([1.0, 1.0, 1.0])
)
print("Convex decomposition and inertia update completed.")
```

## Method 2: Command Line Interface (CLI)

The tool can also be run directly from the terminal, which is useful for quick batch processing or standalone usage.

**Command Structure:**

```bash
python -m embodichain.toolkits.acd.urdf_modifider [OPTIONS]
```

### Argument Descriptions

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--urdf_path` | str | Required | Path to the source URDF file. |
| `--output_urdf_name` | str | `articulation_acd.urdf` | Name of the generated URDF file. |
| `--max_convex_hull_num` | int | 8 | Maximum number of convex hulls for decomposition. |
| `--recompute_inertia` | flag | False | If present, recomputes inertia based on mesh geometry. |
| `--scale` | float | None | Scale factors (x y z). Example: `--scale 1.5 1.5 1.5`. |

**Example Usage:**

```bash
# Basic decomposition
python -m embodichain.toolkits.acd.urdf_modifider \
    --urdf_path ./assets/my_robot.urdf \
    --output_urdf_name my_robot_convex.urdf \
    --max_convex_hull_num 16

# Decomposition with scaling and inertia recomputation
python -m embodichain.toolkits.acd.urdf_modifider \
    --urdf_path ./assets/my_robot.urdf \
    --output_urdf_name my_robot_scaled.urdf \
    --recompute_inertia \
    --scale 0.5 0.5 0.5
```
