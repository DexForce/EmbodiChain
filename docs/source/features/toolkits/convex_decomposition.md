# URDF Convex Decomposition

The URDF convex decomposition toolkit prepares mesh-based robot models for
physics simulation. It processes each URDF link, decomposes its collision mesh
into convex hulls with CoACD, writes the generated meshes beside the source
asset, and updates a copy of the URDF to reference them.

If a link has no collision geometry, the toolkit uses its visual mesh and
creates a collision element with the same origin. Primitive geometries and
links without usable meshes are left unchanged.

## Capabilities

- **Convex collision generation** — decomposes concave meshes into a
  configurable number of convex hulls for more stable collision detection.
- **URDF rewriting** — stores generated meshes under `Collision/` and updates
  the output URDF without replacing the source URDF.
- **Inertia recomputation** — optionally calculates mass, center of mass, and
  inertia from the generated collision mesh at unit density.
- **Model scaling** — optionally scales mesh geometry, link and joint origins,
  mass, and prismatic-joint limits.

```{note}
Mesh paths are resolved relative to the input URDF directory. Keep the
generated `Collision/` and `Scale/` directories together with the output URDF
when moving the processed asset.
```

## Python API

Use `generate_urdf_collision_convexes` when integrating asset preprocessing
into a Python pipeline.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `urdf_path` | `str` | required | Path to the source URDF. |
| `output_urdf_name` | `str` | required | Name of the output URDF, written in the source URDF directory. |
| `max_convex_hull_num` | `int` | `16` | Maximum hull count passed to CoACD for each mesh. |
| `recompute_inertia` | `bool` | `False` | Recompute inertial properties from collision geometry. |
| `scale` | sequence of 3 floats or `None` | `None` | Per-axis scale factors `[x, y, z]`. |

```python
import numpy as np

from embodichain.toolkits.acd.urdf_modifider import (
    generate_urdf_collision_convexes,
)

generate_urdf_collision_convexes(
    urdf_path="./assets/robot.urdf",
    output_urdf_name="robot_processed.urdf",
    max_convex_hull_num=16,
    recompute_inertia=True,
    scale=np.array([1.0, 1.0, 1.0]),
)
```

## Command-Line Interface

Run the module directly for one-off or batch asset preparation:

```bash
python -m embodichain.toolkits.acd.urdf_modifider [OPTIONS]
```

### Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--urdf_path` | `str` | required | Path to the source URDF. |
| `--output_urdf_name` | `str` | `articulation_acd.urdf` | Name of the generated URDF. |
| `--max_convex_hull_num` | `int` | `8` | Maximum hull count for each mesh. |
| `--recompute_inertia` | flag | disabled | Recompute inertia from generated collision geometry. |
| `--scale` | 3 floats | `None` | Per-axis scale factors, for example `--scale 1.5 1.5 1.5`. |

The Python API and CLI intentionally have different default hull limits: `16`
for the function and `8` for the CLI.

### Examples

```bash
# Generate convex collision meshes.
python -m embodichain.toolkits.acd.urdf_modifider \
    --urdf_path ./assets/my_robot.urdf \
    --output_urdf_name my_robot_convex.urdf \
    --max_convex_hull_num 16

# Scale the model and recompute inertia.
python -m embodichain.toolkits.acd.urdf_modifider \
    --urdf_path ./assets/my_robot.urdf \
    --output_urdf_name my_robot_scaled.urdf \
    --recompute_inertia \
    --scale 0.5 0.5 0.5
```

## Output Layout

For an input at `assets/robot.urdf`, the toolkit writes generated files relative
to `assets/`:

```text
assets/
├── robot.urdf
├── robot_processed.urdf
├── Collision/
│   └── *_auto_convex.obj
└── Scale/                  # Created only when --scale is used.
```

Review the generated URDF before simulation, especially when using non-uniform
scaling or inertia recomputation. The inertia calculation assumes unit mesh
density, and non-watertight geometry may produce inaccurate values.
