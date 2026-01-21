# Event Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available event functors that can be used with the Event Manager. Event functors are configured using {class}`~envs.managers.cfg.EventCfg` and can be triggered at different stages: ``startup``, ``reset``, or ``interval``.

## Physics Randomization

```{list-table} Physics Randomization Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``randomize_rigid_object_mass``
  - Randomize object masses within a specified range. Supports both absolute and relative mass randomization.
```

## Visual Randomization

```{list-table} Visual Randomization Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``randomize_visual_material``
  - Randomize textures, base colors, and material properties (metallic, roughness, IOR). Implemented as a Functor class. Supports both RigidObject and Articulation assets.
* - ``randomize_light``
  - Vary light position, color, and intensity within specified ranges.
* - ``randomize_camera_extrinsics``
  - Randomize camera poses for viewpoint diversity. Supports both attach mode (pos/euler perturbation) and look_at mode (eye/target/up perturbation).
* - ``randomize_camera_intrinsics``
  - Vary focal length (fx, fy) and principal point (cx, cy) within specified ranges.
```

## Spatial Randomization

```{list-table} Spatial Randomization Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``randomize_rigid_object_pose``
  - Randomize object positions and orientations. Supports both relative and absolute pose randomization.
* - ``randomize_robot_eef_pose``
  - Vary end-effector initial poses by solving inverse kinematics. The randomization is performed relative to the current end-effector pose.
* - ``randomize_robot_qpos``
  - Randomize robot joint configurations. Supports both relative and absolute joint position randomization, and can target specific joints.
```

## Asset Management

```{list-table} Asset Management Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``replace_assets_from_group``
  - Swap object models from a folder on reset for visual diversity. Currently supports RigidObject assets with mesh-based shapes.
* - ``prepare_extra_attr``
  - Set up additional object attributes dynamically. Supports both static values and callable functions. Useful for setting up affordance data and other custom attributes.
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import EventCfg, SceneEntityCfg

# Example: Randomize object mass on reset
events = {
    "randomize_mass": EventCfg(
        func="randomize_rigid_object_mass",
        mode="reset",
        params={
            "entity_cfg": SceneEntityCfg(uid="cube"),
            "mass_range": (0.1, 2.0),
            "relative": False,
        },
    ),
}
```
