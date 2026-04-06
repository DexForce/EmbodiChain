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
* - ``randomize_rigid_object_center_of_mass``
  - Randomize the center of mass of rigid objects by applying position offsets. Only works with dynamic objects.
* - ``randomize_articulation_mass``
  - Randomize articulation link masses within a specified range. Supports regex-based link selection via the ``link_names`` parameter, and both absolute and relative mass randomization.
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
* - ``randomize_emission_light``
  - Randomize global emission light color and intensity. Applies the same emission light properties across all environments.
* - ``randomize_camera_extrinsics``
  - Randomize camera poses for viewpoint diversity. Supports both attach mode (pos/euler perturbation) and look_at mode (eye/target/up perturbation).
* - ``randomize_camera_intrinsics``
  - Vary focal length (fx, fy) and principal point (cx, cy) within specified ranges.
* - ``set_rigid_object_visual_material``
  - Set a rigid object's visual material deterministically (non-random). Useful for configs that want fixed colors/materials during reset.
* - ``set_rigid_object_group_visual_material``
  - Set a rigid object group's visual material deterministically (non-random). Useful for configs that want fixed colors/materials during reset.
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
* - ``randomize_articulation_root_pose``
  - Randomize the root pose (position and rotation) of an articulation. Supports both relative and absolute pose randomization. Similar to randomize_rigid_object_pose but for multi-link rigid body systems.
* - ``randomize_target_pose``
  - Randomize a virtual target pose and store it in env state. Generates random target poses without requiring a physical object in the scene.
* - ``planner_grid_cell_sampler``
  - Sample grid cells for object placement without replacement. Implemented as a Functor class. Divides a planar region into a regular 2D grid and samples cells to place objects at their centers.
```

## Geometry Randomization

```{list-table} Geometry Randomization Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``randomize_rigid_object_scale``
  - Randomize a rigid object's body scale factors (multiplicative). Supports uniform scaling across all axes or independent per-axis scaling.
* - ``randomize_rigid_objects_scale``
  - Randomize body scale factors for multiple rigid objects. Supports shared sampling (same scale for all objects) or independent sampling per object.
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
* - ``register_entity_attrs``
  - Register entity attributes to a registration dict in the environment. Supports fetching attributes from both entity properties and prepare_extra_attr functor.
* - ``register_entity_pose``
  - Register entity poses to a registration dict. Supports computing relative poses between entities and transforming object-frame poses to arena frame.
* - ``register_info_to_env``
  - Batch register multiple entity attributes and poses using a registry list. Combines register_entity_attrs and register_entity_pose functionality.
* - ``drop_rigid_object_group_sequentially``
  - Drop objects from a rigid object group one by one from a specified height with position randomization.
* - ``set_detached_uids_for_env_reset``
  - Set UIDs of objects that should be detached from automatic reset. Useful for objects that need custom reset handling.
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
