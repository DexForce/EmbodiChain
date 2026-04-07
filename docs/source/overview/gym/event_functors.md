# Event Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available event functors that can be used with the Event Manager. Event functors are configured using {class}`~cfg.EventCfg` and can be triggered at different stages: ``startup``, ``reset``, or ``interval``.

## Physics Randomization

```{list-table} Physics Randomization Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~randomization.physics.randomize_rigid_object_mass`
  - Randomize object masses within a specified range. Supports both absolute and relative mass randomization.

    ```json
    {"func": "randomize_rigid_object_mass", "mode": "reset",
     "params": {"entity_cfg": {"uid": "bottle"},
                "mass_range": [0.01, 0.1], "relative": false}}
    ```
* - {func}`~randomization.physics.randomize_rigid_object_center_of_mass`
  - Randomize the center of mass of rigid objects by applying position offsets. Only works with dynamic objects.

    ```json
    {"func": "randomize_rigid_object_center_of_mass", "mode": "reset",
     "params": {"entity_cfg": {"uid": "bottle"},
                "com_range": [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]}}
    ```
* - {func}`~randomization.physics.randomize_articulation_mass`
  - Randomize articulation link masses within a specified range. Supports regex-based link selection via the ``link_names`` parameter, and both absolute and relative mass randomization.

    ```json
    {"func": "randomize_articulation_mass", "mode": "reset",
     "params": {"entity_cfg": {"uid": "CobotMagic"},
                "mass_range": [0.8, 1.2], "relative": true,
                "link_names": ".*link.*"}}
    ```
```

## Visual Randomization

```{list-table} Visual Randomization Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {class}`~randomization.visual.randomize_visual_material`
  - Randomize textures, base colors, and material properties (metallic, roughness, IOR). Implemented as a Functor class. Supports both RigidObject and Articulation assets.

    ```json
    {"func": "randomize_visual_material",
     "mode": "interval", "interval_step": 10,
     "params": {"entity_cfg": {"uid": "table"},
                "random_texture_prob": 0.5,
                "texture_path": "CocoBackground/coco",
                "base_color_range": [[0.2, 0.2, 0.2], [1.0, 1.0, 1.0]]}}
    ```
* - {func}`~randomization.visual.randomize_light`
  - Vary light position, color, and intensity within specified ranges.

    ```json
    {"func": "randomize_light",
     "mode": "interval", "interval_step": 10,
     "params": {"entity_cfg": {"uid": "light_1"},
                "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
                "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                "intensity_range": [50.0, 100.0]}}
    ```
* - {func}`~randomization.visual.randomize_emission_light`
  - Randomize global emission light color and intensity. Applies the same emission light properties across all environments.

    ```json
    {"func": "randomize_emission_light",
     "mode": "interval", "interval_step": 10,
     "params": {"color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                "intensity_range": [0.5, 2.0]}}
    ```
* - {func}`~randomization.visual.randomize_camera_extrinsics`
  - Randomize camera poses for viewpoint diversity. Supports both attach mode (pos/euler perturbation) and look_at mode (eye/target/up perturbation).

    ```json
    {"func": "randomize_camera_extrinsics", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cam_high"}, "mode": "look_at",
                "eye_range": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]}}
    ```
* - {func}`~randomization.visual.randomize_camera_intrinsics`
  - Vary focal length (fx, fy) and principal point (cx, cy) within specified ranges.

    ```json
    {"func": "randomize_camera_intrinsics", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cam_high"},
                "fx_range": [400, 500], "fy_range": [400, 500]}}
    ```
* - {func}`~randomization.visual.set_rigid_object_visual_material`
  - Set a rigid object's visual material deterministically (non-random). Useful for configs that want fixed colors/materials during reset.

    ```json
    {"func": "set_rigid_object_visual_material", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cup"}, "base_color": [0.8, 0.2, 0.2]}}
    ```
* - {func}`~randomization.visual.set_rigid_object_group_visual_material`
  - Set a rigid object group's visual material deterministically (non-random). Useful for configs that want fixed colors/materials during reset.

    ```json
    {"func": "set_rigid_object_group_visual_material", "mode": "reset",
     "params": {"entity_cfg": {"uid": "objects"}, "base_color": [0.8, 0.2, 0.2]}}
    ```
```

## Spatial Randomization

```{list-table} Spatial Randomization Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~randomization.spatial.randomize_rigid_object_pose`
  - Randomize object positions and orientations. Supports both relative and absolute pose randomization.

    ```json
    {"func": "randomize_rigid_object_pose", "mode": "reset",
     "params": {"entity_cfg": {"uid": "bottle"},
                "position_range": [[-0.08, -0.12, 0.0], [0.08, 0.04, 0.0]],
                "relative_position": true}}
    ```
* - {func}`~randomization.spatial.randomize_robot_eef_pose`
  - Vary end-effector initial poses by solving inverse kinematics. The randomization is performed relative to the current end-effector pose.

    ```json
    {"func": "randomize_robot_eef_pose", "mode": "reset",
     "params": {"entity_cfg": {"uid": "CobotMagic",
                "control_parts": ["left_arm", "right_arm"]},
                "position_range": [[-0.01, -0.01, -0.01], [0.01, 0.01, 0]]}}
    ```
* - {func}`~randomization.spatial.randomize_robot_qpos`
  - Randomize robot joint configurations. Supports both relative and absolute joint position randomization, and can target specific joints.

    ```json
    {"func": "randomize_robot_qpos", "mode": "reset",
     "params": {"entity_cfg": {"uid": "CobotMagic"},
                "qpos_range": [[-0.1], [0.1]], "relative": true}}
    ```
* - {func}`~randomization.spatial.randomize_articulation_root_pose`
  - Randomize the root pose (position and rotation) of an articulation. Supports both relative and absolute pose randomization. Similar to randomize_rigid_object_pose but for multi-link rigid body systems.

    ```json
    {"func": "randomize_articulation_root_pose", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cabinet"},
                "position_range": [[-0.05, -0.05, 0], [0.05, 0.05, 0]],
                "relative_position": true}}
    ```
* - {func}`~randomization.spatial.randomize_target_pose`
  - Randomize a virtual target pose and store it in env state. Generates random target poses without requiring a physical object in the scene.

    ```json
    {"func": "randomize_target_pose", "mode": "reset",
     "params": {"position_range": [[0.5, -0.2, 0.8], [0.8, 0.2, 0.8]],
                "target_pose_key": "goal_pose"}}
    ```
* - {class}`~randomization.spatial.planner_grid_cell_sampler`
  - Sample grid cells for object placement without replacement. Implemented as a Functor class. Divides a planar region into a regular 2D grid and samples cells to place objects at their centers.

    ```json
    {"func": "planner_grid_cell_sampler", "mode": "reset",
     "params": {"grid_origin": [0.5, -0.3], "grid_size": [0.4, 0.6],
                "grid_res": [4, 6], "entity_cfg": {"uid": "objects"}}}
    ```
```

## Geometry Randomization

```{list-table} Geometry Randomization Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~randomization.geometry.randomize_rigid_object_scale`
  - Randomize a rigid object's body scale factors (multiplicative). Supports uniform scaling across all axes or independent per-axis scaling.

    ```json
    {"func": "randomize_rigid_object_scale", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cup"},
                "scale_range": [[0.8, 0.8, 0.8], [1.2, 1.2, 1.2]]}}
    ```
* - {func}`~randomization.geometry.randomize_rigid_objects_scale`
  - Randomize body scale factors for multiple rigid objects. Supports shared sampling (same scale for all objects) or independent sampling per object.

    ```json
    {"func": "randomize_rigid_objects_scale", "mode": "reset",
     "params": {"entity_cfg": {"uid": "objects"},
                "scale_range": [[0.8, 0.8, 0.8], [1.2, 1.2, 1.2]]}}
    ```
```

## Asset Management

```{list-table} Asset Management Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {class}`~events.replace_assets_from_group`
  - Swap object models from a folder on reset for visual diversity. Currently supports RigidObject assets with mesh-based shapes.

    ```json
    {"func": "replace_assets_from_group", "mode": "reset",
     "params": {"entity_cfg": {"uid": "cup"}, "group_uid": "cups_group"}}
    ```
* - {class}`~events.prepare_extra_attr`
  - Set up additional object attributes dynamically. Supports both static values and callable functions. Useful for setting up affordance data and other custom attributes.

    ```json
    {"func": "prepare_extra_attr", "mode": "reset",
     "params": {"attrs": [
       {"name": "object_lengths", "mode": "callable",
        "entity_uids": "all_objects", "func_name": "compute_object_length",
        "func_kwargs": {"is_svd_frame": true, "sample_points": 5000}}]}}
    ```
* - {func}`~events.register_entity_attrs`
  - Register entity attributes to a registration dict in the environment. Supports fetching attributes from both entity properties and prepare_extra_attr functor.

    ```json
    {"func": "register_entity_attrs", "mode": "reset",
     "params": {"entity_cfg": {"uid": "bottle"},
                "attrs": ["mass"], "registration": "object_info"}}
    ```
* - {func}`~events.register_entity_pose`
  - Register entity poses to a registration dict. Supports computing relative poses between entities and transforming object-frame poses to arena frame.

    ```json
    {"func": "register_entity_pose", "mode": "reset",
     "params": {"entity_cfg": {"uid": "bottle"},
                "pose_register_params": {"compute_relative": false,
                                         "to_matrix": true}}}
    ```
* - {func}`~events.register_info_to_env`
  - Batch register multiple entity attributes and poses using a registry list. Combines register_entity_attrs and register_entity_pose functionality.

    ```json
    {"func": "register_info_to_env", "mode": "reset",
     "params": {"registry": [
       {"entity_cfg": {"uid": "bottle"},
        "pose_register_params": {"compute_relative": false,
          "compute_pose_object_to_arena": true, "to_matrix": true}}],
      "registration": "affordance_datas", "sim_update": true}}
    ```
* - {func}`~events.drop_rigid_object_group_sequentially`
  - Drop objects from a rigid object group one by one from a specified height with position randomization.

    ```json
    {"func": "drop_rigid_object_group_sequentially", "mode": "reset",
     "params": {"entity_cfg": {"uid": "objects"},
                "drop_height": 0.5,
                "position_range": [[-0.1, -0.1], [0.1, 0.1]]}}
    ```
* - {func}`~events.set_detached_uids_for_env_reset`
  - Set UIDs of objects that should be detached from automatic reset. Useful for objects that need custom reset handling.

    ```json
    {"func": "set_detached_uids_for_env_reset", "mode": "reset",
     "params": {"uids": ["ground_plane"]}}
    ```
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
