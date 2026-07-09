# Event Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available event functors that can be used with the Event Manager. Event functors are configured using {class}`~cfg.EventCfg` and can be triggered at different stages: ``startup``, ``reset``, or ``interval``.

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new event or randomization functor with the correct signature (`env, env_ids, ...`), function or class style, and module placement. Use **`/add-test`** to generate mock-based tests.
````

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
  - Vary light position, color, intensity, and direction within specified ranges.

    .. note::
        ``position_range`` is ignored for global scene lights (``"sun"``, ``"direction"``).
        Use ``direction_range`` instead for these types.

    .. note::
        ``direction_range`` is only applicable for directional light types
        (``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``).

    ```json
    {"func": "randomize_light",
     "mode": "interval", "interval_step": 10,
     "params": {"entity_cfg": {"uid": "light_1"},
                "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
                "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                "intensity_range": [50.0, 100.0],
                "direction_range": [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]}}
    ```
* - {func}`~randomization.visual.randomize_emission_light`
  - Randomize global emission light color and intensity. Applies the same emission light properties across all environments.

    ```json
    {"func": "randomize_emission_light",
     "mode": "interval", "interval_step": 10,
     "params": {"color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                "intensity_range": [0.5, 2.0]}}
    ```
* - {class}`~randomization.visual.randomize_indirect_lighting`
  - Randomize indirect (IBL) lighting or emissive light. Implemented as a Functor class. Operates in one of two **mutually exclusive** modes — configuring both raises a ``ValueError``:

    **HDR mode** — provide ``path`` pointing to a folder of ``.hdr`` files. A random file is selected on each call and applied as the environment map. The ``path`` is resolved via ``get_data_path``, supporting absolute paths, data-root-relative paths, and dataset-class paths.

    ```json
    {"func": "randomize_indirect_lighting",
     "mode": "interval", "interval_step": 10,
     "params": {"path": "EnvMapHDR/EnvMapHDR"}}
    ```

    **Emissive mode** — provide ``emissive_color_range`` (pair of RGB lists) and/or ``emissive_intensity_range`` (pair of floats). Color and intensity are sampled uniformly on each call and applied via ``set_emission_light``.

    ```json
    {"func": "randomize_indirect_lighting",
     "mode": "interval", "interval_step": 10,
     "params": {"emissive_color_range": [[0.8, 0.8, 0.8], [1.0, 1.0, 1.0]],
                "emissive_intensity_range": [80.0, 150.0]}}
    ```

    Applies the same lighting to all environments.
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
* - {class}`~randomization.spatial.randomize_anchor_height`
  - Randomize the height of an anchor object and shift other objects by the same delta. Implemented as a Functor class. Samples a per-environment height delta (uniform range or discrete candidates), moves the anchor object relative to its configured initial position, and adds the same delta to the Z component of every other included object while preserving XY and rotation.

    ```json
    {"func": "randomize_anchor_height", "mode": "reset",
     "params": {"anchor_uid": "table",
                "height_delta_range": [[-0.05], [0.05]],
                "exclude_uids": ["floor"]}}
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

## Recording

```{list-table} Recording Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {class}`~record.record_camera_data`
  - Record RGB frames from a dedicated camera and save them as an MP4 video when an episode resets. This is useful for debugging, qualitative evaluation, and demo capture. The functor creates its own camera from the configured pose and intrinsics, captures frames during ``interval`` execution, and writes one video per episode.

    ```json
    {"func": "record_camera_data", "mode": "interval", "interval_step": 1,
     "params": {"name": "debug_cam",
                "resolution": [640, 480],
                "eye": [1.0, 0.0, 1.2],
                "target": [0.0, 0.0, 0.5],
                "up": [0.0, 0.0, 1.0],
                "save_path": "./outputs/videos"}}
    ```
* - {class}`~record.record_camera_data_async`
  - Record RGB frames for several environments independently, then merge them into a single tiled MP4 once each tracked environment finishes an episode. This variant is useful when you want side-by-side qualitative comparison across vectorized environments.

    ```json
    {"func": "record_camera_data_async", "mode": "interval", "interval_step": 1,
     "params": {"name": "overview_async_cam",
                "resolution": [640, 480],
                "eye": [1.0, 0.0, 1.2],
                "target": [0.0, 0.0, 0.5],
                "up": [0.0, 0.0, 1.0],
                "save_path": "./outputs/videos"}}
    ```
```

### record_camera_data

The ``record_camera_data`` functor lives under the record manager module, but it is configured through the event pipeline because it runs periodically during stepping and flushes video output at episode reset.

```{note}
This is separate from {doc}`dataset_functors`. Use dataset functors when you need training data such as observations and actions. Use ``record_camera_data`` when you need human-viewable videos for debugging or demos.
```

#### Typical Usage

- Set ``mode="interval"`` so the camera captures frames during stepping.
- Use ``interval_step=1`` to record every environment step, or increase it to reduce file size.
- Choose a fixed third-person camera pose with ``eye``, ``target``, and ``up``.
- Save output videos with ``save_path``.

#### Parameters

```{list-table} record_camera_data Parameters
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - ``name``
  - Camera name used for the internally created sensor and output file naming. Default: ``"default"``.
* - ``resolution``
  - Output image size as ``[width, height]``. Default: ``[640, 480]``.
* - ``eye``
  - Camera position in world coordinates. Default: ``[0, 0, 2]``.
* - ``target``
  - Look-at target in world coordinates. Default: ``[0, 0, 0]``.
* - ``up``
  - Up vector for the camera pose. Default: ``[0, 0, 1]``.
* - ``intrinsics``
  - Camera intrinsics as ``[fx, fy, cx, cy]``. Defaults to values derived from the configured resolution.
* - ``max_env_num``
  - Maximum number of environments to tile into one frame when recording vectorized environments.
* - ``save_path``
  - Output directory for generated MP4 files. Default: ``./outputs/videos``.
```

#### Behavior Notes

- In vectorized environments, frames from multiple environments are tiled into a single composite image before video encoding.
- Videos are flushed during episode initialization for environments that are resetting, not by the Dataset Manager.
- If the process exits without another reset, the final episode may not be flushed to disk.
- This functor captures RGB imagery for visualization. It does not record actions, rewards, or structured training data.

### record_camera_data_async

The ``record_camera_data_async`` functor is the multi-environment variant of ``record_camera_data``. It tracks a small set of environments independently, waits until each one has completed an episode, and then writes a single tiled video that combines them.

#### Typical Usage

- Use this variant when you want one comparison video spanning multiple vectorized environments.
- Configure it with ``mode="interval"`` just like ``record_camera_data``.
- Keep the number of tracked environments modest; the current implementation records up to four environments.
- Use the same camera pose parameters as the single-environment recorder.

#### Parameters

The async recorder accepts the same parameters as ``record_camera_data``:

- ``name``
- ``resolution``
- ``eye``
- ``target``
- ``up``
- ``intrinsics``
- ``max_env_num``
- ``save_path``

#### Behavior Notes

- Frames are buffered separately for each tracked environment and merged later into one tiled video.
- The current implementation tracks at most four environments, even if ``num_envs`` is larger.
- Output is written only after all tracked environments have completed an episode, so video generation may lag behind individual resets.
- This variant is meant for qualitative inspection and comparison, not structured dataset export.

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
