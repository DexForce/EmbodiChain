# Dataset Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available dataset functors that can be used with the Dataset Manager. Dataset functors are configured using {class}`~cfg.DatasetFunctorCfg` and are responsible for collecting and saving episode data during environment interaction.

## Recording Functors

```{list-table} Dataset Recording Functors
:header-rows: 1
:widths: 30 70

* - Functor Name
  - Description
* - ``LeRobotRecorder``
  - Records episodes in LeRobot dataset format. Handles observation-action pair recording, format conversion, and episode saving. Requires LeRobot package to be installed.
```

## LeRobotRecorder

The ``LeRobotRecorder`` functor enables recording robot learning episodes in the LeRobot dataset format, which can be used for training with LeRobot's imitation learning algorithms.

### Features

- Records observation-action pairs during episodes
- Converts data to LeRobot format automatically
- Saves episodes when they complete
- Supports vision sensors (camera images)
- Supports robot state (qpos, qvel, qf)
- Supports custom observation features
- Auto-incrementing dataset naming

### Parameters

```{list-table} LeRobotRecorder Parameters
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - ``save_path``
  - Root directory for saving datasets. Defaults to EmbodiChain's default dataset root.
* - ``robot_meta``
  - Robot metadata for dataset (robot_type, control_freq, etc.)
* - ``instruction``
  - Optional task instruction (e.g., {"lang": "pick the cube"})
* - ``extra``
  - Optional extra metadata (scene_type, task_description, episode_info)
* - ``use_videos``
  - Whether to save videos (True) or images (False). Default: False.
* - ``image_writer_threads``
  - Number of threads for image writing
* - ``image_writer_processes``
  - Number of processes for image writing
```

### Recorded Data

The LeRobotRecorder saves the following data for each frame:

- ``observation.qpos``: Joint positions
- ``observation.qvel``: Joint velocities
- ``observation.qf``: Joint forces/torques
- ``action``: Applied action
- ``{sensor_name}.color``: Camera images (if sensors present)
- ``{sensor_name}.color_right``: Right camera images (for stereo cameras)

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

# Example: Record episodes in LeRobot format
dataset = {
    "lerobot_recorder": DatasetFunctorCfg(
        func="embodichain.lab.gym.envs.managers.datasets.LeRobotRecorder",
        params={
            "save_path": "/path/to/dataset/root",
            "robot_meta": {
                "robot_type": "dexforce_w1",
                "control_freq": 30,
            },
            "instruction": {
                "lang": "pick the cube and place it on the target",
            },
            "extra": {
                "scene_type": "table",
                "task_description": "pick_and_place",
                "episode_info": {
                    "rigid_object_physics_attributes": ["mass"],
                },
            },
            "use_videos": False,
        },
    ),
}
```

### Recording Workflow

1. **Initialization**: The Dataset Manager initializes the functor with the configured parameters
2. **Data Collection**: During episode rollout, the functor receives observations and actions
3. **Save Trigger**: When an episode completes, call the functor with `mode="save"`
4. **Finalization**: After all episodes, call `finalize()` to save any remaining data

```python
# Inside environment loop
if episode_done:
    dataset_manager.apply(mode="save", env_ids=completed_env_ids)

# After training completes
dataset_manager.apply(mode="finalize")
```

## Dataset Manager Modes

The Dataset Manager supports the following modes:

- ``save``: Save completed episodes for specified environment IDs
- ``finalize``: Finalize the dataset and save any remaining data

See {class}`~managers.dataset_manager.DatasetManager` for more details.
