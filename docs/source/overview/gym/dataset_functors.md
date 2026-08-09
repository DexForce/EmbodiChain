# Dataset Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available dataset functors that can be used with the Dataset Manager. Dataset functors are configured using {class}`~cfg.DatasetFunctorCfg` and are responsible for collecting and saving episode data during environment interaction.

```{note}
This page covers structured dataset export. If you only need human-viewable debug or demo videos from a fixed camera, use {class}`~record.record_camera_data` on {doc}`event_functors`.
```

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new dataset functor with the correct signature, `DatasetFunctorCfg` registration, and module placement in `datasets.py`.
````

## Recording Functors

```{list-table} Dataset Recording Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {class}`~datasets.LeRobotRecorder`
  - Records episodes in LeRobot dataset format. Handles observation-action pair recording, format conversion, and episode saving. Requires LeRobot package to be installed.

    ```json
    {"func": "LeRobotRecorder", "mode": "save",
     "params": {"robot_meta": {"robot_type": "CobotMagic"},
                "instruction": {"lang": "Pour water from bottle to cup"},
                "extra": {"scene_type": "Commercial",
                          "task_description": "Pour water",
                          "data_type": "sim"},
                "use_videos": true}}
    ```
* - {class}`~async_datasets.AsyncLeRobotRecorder`
  - Drop-in async variant of ``LeRobotRecorder`` for parallel data collection. Saves each completed episode on a background worker thread so ``env.reset()`` no longer blocks on disk writes. Same on-disk format and params; also honors ``image_writer_threads``.

    ```json
    {"func": "AsyncLeRobotRecorder", "mode": "save",
     "params": {"robot_meta": {"robot_type": "CobotMagic"},
                "instruction": {"lang": "Pour water from bottle to cup"},
                "extra": {"scene_type": "Commercial",
                          "task_description": "Pour water",
                          "data_type": "sim"},
                "use_videos": false,
                "image_writer_threads": 4}}
    ```
```

## LeRobotRecorder

The ``LeRobotRecorder`` functor enables recording robot learning episodes in the LeRobot dataset format, which can be used for training with LeRobot's imitation learning algorithms.

### Features

- Records observation-action pairs during episodes
- Converts data to LeRobot format automatically
- Saves episodes when they complete
- Supports RGB, depth, and segmentation-mask camera observations
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
  - Robot identity metadata for the dataset (for example, ``robot_type``). Timing is derived from the environment and must not be configured here.
* - ``instruction``
  - Optional task instruction (e.g., {"lang": "pick the cube"})
* - ``extra``
  - Optional extra metadata (scene_type, task_description, episode_info)
* - ``use_videos``
  - Whether to save videos (True) or images (False). Default: False.
* - ``image_writer_threads``
  - Number of background threads for per-frame PNG writing (lerobot ``AsyncImageWriter``). When > 0, ``add_frame`` no longer blocks on ``PIL.Image.save``. Applies to both recorders. Try 4 threads per camera as a starting point.
* - ``image_writer_processes``
  - Number of background processes for image writing (alternative to threads; higher spawn cost, more isolation). Use 0 to rely on threads only.
* - ``depth_video``
  - Optional :class:`~embodichain.data_pipeline.depth_video.DepthVideoCfg` (or dict) to store camera depth as compressed ``gray12le``/HEVC sidecar videos instead of dense numeric arrays. See [Compressed depth sidecar](#compressed-depth-sidecar).
```

```{note}
Dataset FPS is derived from ``env.step_dt`` and therefore always matches the
actual observation-action sampling cadence. Configure ``sim_steps_per_control``
or ``target_control_frequency`` on the environment. LeRobot currently requires
an integer FPS, so the recorder rejects a non-integer derived control frequency
rather than writing inaccurate metadata.
```

### Recorded Data

The LeRobotRecorder saves the following data for each frame:

- ``observation.state``: Joint positions (proprioceptive state)
- ``action``: Applied action
- ``observation.images.{sensor_name}``: Camera images (if sensors present)
- ``observation.images.{sensor_name}_right``: Right camera images (for stereo cameras)
- ``observation.mask.{sensor_name}``: Native numeric segmentation-mask arrays
- ``observation.mask.{sensor_name}_right``: Right-camera segmentation-mask arrays

Depth and mask features keep the dtype and shape declared by the sensor
observation space. Masks are always stored as numeric LeRobot array features
(exact, lossless) so ``use_videos`` affects only the RGB image features.

Depth has two storage modes:

- **Numeric** (default): ``observation.depth.{sensor_name}`` numeric arrays, exact
  but storage-heavy for dense high-resolution depth.
- **Compressed sidecar** (when ``depth_video.enable=True``): depth is written as
  ``gray12le``/HEVC MP4s alongside the dataset (see below), and the numeric
  feature is dropped unless ``keep_numeric_fallback=True``.

## Compressed depth sidecar

When ``depth_video.enable=True`` and an HEVC encoder (``libx265``) is available,
``LeRobotRecorder`` writes each episode's depth maps as a single-channel
``gray12le`` video encoded losslessly with HEVC. This is issue #424 *Path A*:
an EmbodiChain-owned depth writer that works on Python 3.10/3.11 with LeRobot
0.4.4, without modifying the installed LeRobot package.

- Depth is quantized to 12-bit codes (logarithmic by default) and packed into
  the ``gray12le`` pixel format. With ``lossless=True`` (default) the 12-bit
  codes survive the encode/decode round-trip bit-exactly; the only error is the
  configurable float32 → 12-bit quantization step (typically sub-millimetre).
- Depth never enters LeRobot's RGB-only image/video pipeline; it lives in a
  sidecar tree next to the dataset:

  ```text
  <dataset_root>/
  ├── data/                 # LeRobot: state / action / mask
  ├── videos/               # LeRobot: RGB videos
  ├── depth_videos/         # depth sidecar
  │   └── <sensor>[_right]/episode_000000.mp4
  └── depth_meta.json       # quantization params + per-episode index
  ```

- The metadata schema (``is_depth_map``, ``video.depth_min/max/shift/use_log``,
  ``video.codec``, ``video.pix_fmt``) is aligned with the official LeRobot 0.6.0
  depth pipeline, so sidecar videos remain readable once EmbodiChain upgrades to
  Python 3.12 (issue #424, Path B).
- If no HEVC encoder is available, recording silently falls back to numeric
  depth features (PR #422) rather than failing.
- Segmentation masks are always kept as exact numeric features; they are never
  run through a lossy video codec.

```{attention}
12-bit quantization is **not** bit-exact relative to the original float32 or
uint16 depth input. Applications requiring exact source values must set
``keep_numeric_fallback=True`` (stores both the sidecar video and the raw
numeric feature) or keep depth numeric.
```

Read sidecar depth back with :func:`~embodichain.data_pipeline.depth_video.load_depth_dataset`,
which composes the LeRobot dataset with a :class:`~embodichain.data_pipeline.depth_video.DepthVideoLibrary`:

```python
from embodichain.data_pipeline.depth_video import load_depth_dataset

dataset, depth = load_depth_dataset("/path/to/dataset_root")
# depth in metres, shape (1, H, W), aligned to the LeRobot episode/frame index:
frame = dataset[0]
depth_map = depth.get(
    episode_index=int(frame["episode_index"]),
    sensor_key="camera",
    frame_index_in_episode=int(frame["episode_data_index"]),
)
```

### Dataset Recording vs Video Recording

```{list-table} Recording Options
:header-rows: 1
:widths: 30 35 35

* - Need
  - Use
  - Why
* - Training or imitation-learning data
  - {class}`~datasets.LeRobotRecorder`
  - Saves structured observation, action, and metadata for downstream pipelines.
* - Quick qualitative inspection or demos
  - {class}`~record.record_camera_data`
  - Saves MP4 videos from a dedicated camera without creating a training dataset.
```

## Saving Strategies

Saving is the part of data collection that most often bottlenecks the simulator. Each completed episode triggers a save on ``env.reset()``; with ``num_envs=N`` parallel envs truncating together, the recorder must persist N episodes worth of frames. Two independent levers control how expensive that is:

1. **Per-frame image writing** — ``LeRobotRecorder`` writes each camera frame to PNG synchronously inside ``add_frame`` (``compress_level=6``). Set ``image_writer_threads`` (Opt A) to offload these writes to a thread pool.
2. **Per-episode conversion + flush** — by default the whole convert + ``add_frame`` + ``save_episode`` loop runs inline on ``env.reset()``, blocking the sim. ``AsyncLeRobotRecorder`` (Opt B) clones the finished episode's buffer slice and runs that loop on a background worker, so ``env.reset()`` returns immediately.

```{list-table} Choosing a recorder
:header-rows: 1
:widths: 30 70

* - Situation
  - Use
* - Single env, or debugging / minimal memory
  - {class}`~datasets.LeRobotRecorder` (sync). Simplest, deterministic, lowest RAM. Errors surface at the call site.
* - Many parallel envs, sim must keep stepping
  - {class}`~async_datasets.AsyncLeRobotRecorder` with ``image_writer_threads=4``. Saving is pipelined off the sim thread; recommended for parallel collection.
* - Want most of the speedup without a background thread
  - {class}`~datasets.LeRobotRecorder` with ``image_writer_threads=4``. ~2.5x faster than sync, no episode cloning, bounded memory.
```

````{tip}
**Benchmark** (`scripts/benchmark/data_pipeline/benchmark_lerobot_save.py`, 4 envs x 2 episodes x 100 steps, 480x640, 800 frames/variant):

| Variant | t_total | speedup | sim blocked? |
|---------|---------|---------|--------------|
| ``LeRobotRecorder`` (sync) | 57.4 s | 1.0x | yes (~55 s) |
| + ``image_writer_threads=4`` | 22.0 s | 2.6x | yes (but faster) |
| ``AsyncLeRobotRecorder`` | 56.6 s | ~1.0x | **no** (drain-bound at finalize) |
| ``AsyncLeRobotRecorder`` + threads | 20.8 s | **2.8x** | **no** |

The sync stall grows **linearly** with ``num_envs`` (each reset saves N envs serially). The async recorder's sim stall stays near zero regardless of ``num_envs`` — that is the main reason to prefer it for parallel collection.
````

```{attention}
- ``AsyncLeRobotRecorder`` clones each finished episode (including camera frames) to CPU before enqueuing. For very high resolutions or many envs, monitor RSS — the worker normally keeps up, but a slow disk can let the queue grow.
- A single background worker touches the ``LeRobotDataset`` (which is not thread-safe), so episode order is preserved FIFO. Always let ``env.close()`` / ``dataset_manager.finalize()`` run so the worker drains before the dataset is finalized.
- ``env.close()`` calls ``sim.destroy()``, which exits the process without returning to Python. In scripts that build multiple envs, run each in its own subprocess and write results before closing.
```

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

### Parallel Collection (async recorder)

For ``num_envs > 1`` data collection, switch the functor to {class}`~async_datasets.AsyncLeRobotRecorder` and enable async image writing. Everything else (config structure, on-disk format, save/finalize calls) is identical:

```python
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

dataset = {
    "lerobot_recorder": DatasetFunctorCfg(
        func="embodichain.lab.gym.envs.managers.async_datasets.AsyncLeRobotRecorder",
        params={
            "save_path": "/path/to/dataset/root",
            "robot_meta": {"robot_type": "dexforce_w1"},
            "instruction": {"lang": "pick the cube"},
            "extra": {"scene_type": "table", "task_description": "pick_and_place"},
            "use_videos": False,
            "image_writer_threads": 4,   # Opt A: async per-frame PNG writes
        },
    ),
}
```

The async recorder drains its background worker during ``finalize()``, so make sure ``env.close()`` (or ``dataset_manager.finalize()`) runs at the end of collection.

### Compressed depth recording

To record camera depth as compressed sidecar videos (issue #424, Path A), pass a
``depth_video`` config. Depth is encoded losslessly as ``gray12le``/HEVC;
stereo cameras get a ``_right`` sidecar automatically.

```python
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

dataset = {
    "lerobot_recorder": DatasetFunctorCfg(
        func="embodichain.lab.gym.envs.managers.datasets.LeRobotRecorder",
        params={
            "save_path": "/path/to/dataset/root",
            "robot_meta": {"robot_type": "dexforce_w1"},
            "instruction": {"lang": "pick the cube"},
            "extra": {"scene_type": "table", "task_description": "pick_and_place"},
            "use_videos": False,
            "depth_video": {
                "enable": True,
                "depth_min": 0.05,   # metres mapped to quantum 0
                "depth_max": 5.0,    # metres mapped to quantum 4095
                "shift": 3.5,        # log-mode offset (metres)
                "use_log": True,     # logarithmic quantization
                "lossless": True,    # 12-bit codes preserved bit-exactly
                "input_unit": "m",
                "output_unit": "m",
                # "keep_numeric_fallback": True,  # also keep raw numeric depth
            },
        },
    ),
}
```

## Dataset Manager Modes

The Dataset Manager supports the following modes:

- ``save``: Save completed episodes for specified environment IDs
- ``finalize``: Finalize the dataset and save any remaining data

See {class}`~managers.dataset_manager.DatasetManager` for more details.
