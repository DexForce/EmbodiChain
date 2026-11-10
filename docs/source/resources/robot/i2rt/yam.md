# YAM dual arms

`embodichain_robots.YamI2rtCfg` configures the converted I2RT YAM dual-arm
assembly. Its implementation and asset definition live together in
`embodichain_robots.i2rt.yam`; the robot catalog ID is `YamI2rt`.
Catalog listing reads metadata without loading the model or downloading assets.

## Construction and control

```python
from embodichain.robot_registry import resolve_robot_cfg
from embodichain_robots import YamI2rtCfg

cfg = resolve_robot_cfg("YamI2rt")
restored = YamI2rtCfg.from_dict(cfg.to_dict())
robot = sim.add_robot(cfg=restored)
```

The converted asset contains 16 independent controlled joints. Both fingers
in each hand are independently actuated; no mimic relation is added.

| Control part | Source joints | Coordinates |
| --- | --- | --- |
| `right_arm` | `right_joint1`–`right_joint6` | radians |
| `left_arm` | `left_joint1`–`left_joint6` | radians |
| `right_eef` | `right_joint7`, `right_joint8` | metres |
| `left_eef` | `left_joint7`, `left_joint8` | metres |

`build_pk_serial_chain()` returns six-joint chains for `right_arm` and
`left_arm`, rooted at `<side>_base` and ending at `<side>_tcp`. Both the
simulation component and the chains use the same URDF. Each arm defaults to
`PytorchSolverCfg`; the TCP is the converted source grasp-site frame.

The ready pose, initial world placement, drive gains and collision offsets are
simulation defaults inherited from the reference assembly, not verified hardware
calibration. Override them through ordinary `RobotCfg` fields for another scene.
The preset uses the shared backend defaults; task-specific Newton gravity
compensation and high-friction contact settings belong to consuming deployments.

## Published assets and offline use

```bash
embodichain data download --name YamI2rt
embodichain preview-robot --robot YamI2rt
```

`YamI2rt` downloads the published `robot_assets/YamI2rtV1.zip` from
[DexForceAI/embodichain_data](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/robot_assets)
through the configured download prefix, with the official Hugging Face dataset
as fallback. The archive MD5 is `23601ea820ff3e6260336c4962ef01a5`; it includes
the source MJCF, converted URDF, visual/collision meshes and provenance.

`get_data_path("YamI2rt/yam_i2rt.urdf")` reuses the configured data cache. An
explicit `fpath` selects a prepared local URDF for simulation and kinematics:

```python
cfg = YamI2rtCfg.from_dict({"fpath": "/path/to/YamI2rt/yam_i2rt.urdf"})
```

The optional source-regeneration script verifies pinned source and generated
file hashes before installing a bundle. Mesh-conversion dependencies are needed
only for regeneration:

```bash
python -m pip install coacd==1.0.14 trimesh==4.12.2 scipy==1.18.1
python scripts/tools/prepare_yam_i2rt_assets.py \
  --archive /path/outside/asset-root/YamI2rtV1.zip
```

`--source-dir` selects the pinned offline source directory, and `--output`
selects the prepared asset directory. Model files and ZIP archives are
distributed separately from the source repository and Python wheel.

## Source and license

The source is `assets/embodiments/yam_i2rt/` at RoboSimStudio commit
[`73739abe212f963e8a9f5862b27bca370df091a7`](https://github.com/robosimstudio/robosimstudio/tree/73739abe212f963e8a9f5862b27bca370df091a7/assets/embodiments/yam_i2rt).
Its license and attribution identify the YAM arm and `linear_4310` gripper as
I2RT Robotics assets under MIT. Conversion adds named TCP frames and decomposed
fingertip collision meshes. The archive retains `LICENSE`, `NOTICE` and
`PROVENANCE.json`; the repository preserves the model notice and MIT text.
