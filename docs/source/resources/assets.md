
# Data Assets

EmbodiChain provides a comprehensive set of pre-built data assets hosted on HuggingFace, covering robots, end-effectors, objects, scenes, materials, and more. Assets are automatically downloaded on first use, but you can also pre-download them using the built-in CLI tool.

## Data Root Directory

By default, assets are stored in `~/.cache/embodichain_data`. You can override this by setting the `EMBODICHAIN_DATA_ROOT` environment variable:

```bash
export EMBODICHAIN_DATA_ROOT=/mnt/shared/embodichain_data
```

Similarly, the dataset recording root (used by `LeRobotRecorder`) defaults to `~/.cache/embodichain_datasets` and can be overridden with:

```bash
export EMBODICHAIN_DATASET_ROOT=/mnt/shared/embodichain_datasets
```

## Download CLI

The `embodichain.data` module provides a command-line interface for managing assets.

### List Available Assets

```bash
# List all assets across every category
python -m embodichain.data list

# List assets in a specific category
python -m embodichain.data list --category robot
```

The output shows each asset name and whether it has already been downloaded (`✓`):

```text
[robot] (18 assets)
  [✓] ABB
  [ ] ARX5
  [ ] Agile
  [✓] Aubo
  ...

Data root: /home/user/.cache/embodichain_data
```

### Download Assets

```bash
# Download a single asset by name
python -m embodichain.data download --name CobotMagicArm

# Download all assets in a category
python -m embodichain.data download --category robot

# Download everything
python -m embodichain.data download --all
```

Downloaded files are saved to `<data_root>/download/` and automatically extracted to `<data_root>/extract/`. Non-zip assets (e.g. `.glb` files) are copied into the extract directory.

## Asset Categories

| Category     | Description                                    | Examples                                        |
|-------------|------------------------------------------------|-------------------------------------------------|
| `robot`     | Robot URDF models                              | CobotMagicArm, Franka, UniversalRobots, UnitreeH1 |
| `eef`       | End-effector / gripper models                  | DH_PGC_140_50, Robotiq2F85, InspireHand        |
| `obj`       | Manipulable objects and furniture              | ShopTableSimple, CoffeeCup, TableWare           |
| `scene`     | Full scene environments                        | SceneData, EmptyRoom                            |
| `materials` | Rendering materials, IBL, and backgrounds      | SimResources, CocoBackground                    |
| `w1`        | DexForce W1 humanoid robot and components      | DexforceW1V021, DexforceW1ChassisV021           |
| `demo`      | Demo scene data                                | ScoopIceNewEnv, MultiW1Data                     |

## Using Assets in Code

Use `get_data_path` to resolve asset paths in your configuration. It accepts a relative path in the format `<AssetClassName>/<subpath>`:

```python
from embodichain.data import get_data_path

# Resolves to the URDF file, downloading if necessary
urdf_path = get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf")
```

`get_data_path` resolves paths in the following order:

1. **Absolute path** — returned as-is.
2. **Local data root** — if the file exists under `EMBODICHAIN_DATA_ROOT`, it is returned immediately without triggering a download.
3. **Data-class download** — falls back to the registered asset class, which downloads and extracts the asset from HuggingFace.

You can also instantiate asset classes directly:

```python
from embodichain.data.assets import CobotMagicArm

dataset = CobotMagicArm()
print(dataset.extract_dir)  # Path to extracted files
```

## Adding Custom Assets

To add a new asset:

1. **Create a class** in the appropriate file under `embodichain/data/assets/` (e.g., `robot_assets.py` for a robot):

```python
class MyRobot(EmbodiChainDataset):
    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "MyRobot.zip"),
            "<md5_checksum>",
        )
        prefix = "MyRobot"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)
```

2. The class is automatically discovered by the download CLI and `get_data_path` — no additional registration is needed.
