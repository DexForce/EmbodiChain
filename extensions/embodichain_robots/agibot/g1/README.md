# AgiBot G1_120s asset

The `AgibotG1` asset fetches the original [GenieSimAssets G1_120s USD][source]
from AgiBot at revision `a0813aad7c16165daffbe6c2737754e0344809ee`. The
dataset's [CC BY-NC-SA 4.0 license][license] is cached alongside the unchanged
USD layers. SHA-256 hashes for the entry layer, dependencies and license live
in [assets.py](assets.py). Keep the `configuration/` directory with the entry
layer when supplying a local copy.

## Observed-replay URDF

`AgibotG1ObservedReplay` provides a physical URDF derived from the observed
AgiBot G1, with 14 arm cylinders and 22 hand collision meshes. Download the
model through the robot asset registry.

```bash
embodichain data download --name AgibotG1ObservedReplay
```

```python
from embodichain.data import get_data_path

urdf = get_data_path("AgibotG1ObservedReplay/robot.urdf")
```

The versioned archive contains the URDF, referenced meshes, model provenance,
the CC-BY-NC-SA-4.0 license and checksums. The [Hugging Face asset card][replay]
records its source and provides a manual download.

[source]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee/robot/G1_120s
[license]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE
[replay]: https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/e002d649bb292d3e18c3a97d5b5c22c8ddcc81a3/robot_assets/AgibotG1ObservedReplayV2/README.md
