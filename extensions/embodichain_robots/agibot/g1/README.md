# AgiBot G1_120s asset

The `AgibotG1` asset fetches the original [GenieSimAssets G1_120s USD][source]
from AgiBot at revision `a0813aad7c16165daffbe6c2737754e0344809ee`. The
dataset's [CC BY-NC-SA 4.0 license][license] is cached alongside the unchanged
USD layers. SHA-256 hashes for the entry layer, dependencies and license live
in [assets.py](assets.py). Keep the `configuration/` directory with the entry
layer when supplying a local copy.

## Observed-replay URDF

`AgibotG1ObservedReplay` is the separately versioned physical model for the
prepared Real2Sim `03_agibot_toy` recording. It is distributed through
`DexForceAI/embodichain_data/robot_assets/` and the normal EmbodiChain asset
registry:

```bash
embodichain data download --name AgibotG1ObservedReplay
```

```python
from embodichain.data import get_data_path

urdf = get_data_path("AgibotG1ObservedReplay/robot_physical/robot.urdf")
```

The asset uses the `AgibotG1ObservedReplayV1` cache, a pinned distribution
revision, archive MD5/SHA-256 and a trusted file-checksum manifest. It retains
the recorded model's joints, inertials, visuals and calibration, adding 14
official arm cylinders and 22 authored hand collision meshes. The archive
includes the source attribution, CC-BY-NC-SA-4.0 license, exact qualification
inputs, native report and trace. Its native checks establish collision/contact
behavior; they do not establish recorded-task success or hardware accuracy.

The original `AgibotG1` asset and `AgibotG1Cfg` continue to select the official
USD. The observed-replay asset is not a new robot preset; use its bundled
`robot.json` when configuring the recorded joint/TCP/gripper layout. The
[Real2Sim example](../../../../examples/real2sim/README.md#download-prepared-physical-inputs)
contains the pinned complete-input download and reproduction instructions.

[source]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee/robot/G1_120s
[license]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE
