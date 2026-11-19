# AgiBot G1_120s asset

The `AgibotG1` asset resolves a [repaired USD entry][repair] and its
[GenieSimAssets G1_120s layers][source]. The repair moves the 18 gripper
colliders authored on parent Xforms to their geometry meshes, preserving the
source collision materials and approximation settings. The four existing
tip colliders, arm collision settings, gripper linkages and sensor layers
retain their source configuration.

```bash
embodichain data download --name AgibotG1
```

Resolve the model with `get_data_path("AgibotG1/robot.usda")`. The provider
pins and verifies the entry, official dependencies and
[CC BY-NC-SA 4.0 license][license]. Keep the `configuration/` directory
with the entry layer when supplying a local copy.

[source]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee/robot/G1_120s
[license]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE
[repair]: https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/c2097aa5a6745d5d792dfec7dc9547893b7c0aba/robot_assets/AgibotG1V2/README.md
