# AgiBot G1_120s asset

The `AgibotG1` asset resolves a [repaired USD entry][repair] and its
[GenieSimAssets G1_120s layers][source]. The repair moves the 18 gripper
colliders authored on parent Xforms to their geometry meshes, preserving the
source collision materials and approximation settings. It also enables the
14 original arm cylinders and eight wrist camera mount meshes. It adds two
shoulder cylinders and seven body shapes from the [official G1 OmniPicker
USD][omni]. Gripper linkages and sensor layers retain their source configuration.
Self-collision remains disabled.

```bash
embodichain data download --name AgibotG1
```

Resolve the model with `get_data_path("AgibotG1/robot.usda")`. The provider
pins and verifies the entry, official dependencies and
[CC BY-NC-SA 4.0 license][license]. Keep the `configuration/` directory
with the entry layer when supplying a local copy.

[source]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee/robot/G1_120s
[license]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE
[repair]: https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/23370abf1c98b597ac13e40279dbb10cd775ccfb/robot_assets/AgibotG1V4/README.md
[omni]: https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/56b2596945b503385d5770c55ecf420848495e37/robot/G1_omnipicker/robot.usd
