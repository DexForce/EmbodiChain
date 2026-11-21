# PolaRiS DROID scenes

The six catalog entries run independently. Each pairs a real captured Gaussian background with dynamic textured meshes from the same capture. Arhan Jain and the PolaRiS team published the scan and object data under MIT. The EmbodiChain bundles preserve the Gaussian positions, rotations, color coefficients, opacity and original scale channels. Five planar captures gain a 1 nm third axis for DexSim. The bundles keep the initial-condition JSON, convert the USD/USDZ object meshes and UVs to OBJ, retain their textures, and include per-file provenance and checksums. [Original dataset](https://huggingface.co/datasets/owhan/PolaRiS-Hub/tree/8c7e4103e266ef83d8b1ad2e9a63116edd5f155b) · [Pinned EmbodiChain packages](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/3ea9841c1bdccb999f3f7cd447937aed1579df13/scene_assets/polaris_droid_v1) · [Authors and evaluation suite](https://github.com/arhanjain/polaris)

| Catalog key | Object and goal |
|---|---|
| [polaris_move_latte_cup](move_latte_cup/README.md) | Place the latte cup on the cutting board. |
| [polaris_block_stack_kitchen](block_stack_kitchen/README.md) | Place both cubes in the green tray, then stack the green cube over the wood cube. |
| [polaris_food_bussing](food_bussing/README.md) | Place both the ice cream and the grapes in the yellow bowl. |
| [polaris_pan_clean](pan_clean/README.md) | Release the sponge over the frying pan. |
| [polaris_organize_tools](organize_tools/README.md) | Place the scissors in the large container. |
| [polaris_tape_into_container](tape_into_container/README.md) | Place the tape in the container. |

## Running a policy

From a source checkout, load a deployment and supply a batch of eight actions per step. The first seven are absolute Panda joint positions in radians. The final command continuously maps from -1 (open) to +1 (closed). The 15 Hz controller runs up to 450 steps. Each reset and step returns an external and a wrist-mounted 640 × 360 color image in the sensor observation.

```python
from pathlib import Path
import gymnasium as gym
import torch
import yaml
import embodichain.extensions.tasks
from embodichain.lab.gym.utils.gym_utils import config_to_cfg

path = Path("embodichain/extensions/tasks/configs/tasks/manipulation/polaris/move_latte_cup/task.droid.yaml")
deployment = yaml.safe_load(path.read_text())
cfg = config_to_cfg(deployment, source_path=path.resolve())
env = gym.make(deployment["id"], cfg=cfg)
try:
    observation, _ = env.reset(seed=42)
    robot = env.unwrapped.robot
    joints = robot.get_qpos()[:, :7]
    action = torch.cat((joints, -torch.ones_like(joints[:, :1])), dim=1)
    observation, reward, terminated, truncated, info = env.step(action)
    success, metrics = info["success"], info["metrics"]
finally:
    env.close()
```

## Scoring and physical scope

Progress is the fraction of latched criteria; success requires all criteria. Each object's chain first measures whether the native object origin comes within 20 cm of the gripper base, then its lowest collision point above the scene-specific height, then release with at least 80% of the object's axis-aligned XY footprint overlapping the target. Release requires an absolute finger driver angle below 0.10 rad. Lasting progress survives subsequent movement until that arena resets. The kitchen scene adds a seventh criterion for at least 50% overlap of the green cube over the wood cube after both placements. These footprint checks allow vertical separation. They do not check a stable contact stack or a scrubbing trajectory.

The layout, home pose, camera geometry and scoring thresholds follow the Eden port. [Reference configuration and rubric](https://github.com/embodied-ai-nexus/Eden/tree/8bb7ab02c0b6f1355143ac7757b0eb21c9506f34/eden/tasks/polaris)

Lift is clearance above the table here; the original PolaRiS implementation measures displacement from the object's spawn height. A hidden support at Z = 0 supplies tabletop collision beneath the visual capture. The native Panda/Robotiq assembly uses EmbodiChain's joint drives, gripper collision model and convex mesh decomposition. Contact dynamics, camera calibration in hardware, pretrained-policy success rates and real-robot deployment have not been qualified. Evaluation uses the nominal scene configuration by default. Pass `options={"initial_condition": 0}` to `reset` to select a published pose set by zero-based index. Each selected object retains the authored XY and quaternion, with its collision bottom placed on the support. Food Bussing uses the first battery from the author's two-battery scene, as in Eden.
