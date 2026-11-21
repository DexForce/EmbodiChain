# Tool Hang

A seven-joint Franka Panda arm with a parallel-jaw Panda hand assembles a loose frame in a fixed stand, then hangs the wrench by its large ring. The three-part assembly sits on a 0.8 m square table. Reset independently jitters the frame and wrench by up to 2 cm in X/Y and 10 degrees in yaw, and the seven arm joints by up to 0.02 rad.

The headless Newton deployment runs at 20 control steps per second. Its seven inputs are arena-frame translation increments (0.05 m per unit), rotation-vector increments (0.5 rad per unit), and a binary gripper command (positive opens, negative closes). The native Jacobian controller updates every physics step. Observations include joint state, arena-frame poses of the stand, frame and tool, and the native contact sensor report. Use its `is_valid` mask when reading contact slots.

The frame tip must reach the upright stand's base with the rod between both pairs of socket walls. The wrench ring must straddle the bar axis, within 6.75 mm of it and between 5% and 100% of its length. The ring must actually touch the horizontal bar, and both fingers and the palm must release the tool. These conditions must hold together. The manager records each diagnostic under `info["metrics"]`. The default sparse reward is one on success. The Gym episode terminates on success or if the frame or tool falls below 0.70 m, and truncates after 200 control steps (10 seconds).

From the repository root, the following command builds the packaged deployment and takes one step.

```bash
python - <<'PY'
from pathlib import Path
import gymnasium as gym
import torch
import embodichain.extensions.tasks
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.utils.utility import load_config

path = Path("embodichain/extensions/tasks/configs/tasks/manipulation/assembly/tool_hang/task.panda.yaml").resolve()
config = load_config(str(path))
env = gym.make(config["id"], cfg=config_to_cfg(config, source_path=path)).unwrapped
try:
    env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(
        torch.zeros((env.num_envs, 7), device=env.device)
    )
    print(info["success"], info["metrics"])
finally:
    env.close(exit_process=False)
PY
```

The three URDFs reproduce the primitive stand, hook frame and ratcheting wrench from robosuite. The versioned archive includes the generator, upstream MIT notice and asset provenance. Materials use flat colors. The arm and hand resolve independently from EmbodiChain's robot registry. [robosuite ToolHang](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/tool_hang.py) · [Versioned object archive](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/256a4cb8f6b66db4f3e6722847725f6509aae48c/scene_assets/robosuite_tool_hang_v1/RobosuiteToolHangV1.zip)

### Scripted demonstration

`RobosuiteToolHangExpert-v1` uses a single arena and the same scene, reset distribution and contact conditions. Its 2,000-step horizon allows time for the closed-loop insertion. The Panda uses joint-position and velocity targets with bounded motion between IK waypoints. A gripping effort holds each object between the physical pads. The demonstration measures the rod tip relative to the socket and the wrench ring relative to the observed bar, and tests support after the fingers open.

Run one demonstration from the repository root.

```bash
embodichain run-task \
  --gym_config embodichain/extensions/tasks/configs/tasks/manipulation/assembly/tool_hang/task.expert.panda.yaml \
  --headless --seed 7 --max_episodes 1
```

After inserting the frame, the Panda raises the open hand above the crossbar and crosses to the wrench side before turning the wrist for pickup. It tracks the measured bar end while guiding the large wrench ring over the retaining lip. The release sequence checks frame insertion, ring threading, ring–bar contact and insertion depth before commanding the fingers open. A settled hang can be checked by letting the open hand withdraw and observing the same contact conditions for a further 90 control steps (4.5 seconds).

The public collector can reset and try again after an incomplete episode. When measuring first-attempt success, run each seed once and record both the native termination and the state after withdrawal.
