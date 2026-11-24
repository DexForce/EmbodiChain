# Neural Motion Generator (NMG)

NMG generates joint actions for an ordered sequence of goals. A goal can specify
a joint configuration, a tool position, a tool orientation, or a full tool pose.
The policy observes the current joints, the tool center point (TCP), the active
goal and the previous action. The planner scales each action into a bounded
joint increment and advances the active slot when its constraint is satisfied.
For the Transformer architecture and training through differentiable forward
kinematics, see the [NMG project page](https://gs-world.github.io/nmg/).

## Demo scenarios

Both arms visit eight goals in the default EmbodiChain scene. A cube marks the
TCP associated with a joint goal. A sphere marks a Cartesian position, and RGB
axes show a requested orientation. Amber highlights the active goal. Completed
goals turn green. Blue-gray dots trace the planned TCP path, and cyan traces
the TCP measured during playback.

### Free motion

The planner generates the joint path, retimes it under velocity and acceleration
limits, and replays each phase up to its waypoint. Franka mixes all four goal
types. PiPER mixes joint, position and full-pose goals.

### Closed-loop perturbation

During the second goal, the simulator displaces the first joint by 0.55 rad.
The magenta arrow marks the TCP jump and the dot marks its new position.
At each control interval the controller reads the measured joints, rebuilds
the observation and queries the policy again. It advances the goal after
three consecutive measurements satisfy the active constraints.

In the CPU demos the TCP moved about 28 cm on Franka and 14 cm on PiPER
at injection. The arms reached the disturbed goal after 37 and 13 further
policy updates, respectively, then finished the sequence.

## Run and record

From the repository checkout, install the inference backend and the retimer,
then select the robot and scenario. Either scenario accepts either robot.

```bash
python -m pip install -e '.[policy-deploy]' toppra
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu \
    --scenario closed-loop-perturbation --output piper-recovery.json
```

Use `--web --keep-open` to watch interactively. Add `--video` to record the
fixed-camera view at 1280 × 720 and 25 fps. Recording uses a rendering GPU
even with CPU physics. On Linux servers without a display, prefix the recording
command with `xvfb-run -a`.

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu \
    --scenario closed-loop-perturbation --video franka-recovery.mp4
```

## Supported configurations and weights

| Supported configuration | Policy TCP | FP32 observation → raw action | ONNX inference | Native PyTorch |
| --- | --- | --- | --- | --- |
| Franka FR3 (7 DoF) with Robotiq 2F-140 | 0.20 m flange offset with gripper rotation | `[B, 417] → [B, 7]` | [`v1/franka.onnx`](https://huggingface.co/DexForceAI/neural_motion_generator/resolve/1f7398d2539fdc9c54d5e5c81590c784f34146ca/v1/franka.onnx) | [`v1/franka.pt`](https://huggingface.co/DexForceAI/neural_motion_generator/resolve/1f7398d2539fdc9c54d5e5c81590c784f34146ca/v1/franka.pt) |
| PiPER V100 (6 DoF) with parallel gripper | 0.143 m `link6` offset with gripper rotation | `[B, 391] → [B, 6]` | [`v1/piper.onnx`](https://huggingface.co/DexForceAI/neural_motion_generator/resolve/1f7398d2539fdc9c54d5e5c81590c784f34146ca/v1/piper.onnx) | [`v1/piper.pt`](https://huggingface.co/DexForceAI/neural_motion_generator/resolve/1f7398d2539fdc9c54d5e5c81590c784f34146ca/v1/piper.pt) |

Both model formats are pinned to the same release. The simulation examples
download the ONNX file and verify its SHA256.
Use `--download-only` to print its verified local path, `--offline` to
use the cache, or `--model-path` to supply a local copy.

```bash
embodichain run-example sim/motion/planners/nmg/franka --download-only
embodichain run-example sim/motion/planners/nmg/piper --download-only
```

### Prepare the observation

Both exports have 12 waypoint slots. The observation packs current joint
positions and TCP pose, waypoint positions, xyzw quaternions and joints,
the active-slot indicator and constraint masks, the previous action, and
relative waypoint features. Joint order, TCP and feature order must match
the export. The file includes normalization. Inspect `observation_metadata`
on the loaded model for its layout and fingerprint.

## Single-step inference and input gradients

Use `NeuralMotionGenerator` with an encoded FP32 observation. ONNX inference
loads `.onnx` with `from_onnx()` and uses the `policy-deploy` extra.
Input gradients load a matching native `.pt` with `from_checkpoint()` and use
PyTorch from the base installation.

The native `nmg.actor.v1` deployment checkpoint contains the actor state dict,
network configuration, observation metadata and normalization buffers. Its
`nmg.checkpoint_sha256` metadata identifies the source training checkpoint.
Use the deployment checkpoint paired with the ONNX export by policy identity.

```python
from pathlib import Path

import torch
from embodichain.extensions.models.policy.nmg import NeuralMotionGenerator


def compare_step(
    onnx_path: Path,
    checkpoint_path: Path,
    obs: torch.Tensor,
    action_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inference = NeuralMotionGenerator.from_onnx(onnx_path, device=obs.device)
    policy = NeuralMotionGenerator.from_checkpoint(checkpoint_path, device=obs.device)

    obs = obs.requires_grad_(True)
    action = policy(obs)
    (d_obs,) = torch.autograd.grad(action, obs, grad_outputs=action_weights)
    return inference(obs), action, d_obs
```

Load each model once for repeated steps. The native backend freezes policy
weights and follows the caller's PyTorch autograd context. Keep the waypoint
masks and active slot fixed when differentiating continuous observation
features. CPU and CUDA actions and input gradients are compared against the
paired ONNX exports and directional finite differences.

[NeuralPlanner](../../../overview/sim/motion/planners/neural_planner.md)
assembles observations, checks the fingerprint and runs ONNX inference for
complete trajectories. See the [model API](../../../api_reference/model_catalogs.rst)
and [installation guide](../../../quick_start/install.md#nmg-input-gradients).
