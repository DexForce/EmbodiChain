# Neural Motion Generator (NMG)

NMG generates joint actions for an ordered sequence of goals. A goal can specify
a joint configuration, a tool position, a tool orientation, or a full tool pose.
The policy observes the current joints, the tool center point (TCP), the active
goal and the previous action. The planner scales each action into a bounded
joint increment and advances the active slot when its constraint is satisfied.
For the Transformer architecture and training through differentiable forward
kinematics, see the [NMG project page](https://gs-world.github.io/nmg/).

## Watch the policy

Both arms visit eight goals in the default EmbodiChain scene. A cube marks the
TCP associated with a joint goal. A sphere marks a Cartesian position, and RGB
axes show a requested orientation. Amber highlights the active goal. Completed
goals turn green. Blue-gray dots trace the planned TCP path, and cyan traces
the TCP measured during playback.

### Free motion

The planner generates the joint path, retimes it under velocity and acceleration
limits, and replays each phase up to its waypoint. Franka mixes all four goal
types. PiPER mixes joint, position and full-pose goals.

```{raw} html
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:1rem">
<figure style="margin:0"><video controls playsinline preload="none" width="1280" height="720" style="width:100%;height:auto" poster="../../_static/nmg/franka-free-motion.jpg" aria-label="Franka FR3 free motion"><source src="../../_static/nmg/franka-free-motion.mp4" type="video/mp4"></video><figcaption>Franka FR3</figcaption></figure>
<figure style="margin:0"><video controls playsinline preload="none" width="1280" height="720" style="width:100%;height:auto" poster="../../_static/nmg/piper-free-motion.jpg" aria-label="PiPER V100 free motion"><source src="../../_static/nmg/piper-free-motion.mp4" type="video/mp4"></video><figcaption>PiPER V100</figcaption></figure>
</div>
```

### Closed-loop perturbation

During the second goal, the simulator displaces the first joint by 0.55 rad.
The magenta arrow marks the TCP jump and the dot marks its new position.
At each control interval the controller reads the measured joints, rebuilds
the observation and queries the policy again. It advances the goal after
three consecutive measurements satisfy the active constraints.

In these recordings the TCP moved about 28 cm on Franka and 14 cm on PiPER
at injection. The arms reached the disturbed goal after 37 and 13 further
policy updates, respectively, then finished the sequence.

```{raw} html
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:1rem">
<figure style="margin:0"><video controls playsinline preload="none" width="1280" height="720" style="width:100%;height:auto" poster="../../_static/nmg/franka-closed-loop-perturbation.jpg" aria-label="Franka FR3 closed loop perturbation"><source src="../../_static/nmg/franka-closed-loop-perturbation.mp4" type="video/mp4"></video><figcaption>Franka FR3</figcaption></figure>
<figure style="margin:0"><video controls playsinline preload="none" width="1280" height="720" style="width:100%;height:auto" poster="../../_static/nmg/piper-closed-loop-perturbation.jpg" aria-label="PiPER V100 closed loop perturbation"><source src="../../_static/nmg/piper-closed-loop-perturbation.mp4" type="video/mp4"></video><figcaption>PiPER V100</figcaption></figure>
</div>
```

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

## Select a model and prepare the observation

| Published v1 file | Matching embodiment | FP32 observation → raw action |
| --- | --- | --- |
| `v1/franka.onnx` | Franka FR3 with Robotiq 2F-140 | `[B, 417] → [B, 7]` |
| `v1/piper.onnx` | PiPER V100 with parallel gripper | `[B, 391] → [B, 6]` |

The example downloads a pinned revision and checks the file's SHA256.
Use `--download-only` to print its verified local path, `--offline` to
use the cache, or `--model-path` to supply a local copy.
[Published model files](https://huggingface.co/DexForceAI/neural_motion_generator/tree/ddca0e456e097ba42c05e1269febf73fa2835541/v1)

```bash
embodichain run-example sim/motion/planners/nmg/franka --download-only
```

Both exports have 12 waypoint slots. The observation packs current joint
positions and TCP pose, waypoint positions, xyzw quaternions and joints,
the active-slot indicator and constraint masks, the previous action, and
relative waypoint features. Joint order, TCP and feature order must match
the export. The file includes normalization. Inspect `observation_metadata`
on the loaded model for its layout and fingerprint.

## Single-step inference and input gradients

Use `NeuralMotionGenerator` when you already have an encoded observation
and want a single action. The following function evaluates the same input
with both backends. Install the optional PyTorch conversion dependencies
alongside the inference runtime for this example.

```bash
python -m pip install -e '.[policy-deploy,nmg]'
```

```python
from pathlib import Path

import torch
from embodichain.extensions.models.policy.nmg import NeuralMotionGenerator


def compare_step(path: Path, obs: torch.Tensor, action_weights: torch.Tensor):
    # obs: FP32 [B, obs_dim]. action_weights: FP32 [B, output_dim], same device.
    inference = NeuralMotionGenerator.from_onnx(path, device=obs.device)
    policy = NeuralMotionGenerator.from_onnx(path, device=obs.device, enable_grad=True)

    reference = inference(obs)
    obs = obs.requires_grad_(True)
    action = policy(obs)
    (d_obs,) = torch.autograd.grad(action, obs, grad_outputs=action_weights)
    return reference, action, d_obs
```

For repeated steps, load each model once and reuse it. ONNX Runtime returns
detached actions. With `enable_grad=True`, PyTorch evaluates the converted
graph with frozen weights and computes the vector-Jacobian product
`d_obs = Jᵀ action_weights`. Keep the masks and active slot fixed when
differentiating continuous features. The released files have been checked in
FP32 on CPU and CUDA at batch sizes 1 and 3 using ORT forward values and
central differences along joint-observation directions.

For a complete trajectory, [NeuralPlanner](../../overview/sim/motion/planners/neural_planner.md)
assembles the observations, checks the fingerprint and runs the policy in
inference mode through `MotionGenerator`. The single-step gradient contract
ends at the raw action. See the [model API](../../api_reference/model_catalogs.rst)
for loader arguments and the [installation guide](../../quick_start/install.md#optional-nmg-input-gradients-nmg)
for the backend dependencies.
