# NMG v1: Franka and PiPER

Download the released [ONNX policies](https://huggingface.co/DexForceAI/neural_motion_generator/tree/main/v1)
and run them with EmbodiChain's `NeuralPlanner`. The files include weights and
observation normalization; no NMG checkout, actor code or checkpoint conversion
is needed.

The two scenarios use the default EmbodiChain scene and follow the
[NMG project demos](https://gs-world.github.io/nmg/) in showing a target
sequence, TCP motion, and the response to a joint disturbance.

## Run

Install ONNX Runtime and TOPP-RA from the EmbodiChain checkout.

```bash
python -m pip install -e '.[policy-deploy]' toppra
```

Run the default free-motion scenario on either arm. Headless runs use
no-render when the renderer is automatic and there is no video or web viewer.

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu
```

CPU-only installations can use `onnxruntime` instead of `onnxruntime-gpu` (do not
install both). Use `--device cuda:0` for a compatible CUDA stack. For a browser
viewer, replace `--headless` with `--web --keep-open`; a native viewer uses
`--keep-open` without `--headless`. Ctrl+C exits.

| Entry | Robot and policy TCP | Targets |
| --- | --- | --- |
| `franka.py` | FR3 + Robotiq 2F-140; flange offset 0.20 m with gripper rotation | Eight goals covering joint, position, orientation, and full pose |
| `piper.py` | PiPER V100 + parallel gripper; link6 offset 0.143 m with gripper rotation | Eight goals covering joint, position, and full pose |

Each entry defines its joint offsets, constraint types and recording camera.
`_robots.py` matches robot joint order and TCP to the policies; `_common.py`
shares model loading, planning and playback. These are single-robot integration
demos with authored reachable FK goals spanning both sides of the arm's
workspace.

### Read the markers

The viewer and recordings show the planned TCP path as blue-gray dots.
A cyan trace grows from the TCP measured during simulated playback. Gray
markers show pending goals, amber highlights the active one, and green marks
completed playback phases.

A cube locates the TCP of the joint-space goal. Spheres show Cartesian target
positions, and RGB axes show the commanded orientation when that phase is
active or settling. For an orientation-only goal, the axes and their small
anchor sit at the planned arrival position.

## Closed-loop perturbation

Select `--scenario closed-loop-perturbation` to use live policy feedback.
Each control interval reads the simulated arm, rebuilds the NMG observation,
and sends a bounded joint position command. During the second target the
demo displaces the first joint by 0.55 radians. The same controller then
continues toward the active target. A magenta arrow and dot show the measured
TCP displacement, and the cyan trace continues from the displaced tip.

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu --scenario closed-loop-perturbation
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu --scenario closed-loop-perturbation
```

The JSON report gives the per-target feedback steps, measured disturbance
size and recovery steps. The active target advances after three consecutive
measurements meet its constraint tolerances.

## Record

Add `--video` to either scenario to record the fixed-camera view.

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu --video /tmp/franka-free-motion.mp4
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu --video /tmp/piper-free-motion.mp4
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu --scenario closed-loop-perturbation --video /tmp/franka-recovery.mp4
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu --scenario closed-loop-perturbation --video /tmp/piper-recovery.mp4
```

Recording uses simulation time at 1280×720, 25 fps, and waits for encoding before
exit. The output suffix is normalized to `.mp4`, including for `.MP4` inputs.
It requires a rendering GPU even with CPU physics. `--video` selects
Fast-RT for `--renderer auto` and rejects `--no-playback` or `--renderer no-render`.
On a Linux server without a display, run the recording command under `xvfb-run -a`.

## Options and model contract

| Option | Use |
| --- | --- |
| `--scenario` | `free-motion` (default) or `closed-loop-perturbation` |
| `--download-only` | Cache and verify the model without starting simulation |
| `--offline` | Use the cached model; robot assets must also be cached |
| `--model-path PATH` | Use a local copy of the same published ONNX bytes |
| `--no-playback` | Plan and report geometric arrivals only |
| `--output PATH` | Save the printed JSON report |

`_common.py` pins the Hugging Face commit and verifies SHA256. Public downloads
need no token; robot assets use the standard EmbodiChain cache. For a new release,
export and verify in NMG, then update the pin, hashes and robot contract here.

The policies use capacity 12, relative observations, xyzw quaternions and
0.2-radian action scaling, with a 30-step budget per target. Arrival tolerances
are 0.01 m, 0.1 rad and 0.01 rad maximum joint error, applied to the selected
modality. The report also includes unconstrained errors for diagnosis.

Free-motion paths are retimed at 0.6 rad/s and 1.5 rad/s², then replayed phase by phase on the
control grid with 0.5-second settling holds. Reports distinguish geometric
convergence, playback completion and measured joint tracking error. Failed plans
are not replayed. Playback does not certify physical task success or collision
avoidance.

## Single-step actions and observation gradients

`NeuralMotionGenerator` loads the same local ONNX files and evaluates raw
policy actions. Install `embodichain[policy-deploy,nmg-grad]` to use both
backends. Use `--download-only` to find the verified model path.

Pass a batched FP32 tensor in the export's observation layout. Franka maps
`[B, 417]` to `[B, 7]` and PiPER maps `[B, 391]` to `[B, 6]`. The graph
applies observation normalization. Keep the active waypoint and masks fixed
when differentiating continuous features.

```python
import torch

from embodichain.extensions.models.policy.nmg import NeuralMotionGenerator

# obs is an encoded NMG observation with requires_grad=True.
inference = NeuralMotionGenerator.from_onnx(model_path, device=obs.device)
differentiable = NeuralMotionGenerator.from_onnx(
    model_path, device=obs.device, enable_grad=True
)

inference_action = inference(obs)
action = differentiable(obs)
(d_obs,) = torch.autograd.grad(action.square().mean(), obs)
```

The default backend returns detached actions. The PyTorch backend freezes
the weights and records input gradients under the caller's autograd context.
`NeuralPlanner` uses the inference backend for closed-loop rollout, action
scaling and retiming.

To check the published weights against ONNX Runtime and finite differences,
point the test at a local directory with both verified files.

```bash
EMBODICHAIN_NMG_MODEL_DIR=/path/to/v1 python -m pytest tests/models/policy/test_nmg.py --run-gpu
```
