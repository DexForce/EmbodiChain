# NMG v1: Franka and PiPER

Download the released ONNX policies from
[DexForceAI/neural_motion_generator](https://huggingface.co/DexForceAI/neural_motion_generator/tree/main/v1)
and run them through EmbodiChain's existing `NeuralPlanner`. No NMG source checkout,
PyTorch actor implementation, training checkpoint or ONNX conversion is needed.
The ONNX files contain the actor weights, computation graph and observation
normalization. ONNX is an inference format, not encryption; this path does not
provide PyTorch autograd.

Use an EmbodiChain installation with ONNX Runtime and TOPP-RA. From a source
checkout, install the optional dependencies with
`python -m pip install -e '.[policy-deploy]' toppra`. CPU-only environments may
install `onnxruntime` instead of `onnxruntime-gpu` (do not install both).
Robot assets use the normal EmbodiChain data cache.

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu
```

For the browser viewer, replace `--headless` with `--web --keep-open`; Ctrl+C
closes the demo. A native viewer uses `--keep-open` without `--headless`.
Use `--device cuda:0` for a compatible CUDA simulation/ONNX Runtime stack.
These demonstrations use one robot (`--num-envs 1`). The released models accept
dynamic batches; the single-robot restriction belongs to this example.

| Demo | Robot and policy TCP | Constraints |
| --- | --- | --- |
| `franka` | FR3 + Robotiq 2F-140, TCP 0.20 m along the flange with the gripper rotation | Joint, position only, orientation only, full pose |
| `piper` | PiPER V100 + parallel gripper, TCP 0.143 m along link6 with the gripper rotation | Three full-pose waypoints |

The robot assembly and solver TCP in `_robots.py` match the model. Do not swap
the native Franka hand or another PiPER variant without adapting the TCP and
checking joint/control semantics. Goals are deterministic reachable FK poses.
They illustrate integration and are not a benchmark or training data.

Both policies have capacity 12, relative observations, xyzw quaternions and
0.2-radian action scaling. Each demo uses a 30-step-per-constraint planning
budget. Arrival tolerances are 0.01 m for position, 0.1 rad for orientation and
0.01 rad for maximum absolute joint error. Only the selected modality is
scored; a position-only target does not require the displayed orientation.
The report prints all errors for diagnosis, including unconstrained dimensions.

The raw geometric path is retimed at 0.6 rad/s and 1.5 rad/s². Playback regrids
individual rest-to-rest phases to the existing physics period, preserving each
constraint arrival as a phase endpoint. Half-second settling holds are added.
The report distinguishes geometric convergence, playback completion and measured
joint tracking error. Playback completion is not a physical task-success claim.
There are no obstacle or self-collision guarantees. A failed plan is reported
and is not played back.

## Record a video

Record actual simulated playback with a fixed overview camera:

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu --video /tmp/nmg-franka.mp4
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu --video /tmp/nmg-piper.mp4
```

The recorder writes 1280×720 MP4 at 25 fps, using simulation time. A rendering
GPU is required even when physics and ONNX inference use `--device cpu`.
`--video` selects Fast-RT when `--renderer auto`; it cannot be combined with
`--no-playback` or `--renderer no-render`. The process waits for encoding before
exiting. Camera capture and encoding use EmbodiChain's existing recorder.

## Download and reproducibility

The demos pin commit `ddca0e456e097ba42c05e1269febf73fa2835541`, use the Hugging Face
cache and verify SHA256 before loading. Public downloads require no token.

```bash
embodichain run-example sim/motion/planners/nmg/franka --download-only
embodichain run-example sim/motion/planners/nmg/piper --download-only
embodichain run-example sim/motion/planners/nmg/franka --offline --headless --device cpu
embodichain run-example sim/motion/planners/nmg/piper --model-path /path/to/piper.onnx --headless
embodichain run-example sim/motion/planners/nmg/franka --headless --no-playback --output /tmp/franka-report.json
```

`--offline` applies to the model cache; robot assets must also be cached for a
fully offline simulation. `--model-path` accepts the same released model bytes
and performs the same hash check. `--download-only` does not start simulation.

| File | SHA256 |
| --- | --- |
| `v1/franka.onnx` | `2361954fda7b6f0d21445ebfb5099d31bdd67199bdadcecba71e0835f2157ad5` |
| `v1/piper.onnx` | `223a02d7f1c3db1ea96352404f623ebfde7e1b00f3bc3d7a186a634dfea5ecf0` |

For a new release, export and verify in the NMG repository, publish new model
files, then update the revision, hashes and matched robot contract here.
EmbodiChain consumes deployment files and does not convert training checkpoints.
