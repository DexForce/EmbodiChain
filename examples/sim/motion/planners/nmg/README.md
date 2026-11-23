# NMG v1: Franka and PiPER

Download the released [ONNX policies](https://huggingface.co/DexForceAI/neural_motion_generator/tree/main/v1)
and run them with EmbodiChain's `NeuralPlanner`. The files include weights and
observation normalization; no NMG checkout, actor code or checkpoint conversion
is needed.

## Run

Install ONNX Runtime and TOPP-RA in the EmbodiChain environment:

```bash
python -m pip install -e '.[policy-deploy]' toppra
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu
```

CPU-only installations can use `onnxruntime` instead of `onnxruntime-gpu` (do not
install both). Use `--device cuda:0` for a compatible CUDA stack. For a browser
viewer, replace `--headless` with `--web --keep-open`; a native viewer uses
`--keep-open` without `--headless`. Ctrl+C exits.

| Entry | Robot and policy TCP | Targets |
| --- | --- | --- |
| `franka.py` | FR3 + Robotiq 2F-140; flange offset 0.20 m with gripper rotation | Joint, position, orientation, full pose |
| `piper.py` | PiPER V100 + parallel gripper; link6 offset 0.143 m with gripper rotation | Three full poses |

Each entry defines its joint offsets, constraint types and recording camera.
`_robots.py` matches robot joint order and TCP to the policies; `_common.py`
shares model loading, planning and playback. These are single-robot integration
demos with authored reachable FK goals, not training data or benchmark cases.

## Record

Add `--video` to record simulated playback with a fixed camera:

```bash
embodichain run-example sim/motion/planners/nmg/franka --headless --device cpu --video /tmp/franka.mp4
embodichain run-example sim/motion/planners/nmg/piper --headless --device cpu --video /tmp/piper.mp4
```

Recording uses simulation time at 1280×720, 25 fps, and waits for encoding before
exit. The output suffix is normalized to `.mp4`, including for `.MP4` inputs.
It requires a rendering GPU even with CPU physics. `--video` selects
Fast-RT for `--renderer auto` and rejects `--no-playback` or `--renderer no-render`.

## Options and model contract

| Option | Use |
| --- | --- |
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

Paths are retimed at 0.6 rad/s and 1.5 rad/s², then replayed phase by phase on the
control grid with 0.5-second settling holds. Reports distinguish geometric
convergence, playback completion and measured joint tracking error. Failed plans
are not replayed. Playback does not certify physical task success or collision
avoidance.
