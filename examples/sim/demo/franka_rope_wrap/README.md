# Franka floor pickup and rope wrapping

A single Franka arm picks up a rope end directly from the floor, lifts it,
wraps it around one post, tightens it and holds it for two seconds. The default
run simulates 36 seconds with 96 cable segments in one CUDA environment.

The joint targets are planned before execution. During physics, the arm uses
position/velocity drives and the fingers hold the rope through contact. There
is no gripper attachment and no per-frame rope state assignment. The final
hold keeps the gripper closed; release and self-locking are not demonstrated.
Finger geometry, friction and material values are demonstration settings, not
hardware calibration. This is a standalone scripted example, not a Gym task or
an online feedback planner.

## Requirements and assets

Use the cable-capable DexSim development API required by this branch, including
`SpawnedCable`, cable state access and DexUni. See the parent
[Newton cable example](../../../README.md#newton-cable-demo) for SDK installation.
A version string alone does not establish compatibility; the example records
the actual SDK build commit. Full physics execution requires a CUDA GPU.

The original robot is resolved through
`get_data_path("Franka/Panda/PandaWithHand.urdf")`. The existing `Franka` registry
uses `FrankaV4.zip` with MD5 `0c613d884798e8b084604381641cc8e1`. Although the
asset path contains `Panda`, this demo uses the model's `fr3_*` joint/link names.
Do not substitute an arbitrary Panda URDF without adapting the configuration.
Pre-download it for an offline run:

```bash
python -m embodichain data download --name Franka
```

Set `EMBODICHAIN_DATA_ROOT` before starting Python to select an existing asset
cache. A complete replacement configuration may instead set `robot_urdf` to an
absolute local URDF path. Missing meshes or other referenced model files fail
before simulation.

The example generates a robot URDF with flat finger pads, preserving the source
joint definitions and inertias. It copies referenced meshes and material
resources into the run directory using relative paths. The original asset cache
is unchanged. `generated_assets/provenance.json` records the asset reference,
source and generated model hashes, resource hashes and finger modifications.
Original model notices remain part of their resources.

The floor slab, fixed rope clamp and post are primitive shapes. The rope is
created from configuration; these objects require no additional model bundle.
The default run uses `no-render` and does not load native rendering resources.
Native rendering helpers, if used separately, also need the existing
`SimResources` asset (`python -m embodichain data download --name SimResources`).
Video generation, browser visualization and report publishing are outside this
example's command-line interface.

## Run

```bash
python -m embodichain run-example sim/demo/franka_rope_wrap \
  --output /tmp/franka-rope-wrap-run
```

The output directory must not already exist. The bundled `config.json` is the
default regardless of the working directory. To replace it with a complete
configuration:

```bash
python -m embodichain run-example sim/demo/franka_rope_wrap \
  --config /path/to/config.json --output /tmp/franka-rope-wrap-custom
```

Generate assets and the joint plan without starting physics:

```bash
python -m embodichain run-example sim/demo/franka_rope_wrap \
  --plan-only --output /tmp/franka-rope-wrap-plan
```

`--seconds 2` truncates execution for a smoke check. It records `partial`, keeps
`passed=false` and exits with code 1; it does not qualify the full task. A full
run exits with code 0 only when every physical acceptance check passes.
`--plan-only` exits with code 0 after a valid plan and records `planned`.

## Results and acceptance

The run directory contains:

- `config.json`, `source_snapshot.py` and `manifest.json`: exact inputs, source
  hash, optional repository commit, package/build versions and device metadata.
- `generated_assets/`: the modified URDF, copied resources and provenance.
- `robot_motion.npz` and `motion_validation.json`: planned targets and motion
  checks before physics.
- `states.npz`, `metrics.json` and `settled_state.json`: recorded states and
  measurements from physical execution.
- `validation.json`: the 19 acceptance checks, thresholds and measured summary.
  Exceptions are recorded in `failure.txt` with a failed manifest.

The acceptance checks cover completion, finite state, joint limits, direct floor
pickup, retained grasp, material-point slip, bilateral finger contact, winding,
contact arc, hold stability, rope height, post penetration, observed robot/post
clearance, anchor direction and exit bend, fixed anchor, segment gaps, dropped
contacts and floor penetration. They use actual simulated states and valid
contact geometry. DexUni contact geometry does not imply that normal or friction
impulses are available; consumers must inspect query capabilities.

On 2026-11-08 the default 36-second NoRender run passed all 19 checks with
DexSim `0.6.0.dev11491` (dev commit
`505ee1b5600ca0b3d0a2c8235930e7684283d049`), Newton 1.7.0, MuJoCo/MuJoCo-Warp
3.15.0, Python 3.12 and an RTX 5090 D v2. Final winding was 524.86 degrees,
maximum material grasp slip was 0.123 mm, and hold winding drift was
0.00949 degrees. No post penetration or dropped contacts were reported.

The included checks do not constitute an independent robot self-collision or
continuous swept-volume certification. One successful run also does not measure
multi-seed reliability, batched execution or reset stability. Recorded states
can support later replay, but the CLI does not generate a video.
