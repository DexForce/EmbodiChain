# Pick/Place in three 3DGS scenes with Franka and two cameras

Run a fresh Task Program against the registered, standardized Gaussian scene
packages. The example retains the FR3/parallel-hand model and initial pose,
executes physical Pick/Place, and records a fixed third-person camera plus a
camera rigidly mounted on `fr3_hand`. It generates actions from the live state;
it does not read a prerecorded action trajectory or animate robot joint states.

```bash
embodichain run-example gym/gaussian_pick_place/run --scene office
embodichain run-example gym/gaussian_pick_place/run --scene discoverse
embodichain run-example gym/gaussian_pick_place/run --scene robosimgs
# Run all three in fresh processes, audit, then export the full video collection:
embodichain run-example gym/gaussian_pick_place/run_all --gpu 0 --seed 42
```

`run.py` accepts `--scene`, `--output`, `--seed` (42), `--spp` (8),
`--prepare-only` and `--viewer`. Headless is the default. GPU visibility may be
selected with `CUDA_VISIBLE_DEVICES`; physics uses Default/CPU. Use the Python
environment containing the qualified DexSim Gaussian/Scene implementation and
FFmpeg for video export. Shared assets download on demand and can be cached
through `embodichain data download --name <asset-name>`.

## Deployments and ownership

| Scene | Runnable config | Registered background |
| --- | --- | --- |
| RoboSplatter office | [task.office.yaml](configs/task.office.yaml) | `RoboSplatterOfficeGS/asset.usda` |
| DISCOVERSE lab3 | [task.discoverse.yaml](configs/task.discoverse.yaml) | `DiscoverseLab3GS/asset.usda` |
| RoboSimGS tabletop | [task.robosimgs.yaml](configs/task.robosimgs.yaml) | `RoboSimGSTabletopGS/asset.usda` |

Each deployment selects one physical `env.<scene>.yaml`, one
`embodiment.<scene>.yaml`, a two-segment `program.<scene>.yaml`, the common
semantic integration and a single-arm verified execution policy. Configuration
loading performs strict schema, physical UID, capability-contract and lowering
checks before World construction. No environment subclass is added.

The physical environments own the backend, scene and settling event. Embodiments
own the FR3, parallel hand, reusable skill profile and camera declarations.
Programs own `Pick → Place → wait_stable`, world targets and acceptance.
The integration binds only the cube and support surface to semantics. Gaussian
backgrounds are imported as pure visuals with their authored placement preserved;
Office's packaged coordinate rotation must not be repeated in the deployment.

| Scene | Robot base | Stable object target |
| --- | --- | --- |
| Office | (-0.15,-0.1,0.76) | (0.30,-0.22,0.785) |
| Lab3 | (-0.4,1,0.7) | (0.05,0.88,0.725) |
| RoboSimGS | (0,0,0) | (0.45,-0.12,0.025) |

## Physics and cameras

The cube is a 5 cm, 50 g **dynamic** body. Gripping uses normal jaw commands,
contact and friction; the example never teleports the cube, changes its body
type or attaches it through a fixed constraint. The support is an independent
static box. Office's workbench is visible; the other collision supports are
hidden to display the scan. The background itself supplies no collision.

The Default backend advances four 0.01 s physical substeps per 25 Hz action.
FEP IK and the verified policy produce live robot commands. Measured effects
gate acquisition and release. The release target is 3 cm above the stable target
so the cube actually falls clear of the opened hand before retracting. Acceptance
requires lift >=4 cm, position error <=2 cm, tilt <=15°, open jaws and stable pose.

Both RGB cameras are 960x540. `cam_third` is a fixed independent camera;
`cam_wrist` is a robot-owned camera attached to `FrankaPanda/fr3_hand`.
The latter's local position is (0.10,0,0.065) m, with fx=fy=430 and principal
point (480,270). Camera poses use the mounted OpenGL convention (-Z forward,
+Y up) and XYZW quaternions. Its optical axis points toward hand-local
(0,0,0.10); projected hand +Z sets image-up and hand +Y sets image-right.
Thus the gripper appears below the working point and the jaws close horizontally.
This is a real mount transform, not a video rotation or a tracking camera.

Hybrid/8 spp and OptiX offscreen denoising are used. The captures are
display-referred, so `apply_display_srgb=False` avoids an extra display transfer.

## Recording and completion

After reset/settling the recorder copies s0. The existing demo step observer
then copies both cameras at every acknowledged action boundary through sT,
including ordinary final settling. It does not update a camera or render extra
frames. Images use MediaSpool NPZ chunks with a 64 MiB retained-frame budget.
Image rollouts and automatic trajectory saving are disabled.

```text
<output>/<scene>-<run-id>/
  run.json
  resolved/                  # effective YAML, runtime overrides and code hashes
  initial_cam_third.png / initial_cam_wrist.png
  episodes/<episode_uuid>/
    trajectory.pt
    execution_result.json
    captures.jsonl           # both native frame identities and physical state
    media/manifest.json / chunk-*.npz
    physical_metrics.json
    cam_third.mp4 / cam_wrist.mp4
    sensor_full.mp4          # paired views, original rate and complete task
    preview.jpg
    episode.json             # published only after successful verification
```

Every T-action trajectory has T+1 observations per camera. Native camera
identity, calibration and actual simulation clocks are checked. Third-person
extrinsics stay fixed; every wrist pose must equal measured hand pose times the
constant mount. The image-up/closing-axis convention and projected gripper-below-
TCP framing are also checked. Independent physical samples must agree with the
native trajectory. Metadata/media references are relative within the run.

`run_all.py` runs each deployment in a fresh process, retains return codes and
logs, then verifies all source artifacts and actual attach/release/settled
evidence. It decodes each video and compares it to the original uint8 RGB before
publishing a collection. Failure keeps diagnostic files and no complete episode
marker. Existing SDK close/reference-release/deferred cleanup is used.

```bash
# Read-only artifact audit; no World creation, network access or re-rendering:
python examples/gym/gaussian_pick_place/_verify.py /path/to/completed/run
```

The current native Python environment must remain available for SDK imports.
FFmpeg/H.264 previews are lossy; raw NPZ RGB is the pixel-consistency source.

## Scope and qualification

See [QUALIFICATION.md](QUALIFICATION.md) for measured outcomes and published
evidence. These runs qualify RGB and the configured independent task physics.
They do not qualify metric Gaussian depth. Bookcases, chairs and other scanned
objects have not been collisionized, and scanned lighting remains baked.
Upstream licensing is preserved per asset; DISCOVERSE model licensing is not
separately declared and is not inferred from its software license.
