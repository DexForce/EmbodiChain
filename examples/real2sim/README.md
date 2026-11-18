# Real2Sim: reconstruction and trajectory adaptation

Use a real robot recording to reconstruct a tabletop scene, annotate the recorded
motion, and correct that motion through native physics replay. The example keeps
the recorded robot and the reconstructed assets. It does not retarget human
motion or generate a new demonstration from atomic skills.

The commands below run from the repository root. Each stage writes to a **new**
directory. Reconstruction and trajectory execution are separate entry points;
both are implemented by this Real2Sim feature.

| Stage | Example entry point | Implementation |
| --- | --- | --- |
| Repair the pinned AgiBot physical input | `python -m examples.real2sim.prepare_agibot` | Model-owned authored collision recipe and prepared-package validation |
| Video to native Scene Engine export | `python -m examples.real2sim` | `gen_sim.real2sim.scene.reconstruct_video()` |
| Export to calibrated, registered scene | Python snippet below | `gen_sim.real2sim.scene.register_scene_export()` |
| Annotation, bounded correction, physical replay | `python -m examples.real2sim.agent` | `gen_sim.real2sim.execution.run_prepared_real2sim()` |
| Replay an already packaged trajectory | `python -m examples.real2sim.replay` | `gen_sim.real2sim.replay.replay_recording()` |

Reusable input, geometry, numerical correction, and evidence helpers live in
`embodichain/data_pipeline/real2sim/adaptation/`. Composition, service calls, and
simulation execution live in `embodichain/gen_sim/real2sim/`. The example modules
only parse arguments and call those packages.

## Download prepared physical inputs

The repaired assets are distributed through the project's Hugging Face dataset
`DexForceAI/embodichain_data`. [resources.lock.json](resources.lock.json) pins the
robot and complete-input revisions, archive hashes, internal checksum manifests,
original source package and the complete input package's qualification evidence.

For the complete five-case input package (about 166 MB):

```bash
curl -L --fail \
  'https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/9ff78edea77dedbdc5d5cce9e0b6ef4792cb11d5/demo/real2sim_inputs/real2sim_inputs_5_agibot_physical_v2.zip' \
  -o real2sim_inputs_5_agibot_physical_v2.zip
echo '593d6aec77b22577e7076a92afd150e9a1ff2b52857fe02dddd3cdd66bd90d6b  real2sim_inputs_5_agibot_physical_v2.zip' | sha256sum -c -
unzip real2sim_inputs_5_agibot_physical_v2.zip
python Real2SimInputs5AgiBotPhysicalV2/verify_inputs.py --load
export REAL2SIM_SOURCE="$PWD/Real2SimInputs5AgiBotPhysicalV2/samples/03_agibot_toy"
```

The AgiBot sample already selects `robot_physical/robot.urdf`; no repair command
is required. The complete package owns recording-specific `robot.json`,
calibration, trajectories, and `physical_validation/native/report.json` and
`trace.npz`. Original videos, depth and robot resources are preserved. The
upstream input archive remains available at its pinned source revision.

For the standalone robot model (about 7.0 MB), use the asset registry:

```bash
embodichain data download --name AgibotG1ObservedReplay
```

```python
from embodichain.data import get_data_path

urdf = get_data_path("AgibotG1ObservedReplay/robot.urdf")
```

This resolves the versioned `AgibotG1ObservedReplayV2` cache and verifies the
pinned archive and extracted-file hashes. A [direct ZIP download](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/e002d649bb292d3e18c3a97d5b5c22c8ddcc81a3/robot_assets/AgibotG1ObservedReplayV2.zip)
is also available; extract it into a new directory and check its `SHA256SUMS`.
The archive contains root-level `robot.urdf`, its visual/collision meshes,
license, model provenance, README and checksums. Recording metadata, trajectory
arrays and native validation reports/traces are only in the complete package.
The model's URDF and all 63 mesh resources match the complete package's
`robot_physical/` directory.

The model retains CC-BY-NC-SA-4.0 attribution and remains separate from the
original `AgibotG1` USD preset. The complete input bundle retains its original
per-sample licenses and model-resource disclosures.

## AgiBot prepared-input collision repair (#1212)

The `03_agibot_toy` URDF in the
[pinned prepared-input package](https://huggingface.co/datasets/DestoFaygo/embodichain-real2sim-inputs/tree/1b34442c9e3d3e6a3139d3df95d5cb0e61a93e1f)
contains 41 visuals and no collisions. Loading, FK and file checksums do not
establish physical replay eligibility. Physical execution now checks the URDF,
mesh resources and declared gripper contact links before starting inference or
native simulation; visual-only models remain available for nonphysical input
loading. Replay never synthesizes collision geometry automatically.

In an installed development environment, explicitly prepare a **new** package:

```bash
curl -L --fail \
  'https://huggingface.co/datasets/DestoFaygo/embodichain-real2sim-inputs/resolve/1b34442c9e3d3e6a3139d3df95d5cb0e61a93e1f/real2sim_inputs_5.zip' \
  -o real2sim_inputs_5.zip
unzip real2sim_inputs_5.zip -d real2sim_inputs_original
python -m examples.real2sim.prepare_agibot \
  real2sim_inputs_original real2sim_inputs_physical --native-check
```

Use `real2sim_inputs_physical/samples/03_agibot_toy` as `REAL2SIM_SOURCE` for
subsequent registration and replay. Omit `--native-check` for offline preparation;
the resulting report explicitly says native qualification has not run. The
native check uses the default backend without rendering or model services.

The model-owned recipe in
`embodichain_robots/agibot/g1/_replay_collisions.py` accepts only the pinned input
URDF. It checks official source hashes and link-local geometric agreement before
transferring 14 arm cylinders from the
[official URDF](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/37e760fdd15d27c45233baf11811059ecc65ee9d/G1_120s/G1_120s.urdf)
and 22 gripper meshes from the
[official USD collision prims](https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee/robot/G1_120s).
This is an explicit derived physical model: the USD disables arm collisions,
whereas this recipe enables the official URDF cylinders. Native URDF mesh
cooking also differs from the USD's authored convex-decomposition settings on
four support links. These differences are recorded, not treated as an identical
USD dynamics model. The CC-BY-NC-SA-4.0 license and attribution accompany the
derived resources. `--source-bundle` accepts an already downloaded, hash-verified
official bundle with `old-G1_120s.urdf` for offline use.

The output preserves the original robot directory, videos, source trajectories,
timestamps, TCP definitions and gripper calibration. `robot.json` selects the
separate `robot_physical/robot.urdf`; `input.json`, `manifest.json`,
`input_validation.json` and `SHA256SUMS` record the new geometry identity.
`robot_physical/collision_provenance.json` records official revisions and the
conversion. `physical_validation/preflight.json` records structural validation;
`physical_validation/native/report.json` and `trace.npz` record native collision
import, finite stepping, calibrated opening/closing, sustained bilateral pad
contact and release. Attaching a report verifies its complete input hashes,
including calibration, initial trajectory and URDF inertials, before updating
qualification status and package checksums. The command checks all 36 native
collision-bearing links
and every calibrated pad. These checks do not establish grasp stability,
hardware validity, or successful execution of the toy-placement task. They also
do not add Task Program/DemoSkill deployment integration.

## How to run

### 1. Install and configure services

Start with a working EmbodiChain development environment, including DexSim and
the isolated Blender worker. Follow the repository's
[installation guide](../../docs/source/quick_start/install.md), including its
package indexes and GenSim setup. Then install the additional dependencies:

```bash
python -m pip install -e '.[gensim,geometry]'
python -m pip install -r examples/real2sim/requirements.txt
```

The supplied [services.json](services.json) uses an authenticated local Codex CLI
for scene understanding and HTTP services for segmentation and asset generation.
The trajectory agent also uses that CLI; GitHub authentication does not configure
model access. Choose endpoints for your deployment:

```bash
export SAM3_BASE_URL='http://sam3-host:port'
export SAM3D_BASE_URL='http://sam3d-host:port'
export ARTICRAFT_BASE_URL='http://articraft-host:port'
export IMAGE_EDIT_BASE_URL='http://image-edit-host:port'
```

Replace the placeholders before running. The image editor must implement
`POST /generate_image_by_image` with image/prompt/width/height/seed form fields
and a PNG response. SAM3 and SAM3D use the repository's segmentation and geometry
adapters. Articraft is used when Scene Engine identifies articulated objects.
See the [model services guide](../../docs/source/features/generative_sim/model_services.md)
for a different service configuration. Image editing is selected separately by
`--image-edit-url`.

Set a prepared recording, a new output root, and the observed task instruction:

```bash
export REAL2SIM_SOURCE='/absolute/path/to/prepared_recording'
export REAL2SIM_OUTPUT='/absolute/path/to/new_real2sim_run'
export REAL2SIM_INSTRUCTION='Pick up the banana and place it on the plate.'
```

For scene reconstruction alone, only the video is needed. Registration and
trajectory execution require the [prepared recording](#prepared-recording-contract)
described below; this example does not download or import raw datasets.

### 2. Reconstruct the scene

```bash
python -m examples.real2sim \
  --video "$REAL2SIM_SOURCE/original_front.mp4" \
  --output "$REAL2SIM_OUTPUT/reconstruction" \
  --services examples/real2sim/services.json \
  --image-edit-url "$IMAGE_EDIT_BASE_URL"
```

Optionally append `--reference-frames 30 90` with indices selected for **your**
video. These are zero-based decoded frame indices from the same fixed camera,
not trajectory sample indices. Select frames that reveal task objects near
contact events while retaining the intended initial scene state. No automatic
keyframe selector runs here.

The stage decodes frame zero, optionally recovers occluded pixels from the
selected frames, removes robot/background/lighting clutter through image editing,
and passes the resulting image to the complete native Scene Engine. It does not
substitute a separate object reconstruction pipeline.

The image editor receives a 32-aligned canvas with a long side of 768–1024 pixels.
Provider padding is removed, and the cleaned image is **not** reduced back to a
low-resolution source frame. Inspect
`reconstruction/scene_input/image_edit/scene_input.png` and
`reconstruction/scene_engine/scene_export/` before continuing. Unseen surfaces
completed by the image model are estimates.

### 3. Register the exported scene

This step uses the original RGB-D observation and camera calibration to estimate
one shared scene transform. It preserves generated shapes and relative object
placement. Keep the full Scene Engine generation directory: registration needs
its segmentation masks as well as the exported assets.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from embodichain.data_pipeline.real2sim.adaptation.prepared import load_prepared_source
from embodichain.gen_sim.real2sim.scene import register_scene_export

source = Path(os.environ["REAL2SIM_SOURCE"])
output = Path(os.environ["REAL2SIM_OUTPUT"]).resolve()
inputs = load_prepared_source(source, os.environ["REAL2SIM_INSTRUCTION"])
native_run = output / "reconstruction/scene_engine"
scene = register_scene_export(inputs, native_run)
registered = output / "registered"
registered.mkdir(parents=True, exist_ok=False)
(registered / "scene.json").write_text(
    json.dumps(scene.description, indent=2, allow_nan=False) + "\n"
)
np.savez_compressed(registered / "scene_vertices.npz", **scene.vertices)
(registered / "scene_export_source.json").write_text(
    json.dumps({"path": str(native_run / "scene_export")}, indent=2) + "\n"
)
print(registered)
PY
```

`native_run` points to `scene_engine`, **not** its `scene_export` child. Without
usable calibrated depth, supply an already registered scene in the format below.
An RGB video alone cannot provide this registration.

### 4. Annotate, replay, and correct the trajectory

```bash
python -m examples.real2sim.agent \
  "$REAL2SIM_SOURCE" \
  "$REAL2SIM_OUTPUT/registered" \
  "$REAL2SIM_OUTPUT/execution" \
  --instruction "$REAL2SIM_INSTRUCTION" \
  --model gpt-6-astra \
  --reasoning-effort medium \
  --max-revisions 2
```

The first replay uses the original joint trajectory. The agent annotates observed
semantic phases, reviews replay evidence, and proposes a few Cartesian/gripper
correction parameters. Numerical code produces the corrected commands; the agent
cannot supply joint arrays or edit the scene. Native `play_joint_trajectory`
executes the commands with physical contacts. Object poses are initialized once;
the task does not attach or teleport objects to follow the recording.

`--max-revisions 2` permits the baseline plus at most two corrected executions,
stopping on the first agent-reported success. Set it to `0` for annotation and one
baseline replay only. Completion of a process or an IK solve does not mean the
task succeeded; inspect `execution/result.json`, `review.json`, and the videos.

Optional flags:

- `--place-base`: explicitly recompute robot mounting before the loop to reduce
  overlap with the fixed scene. The default preserves the registered mounting
  (identity when absent). This option moves the robot base, not scene objects.
- `--visual-urdf /path/to/visual.urdf`: replace replay visuals while retaining the
  physical robot's joints and collision geometry.
- `--initial-trajectory /path/to/reference.npz`: start from existing finite
  `qpos` matching the source shape. Any supplied timestamps and joint names must
  match the recording.
- `--continue-from /path/to/completed_run`: reuse the exact prior annotation and
  completed replay, then execute only newly corrected trajectories. Input hashes,
  recording, scene, instruction, and visual URDF must match. Each completed replay
  saves `iterations/<index>/replay_receipt.json`, binding its trajectory and physics
  rollout hashes. Changed artifacts or runs without this receipt require a fresh
  replay. This option cannot be combined with `--place-base` or `--initial-trajectory`.

The numerical solver is the feature-specific approximate IK in
`embodichain/data_pipeline/real2sim/adaptation/_real2sim_ik.py`. Its residuals and
URDF limit excess are diagnostics; it can return a finite approximate solution.
Native replay separately clips commands to actual URDF limits and records those
clips. This is not a new registered EmbodiChain solver or a guarantee of physical
task completion.

### 5. Inspect results or replay an existing package

The latest completed execution is available at:

```text
execution/videos/comparison_full.mp4   original front view | simulation
execution/videos/simulation.mp4        simulation front view
execution/videos/overview.mp4          simulation overview
execution/result.json                 loop status and latest artifact paths
```

`videos` is a symlink to the latest completed iteration's video directory.
Rendering defaults are 640×480 for the front view, 480×360 for the overview,
10 fps, and one sample per pixel. These lightweight replay settings are separate
from the reconstruction image resolution.

To replay the baseline package without calling the agent or reconstruction
services:

```bash
python -m examples.real2sim.replay \
  "$REAL2SIM_OUTPUT/execution/iterations/000/native_inputs" \
  "$REAL2SIM_OUTPUT/replay_baseline"
```

Pass `--renderer no-render` for physics and diagnostics without videos. The replay
input is `native_inputs`, not the original recording directory. Use each command's
`--help` for its arguments.

## Prepared recording contract

The loader accepts a prepared, calibrated recording of the **same robot** used for
replay. Preserve the original video, measured joints, and calibration files.

```text
prepared_recording/
  robot.json
  robot_visuals.json
  trajectory_original.npz
  original_front.mp4
  first_front_original.png           first_front.png is also accepted
  depth_first.npy                    required for RGB-D scene registration
  robot.urdf                        path selected by robot.json; meshes must resolve
  original_left.mp4                  optional auxiliary view
  first_left_original.png            optional auxiliary first frame
  depth_left_first.npy               optional auxiliary depth
```

The same optional naming applies to `right`. The primary view is `front`.

| `trajectory_original.npz` key | Contract |
| --- | --- |
| `qpos` | Finite `(T, J)` array; radians for revolute joints, metres for prismatic joints |
| `timestamps` | Increasing `(T,)` sample times in seconds |
| `joint_names` | `(J,)` string array in command order; all names must exist in the URDF; no pickle/object array |
| `source_frame_indices` | Optional `(T,)` explicit decoded-video frame map; defaults to `arange(T)`; the prepared loader shares this map across views |
| `front_K` | `(3, 3)` original-image camera intrinsics |
| `front_camera` | `(N, 4, 4)` optical-camera-to-robot-base poses; the loader uses the first pose |
| `left_K`, `left_camera`, `right_K`, `right_camera` | Corresponding optional view calibration |
| `acquisition_timestamps` | Optional RH20T acquisition clock; for `robot_type` beginning `rh20t-`, this becomes the relative trajectory clock |

First images must match the decoded first observation. Depth arrays are aligned
to those original images, in metres, with zero for invalid pixels. Registration
maps high-resolution reconstruction masks onto this original calibration grid.
For RH20T, replay packaging preserves the original observation clock separately.

`robot.json` declares `urdf`, `robot_type`, `side_names` (or `prefixes`),
`tcp_links`, and the calibrated 4×4 `flange_to_tcp` transform. It can additionally
declare `arm_joint_groups`, measured `gripper_calibrations`,
`pad_center_tcp_z_m`, and `max_half_open_m`. The default layout assumes one or two
six-axis arms with paired prismatic jaws. Other layouts need explicit arm groups
and gripper calibration; robot-specific calibration cannot be inferred from the
task video.

`robot_visuals.json` contains `{"visuals": [...]}` records with `link`, mesh `path`,
and local `origin` matrices. URDF and visual paths may be absolute or relative to
the recording directory. An empty visual list is accepted by the loader.

A registered scene directory contains `scene.json`, `scene_vertices.npz` keyed by
object ID, and `scene_export_source.json` pointing to the native `scene_export`.
`register_scene_export()` creates the description and vertex data; step 3 persists
them. Mesh paths must remain accessible. A previously established
`robot_base_pose_world` in the scene description is used by execution.

## Output organization and interpretation

```text
new_real2sim_run/
  reconstruction/
    first_frame.png
    input.json, status.json, service_calls.json
    scene_input/                     recovery evidence and high-resolution image edit
    model_calls/                     scene model request artifacts
    scene_engine/                    complete native Scene Engine run
      scene_export/                  generated assets and scene configuration
  registered/
    scene.json, scene_vertices.npz, scene_export_source.json
  execution/
    context.json, fixed_inputs.json, robot_model.json
    scene.json, scene_vertices.npz, source_trajectory.npz
    annotation/                      Task Program JSON and source sample bindings
    iterations/000/
      trajectory.npz, evidence.json, review.json
      trajectory/trajectory_refined.npz
      native_inputs/                 replay package referencing source assets/video
      native_render/
        videos/                      comparison, front simulation, overview
        assets/                      runtime robot and collision assets
        diagnostics/                 measured physics, contacts, limits, clocks
        report.json
      replay_command.json, replay.log
    videos -> iterations/<latest>/native_render/videos
    status.json, result.json
```

Later iterations additionally retain correction proposals and any numerical
rejection evidence. Packages use absolute paths and symlinks; keep the source,
native scene export, and prior run when reusing them. This is a local run layout,
not a self-contained distribution archive.

Annotation uses the canonical Task Program JSON schema and separate half-open
trajectory sample bindings `[start, stop)`. `annotation/scope.json` marks the
result as observation annotation with `execution_ready=false`. Runtime bindings,
executable graph deployment, and trajectory augmentation are outside this example.
The agent's success assessment is retained as review evidence, not independent
task certification.

## Earlier contact-fit entry points

`gen_sim.real2sim.pipeline.run_real2sim()` is the earlier contact-fit composition,
with optional source graph annotation. It is also implemented by this feature;
it is **not** an alias of `execution.run_prepared_real2sim()` and does not run this
bounded replay loop. `python -m examples.real2sim.refine` wraps
`execution.refine_prepared_recording()` and requires an existing
`REGISTERED/grounding/grounding.json`. These paths remain available for existing
callers; the staged commands above describe the current agent workflow.

## Validation

The focused software tests use synthetic fixtures and injected model/replay
boundaries. They do not call generation services or establish real-task success:

```bash
python -m pytest -q \
  tests/data_pipeline/test_real2sim_adaptation.py \
  tests/data_pipeline/test_real2sim_hand_geometry.py \
  tests/data_pipeline/test_real2sim_placement.py \
  tests/data_pipeline/test_real2sim_trajectory_tools.py \
  tests/gen_sim/test_real2sim_composition.py \
  tests/gen_sim/test_real2sim_agent_refinement.py
```
