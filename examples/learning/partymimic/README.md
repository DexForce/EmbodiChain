# RP1 PartyMimic carrying demos

RP1 carries a 2 kg crate along level ground and a 1.5 kg crate onto a 48 cm
platform in the RP Hand video profile. EmbodiChain advances Newton at 200 Hz
and evaluates the official sparse5 actor through CPU ONNX Runtime at 50 Hz.
Each crate has its prescribed mass and inertia and two wrist welds.
[PartyMimic carrying-loads demo](https://roboparty.github.io/PartyMimic/#agent-load)

The recordings use a low third-person camera at 1280 × 720 and 50 fps, with
a checkerboard floor, a brown crate and a gray platform.

## Prepare and run

Use an EmbodiChain Python environment with DexSim's Newton and USD support.
The recommended command uses CUDA for DexUni and offscreen capture. The
preparation command requires Git and `uv`. Install the runtime packages if needed.

~~~bash
python -m pip install onnxruntime imageio imageio-ffmpeg
~~~

Run from the EmbodiChain checkout. The preparation command downloads the
demo ONNX and its matching deployment YAML from one pinned Hub revision.
[RP1 sparse5 model files](https://huggingface.co/DexForceAI/embodichain_model/tree/0a5bf9a5ad1e89061bf68d25d769407f79ca5492/policies/rp1-partymimic-sparse5)

~~~bash
embodichain run-example learning/partymimic/prepare \
  --output-dir outputs/partymimic

embodichain run-example learning/partymimic \
  --policy outputs/partymimic/rp1_sparse5.onnx \
  --solver dexuni --scenario all --step-reference video --hand rp \
  --output-dir outputs/partymimic-dexuni-rp-hand
~~~

Preparation caches the pinned PartyMimic checkout, RP1 meshes and converter
in `~/.cache/embodichain/partymimic`. It writes the ONNX, deployment YAML and
a provenance receipt with hashes to the output directory. The published files
were generated with the pinned PartyMimic exporter and the MIT-licensed RP1
kneeling and crawling clips. [PartyMimic MIT notice](resources/LICENSE.PartyMimic)

To export a local sparse5 checkpoint, pass `--checkpoint /path/to/checkpoint.pt`
to the preparation command. When ONNX Runtime is installed, preparation checks
the export using the temporary `.check.npz` and records the error in its receipt.
[Official exporter](https://github.com/Roboparty/PartyMimic/blob/c3da040573cb9ce06dfa6a2ee007873dcec0c076/tools/export_deploy_onnx.py)

Select `walk` or `step-up` to run one scene. Use `--no-video` for a physics
check and `--unloaded` to evaluate the same reference with zero payload.
The `mujoco_warp` solver option uses native MuJoCo on CPU and MuJoCo Warp on CUDA.

## Official RP Hands

`--hand rp` uses the hands from the pinned RoboParty RP1 description, including
the original wrist mounts, meshes, collision capsules, masses and inertias.
The example uses the official finger coupling curves to set a fixed grasp and
combines the two hand assemblies with their respective wrist links. The ONNX
actor drives the 28 RP1 joints. The wrist welds carry the payload.
The video step reference raises the lead foot, advances it over the edge,
then transfers weight toward that foot and brings up the trailing leg.
[RoboParty RP1 hand model](https://github.com/Roboparty/rp1_description/blob/7b7d807e52cbec23f22f36444ed3129f082d76ca/mjcf/robot_hand.xml)

`--hand proxy` selects the original simplified white hand visuals, sized to
the published DH116S envelope. [Leadshine DH116S dimensions](https://www.leadshine.com/robots/dextrous-hands/dh116s-series.html)

## Scene and solver

DexUni defaults to CUDA for the scene's authored USD collision masks. The
robot and crate share its MuJoCo partition (`pure_mujoco`) to retain both wrist
welds. The example checks the welds after initialization and reports the
resolved mode and VBD body count.

The `video` step reference leads with the right foot, turns the torso, and
brings the left foot onto the platform. Sagittal reflection maps the right-lead
command into the policy's left-lead frame and maps the joint targets back to
the robot. `--step-reference baseline` selects the left-lead reference.

## Evidence

Each rollout writes a USD scene and, with capture enabled, a third-person MP4.
The combined `metrics.json` records hashes, Newton's measured payload mass,
displacement, falls, maximum weld position error, support contacts and control
loop latency. `onnx_max_abs_error` contains the measured export error when
present in the provenance receipt and `null` otherwise. The walk must travel
3.50 ± 0.15 m in 400 commands. The step must
finish with at least 0.50 s of simultaneous foot contact on the platform top.
Both must finish upright with each weld within 3 cm of its configured position.
The clearance check measures the wrist and hand geometry in the crate frame at
reset and each 50 Hz control sample. Overlapping axis projections are resolved
with MuJoCo's signed geometry distance. The loaded RP Hand video profile on
CUDA also checks for platform-side impacts, crate contact with the platform,
and a return of the right foot to the floor after lift and before planting.
RP Hand runs record the pinned hand asset revision and MJCF hash.
The command returns a nonzero status if a scene fails.

The two reference motions and scene dimensions are reconstructions from the
published footage. The RP Hand video profile uses a 0.56 × 0.65 × 0.48 m platform
estimated from its projected edges and the RP1 foot geometry. The metrics record
first sustained foot contacts and torso yaw during the step.
[Carrying loads](https://roboparty.github.io/PartyMimic/#agent-load)

The crate uses the published payload masses. Its reconstructed dimensions are
0.29 × 0.33 × 0.30 m, with a 0.315 m width in the RP Hand video step profile for
finger clearance during the weight transfer. The footholds, weight transfers
and chest-height carrying pose are authored against the upstream RP1 kinematics.
The closed-loop simulator supplies the measured states for the policy and
the camera recordings.

## Policy interface

`PartyMimicPolicy.from_onnx(path)` loads the official graph and its adjacent
YAML. `joint_indices()` maps from simulator to policy order. Call `reset()` at the
initial state and `step()` with each measured state and 11-frame reference
window. The public API accepts world-frame xyz plus xyzw poses and converts
them to the exported graph's convention. The output is a position target in
radians for each policy joint.
