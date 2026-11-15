# Precision assembly examples

See the [Precision Assembly guide](../../../docs/source/features/toolkits/precision_assembly.md)
for functional entry points, embedded native recordings, reproducible commands
and the original mjfit paper citation.

The project selects DexSim 0.6.0rc2. Python 3.12 direct-compound construction,
raw-contact checks and native RGB previews were verified with rc2 using the
installed SDK. The static compound USD import route is unavailable
on 0.6.0rc1 and 0.6.0rc2: its parser drops child colliders under disabled rigid bodies.
EmbodiChain rejects that loss explicitly. Exported USD retains the authored
static ownership; use direct manifests for runtime construction until the SDK
parser is corrected. This dependency gap is retained as one strict expected
failure in the USD round-trip tests.

The native path attaches multiple convex collision pieces to one rigid body.
An independent visual mesh never closes its collision aperture. The generator
uses the separately installed optional mjfit provider pinned to revision
`5833f465896e1e06bc4be4f003be847f5183a4db`. On non-Windows SDK profiles,
DexSim installs `newton[sim]`, which owns the compatible MuJoCo version.
The separately installed generator also declares its MuJoCo dependency.
Profiles without Newton need MuJoCo installed explicitly for standalone MJCF
import; loading prepared manifests does not. The Git installation retains
the required generator revision provenance:

```bash
pip install .
pip install 'mjfit @ git+https://github.com/younghyopark/mjfit.git@5833f465896e1e06bc4be4f003be847f5183a4db'
embodichain run-example sim/assembly/asset_gallery --output outputs/assembly_gallery
embodichain run-example sim/assembly/asset_gallery --configs circle usb_c --collision-colors
embodichain run-example sim/assembly/drop_peg --backend mujoco_cpu
embodichain run-example sim/assembly/thread_pair --backend mujoco_cpu
embodichain run-example sim/assembly/gear_pair --backend mujoco_cpu
```

The reusable API lives in `embodichain.toolkits.assembly`, independently of
GenSim. Use `build_assembly` or `build_many_pegs` during asset preparation, then
load an `AssemblyAsset` at runtime. Native conversion belongs to
`embodichain.toolkits.assembly.native`. Its `prepare_native_configs` prepares
rigid and hinged bodies together, including optional file-backed appearance,
while preserving authored collision pieces and mass properties. Precision measurements belong to
`embodichain_benchmarks.assembly`. GenSim pipelines can consume the same toolkit.

The gallery writes self-contained manifests and USD files. A manifest can be
loaded and constructed without mjfit or MuJoCo installed. Raw MJCF conversion
requires MuJoCo. Names, source poses, explicit mass/COM/inertia, motor metadata,
and source contact parameters remain in the manifest.

Motion planning uses a separate convex proxy for each analytic cylinder and
retains the other collision pieces individually. cuRobo currently rejects a
batch whose prepared environments have different collider geometry; it can
still plan with per-environment obstacle poses when the geometry is shared.

The dynamic visual demo uses the same physical runner as the precision probes.
Viser shows the native simulation; GIF/MP4 files render recorded physical poses
after simulation. Playback speed changes only presentation, not the physics
timestep. Install `matplotlib` and `pillow` for recording; MP4 also needs FFmpeg.

```bash
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --viser --startup-delay 10 --hold-seconds 60
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --collision-colors --record gif --output outputs/assembly_visual/circle
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --route usd --record mp4 --output outputs/assembly_visual/circle_usd
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --peg-offset .02 --record gif --output outputs/assembly_visual/blocked
embodichain run-example sim/assembly/visual_demo --config m10 --duration 1.5 --playback-speed .4 --record mp4 --output outputs/assembly_visual/thread
embodichain run-example sim/assembly/visual_demo --config spur_pair --backend mujoco_cpu --duration 1.5 --playback-speed .4 --record gif --output outputs/assembly_visual/gear_reference
```

The viewer binds to loopback with physics-changing commands disabled. In Viser,
zoom into the small assembly and hide `/world` in the scene tree if the large
reference axes obscure it. Viser currently assigns one category color to a
rigid object; per-piece collision colors are available in the recorded view
and inspection USD. The offset peg is an intentional blocking control. Success
requires seating through the plate and lateral alignment, not just falling the
initial spawn gap. Source gear PGS remains unsupported by MuJoCo Warp. Select `--solver cg`
explicitly for the measured Warp comparison profile. The independent CPU
reference supports `--backend mujoco_cpu --solver pgs`. Historical Newton CPU
PGS profiles are retained in the runtime archive linked below. Generated traces
record the executor, selected solver and measured outcome.

Newton solver selection is explicit: `--newton-solver mujoco_warp|dexuni`.
Omission preserves MuJoCo Warp. `--solver pgs|cg|newton` continues to select the
MuJoCo algorithm. The Newton selector requires `--backend newton`.

```bash
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --newton-solver dexuni --native-window --native-record --renderer fast-rt --output outputs/assembly_native_window/dexuni
embodichain run-example sim/assembly/gear_pair --newton-solver dexuni --solver cg --duration .5
python scripts/benchmark/assembly/precision.py --configs circle --backends newton --routes direct --newton-solvers mujoco_warp dexuni --duration .2
```

`PrecisionRunCfg.newton_solver` and `run_native_probe(..., newton_solver=...)`
expose the same selection in the packaged benchmark API. DexUni routes MuJoCo
options through `mujoco_options`. JSON results retain the requested configuration,
actual Newton family and SDK dispatch diagnostics; reports keep the two families
separate, including failures. These assembly profiles require DexUni's measured
`pure_mujoco` dispatch; missing diagnostics or a VBD/coupled dispatch is explicitly
unsupported. Native capacity/precision qualification remains `unverified`.

For the native DexSim rendering window, use a desktop display and a supporting
SDK runtime. The rc2 RGB recordings below use FastRT, 16 samples per pixel and
OptiX denoising, with a camera framed around the centimetre-scale part. Hybrid
is also selectable; its native recording crashed under this remote Xvfb profile.

```bash
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --collision-colors --native-window --native-record --renderer fast-rt --startup-delay 2 --hold-seconds 10 --output outputs/assembly_native_window/circle
embodichain run-example sim/assembly/visual_demo --config m10 --duration 1.5 --playback-speed .4 --native-window --native-record --renderer fast-rt --startup-delay 2 --hold-seconds 10 --output outputs/assembly_native_window/m10
```

`--native-record` writes native RGB PNG frames and `native_window.mp4`. The
actual window remains open throughout the physical rollout and the requested
hold time. The current SDK's Hybrid/FastRT window `get_picture()` has no pixel
readback implementation, so recording uses a native Camera sensor with the
same scene, pose, projection and clipping planes as the window. These previews
use DexSim rendering; they do not use the offline mesh renderer. Four camera
warm-up frames at the unchanged initial pose initialize render buffers without
advancing physics. The trace records the capture source, camera parameters,
physical sample steps and captured frame count. Movie export is bounded by
this run's frame count to avoid including stale frames after a shorter rerun.

Native direct previews prepare a file-backed OBJ/MTL visual per body through
`prepare_native_configs(..., file_visuals=True)` in the native toolkit adapter. This avoids the installed SDK's failure
to publish large procedurally appended render bodies. The OBJ groups source
triangle surfaces by material; it does not supply collision geometry. M10
therefore renders the bolt as two material meshes while Newton retains all
339 bolt colliders and 186 nut colliders. Source vertices and local transforms
are preserved, and primitive appearance uses the SDK's existing tessellation.
The underlying procedural-render construction issue remains an SDK limitation.

OBJ visuals explicitly author normals: source convex meshes retain flat face
normals at hard edges, while primitive cylinders keep the SDK's smooth side
normals and independent cap normals. USD meshes use matching `faceVarying`
normals. This prevents the importer from rounding hex heads, denting nut faces
or making flat hole plates look corrugated. The native visual cache directory
includes the visual format version (`visual_v2`); regenerate old exported
assets to obtain these normals. Geometry, collision pieces and mass properties
remain unchanged.

Newton's raw source contacts and preparation-time rigid variants require an SDK
exposing the matching public descriptor fields. Unsupported builds fail
explicitly. Development-only
`--backend-contacts` selects native default contacts and is recorded as such.
Default cannot reproduce raw MuJoCo contact semantics; it is a measured
comparison backend. Default's cylinder cooking also uses a polygonal proxy.

```bash
embodichain run-example sim/assembly/drop_peg --backend newton --route direct
embodichain run-example sim/assembly/drop_peg --backend newton --route usd
embodichain run-example sim/assembly/gear_pair --backend newton --solver cg
```

The thread example prescribes bolt rotation, retaining contact-driven axial
advance. The gear example applies torque only to the driver; follower motion
comes from tooth contact and both gears retain actual hinges.

The precision matrix runs native cells in separate processes and retains
failed and unverified results:

```bash
python -m scripts.benchmark.assembly.precision --configs circle usb_c m10 spur_pair
python -m scripts.benchmark.assembly.precision --configs circle --num-envs 1 64 256 1024 --timesteps .001 .0005 .00025
python -m scripts.benchmark.assembly.precision --configs circle --probe S2 --routes direct --backends mujoco_cpu mujoco_warp newton
python -m scripts.benchmark.assembly.precision --configs circle --clearances .00002 .00005 .0001 .0002 --probe S2 --routes direct --backends mujoco_cpu
```

Each invocation creates a fresh `run_*` directory below `--output`, containing
three comparison tables, companion raw JSON and isolated worker logs. Geometry
or USD ownership failures remain visible in the report while independent
source/direct cells continue. Worker crashes and mismatched output identities
are reported as failures; earlier runs are never reused.
Unmeasured memory and unavailable overflow accounting are marked explicitly.
Passive success is distinct from S0/S2/S3/S4 clearance qualification.
The constrained reference probes in `embodichain_benchmarks.assembly.probes` use
analytic circle/square ground truth and real slide/hinge DOFs; unsupported
outlines are rejected rather than measured against the wrong oracle.
Constrained probes disable the global ground. Passive probes author the source
floor's friction and contact parameters through `ground_plane_attrs`; this
avoids measuring a different floor under an otherwise identical assembly.

The pinned historical reference environment is described in
[the archived reference environment](https://github.com/DexForce/EmbodiChain/blob/e15c945a8976b06b447f933400d12e5bd936dd3a/scripts/benchmark/assembly/reference-requirements.txt).
It is independent of the installed EmbodiChain runtime; each run records its
actual versions. Generated measurements and recordings belong in `outputs/`.
These examples do not establish paper-level training or precision qualification.

For heterogeneous cells in one scene:

```bash
embodichain run-example sim/assembly/many_pegs --native-window --output outputs/many_pegs
```


Constrained S2–S4 profiles refine each requested interval into four physical
integration steps by default (`--integration-substeps 4`). CPU, raw Warp and
native routes use the same effective interval and score every settled physical
step. At a requested 1 ms interval the actual interval is 0.25 ms; duration,
forces and error budgets remain unchanged. `--integration-substeps 1` reproduces
the former coarse protocol. Reports retain the configured integration interval,
step count and per-direction sample count; finer integration increases work.
