# Precision assembly examples

The native path attaches multiple convex collision pieces to one rigid body.
An independent visual mesh never closes its collision aperture. The generator
is an optional dependency pinned to mjfit revision
`5833f465896e1e06bc4be4f003be847f5183a4db`:

```bash
pip install '.[assembly]'
embodichain run-example sim/assembly/asset_gallery --output outputs/assembly_gallery
embodichain run-example sim/assembly/asset_gallery --configs circle usb_c --collision-colors
embodichain run-example sim/assembly/drop_peg --backend mujoco_cpu
embodichain run-example sim/assembly/thread_pair --backend mujoco_cpu
embodichain run-example sim/assembly/gear_pair --backend mujoco_cpu
```

The gallery writes self-contained manifests and USD files. A manifest can be
loaded and constructed without mjfit or MuJoCo installed. Raw MJCF conversion
requires MuJoCo. Names, source poses, explicit mass/COM/inertia, motor metadata,
and source contact parameters remain in the manifest.

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
explicitly for the measured Warp comparison profile. To retain the source PGS,
select `--use-mujoco-cpu --solver pgs` and apply the separate optional Newton
CPU patch described below. The executor and solver choice remain in each trace. See [VISUAL_VALIDATION.md](VISUAL_VALIDATION.md)
for actual measured cases and qualification limits.

For the native DexSim rendering window, use a desktop display and the patched
SDK runtime described below. The preview uses Hybrid rendering, 16 samples per
pixel and OptiX denoising, with a camera framed around the centimetre-scale part:

```bash
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --collision-colors --native-window --native-record --startup-delay 2 --hold-seconds 10 --output outputs/assembly_native_window/circle
embodichain run-example sim/assembly/visual_demo --config m10 --duration 1.5 --playback-speed .4 --native-window --native-record --startup-delay 2 --hold-seconds 10 --output outputs/assembly_native_window/m10
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
the private assembly visual adapter. This avoids the installed SDK's failure
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

For Newton's exact raw source contacts and preparation-time rigid variants,
apply the reproducible SDK patches described in
[`scripts/patches/README.md`](../../../scripts/patches/README.md). The unpatched
SDK fails explicitly when those capabilities are requested. Development-only
`--backend-contacts` selects native default contacts and is recorded as such.
Default cannot reproduce raw MuJoCo contact semantics; it is a measured
comparison backend. Default's cylinder cooking also uses a polygonal proxy.
Wheels include these patches under `share/embodichain/assembly_patches/` in the
installation prefix; source distributions retain `scripts/patches/`.

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

The report writes exactly three comparison tables and companion raw JSON.
Unmeasured memory and unavailable overflow accounting are marked explicitly.
Passive success is distinct from S0/S2/S3/S4 clearance qualification.
The constrained reference probes in `embodichain.gen_sim.assembly.probes` use
analytic circle/square ground truth and real slide/hinge DOFs; unsupported
outlines are rejected rather than measured against the wrong oracle.
Constrained probes disable the global ground. Passive probes author the source
floor's friction and contact parameters through `ground_plane_attrs`; this
avoids measuring a different floor under an otherwise identical assembly.

The pinned historical environment is reproducible from
[`reference-requirements.txt`](../../../scripts/benchmark/assembly/reference-requirements.txt).
It is independent of the installed EmbodiChain runtime; current-runtime
measurements record their actual versions. See [IMPLEMENTATION.md](IMPLEMENTATION.md)
for measured evidence and outstanding qualification limits.

## Official many-pegs scene and hinged gears

`many_pegs` uses the pinned public sampler/build/grid calls. Its default is the
official 25 primitive assemblies, seed 0 and 0.3 m grid spacing: 50 rigid bodies
with 1050 body collision pieces plus one host-owned floor. This is one physical
scene with 25 cells. It is separate from heterogeneous per-arena RL replication.

```bash
embodichain run-example sim/assembly/many_pegs --native-window --native-record --timestep .00025 --playback-speed .05 --output outputs/many_pegs
embodichain run-example sim/assembly/many_pegs --count 25 --clearance .0002 --max-error .00002 --contact-capacity 16384 --timestep .00025 --native-window --native-record --output outputs/many_pegs_200um
embodichain run-example sim/assembly/visual_demo --config spur_pair --solver cg --duration 1.5 --native-window --native-record --playback-speed .3 --output outputs/gears_warp_cg
embodichain run-example sim/assembly/visual_demo --config spur_pair --solver pgs --use-mujoco-cpu --duration 1.5 --native-window --native-record --playback-speed .3 --output outputs/gears_cpu_pgs
```

The CPU PGS command requires the optional
[`newton_cpu_pgs.patch`](../../../scripts/patches/newton_cpu_pgs.md) in addition
to the DexSim patches. It retains PGS, fixes CPU contact-property staging and
avoids a stale CPU midphase BVH. It does not add GPU PGS support. Original
dependency packages are not modified automatically. Both gear profiles retain
real unlimited hinges and apply torque only; the follower has no prescribed
angle or velocity. Source URDF passive damping/friction are retained separately
from PD gains. Native moving-link visuals use body-local OBJ/MTL normals.

`--camera-eye X Y Z` and `--camera-target X Y Z` override both the native window
and its matching recording camera. Default framing fits the grid or gear pair.
Manifests passed with `--config path/to/asset.json` load their fixed geometry
without generation. Curve accuracy overrides are preparation-only, and saved
assets record the original sampled configuration and actual curved shape specs.

A successful default grid does not establish micron-scale precision. The
20/50/100/200 µm precision profiles use chord error 0.1 times clearance and keep
all convex pieces. CPU and USD validation pass, but the current native stack
rejects a genuinely three-dimensional thin convex piece at 20 µm and has
nonfinite states in some refined profiles. These are recorded failures, not
replaced by coarser curves, removed pieces or enlarged clearances. Native
health unavailable through the Scene facade remains explicitly unverified.

The source/native gear benchmark commands are:

```bash
python -m scripts.benchmark.assembly.gear_reference --manifests outputs/gear_assets/*.json
python -m scripts.benchmark.assembly.gear_native --manifests outputs/gear_assets/*.json --profiles cpu_pgs warp_cg --output outputs/gear_alignment
```

The native benchmark confines private solver diagnostics to the experiment
harness and records warnings, overflow, runtime resets, actual model parameters,
static contact boundaries and full trajectories. Its diagnostic pass is kept
separate from the public Scene facade qualification. Timing with per-step
diagnostics or rendering is not a steady-state throughput measurement.

Solver calibration accepts `--solver-tolerance` and `--multiccd` /
`--no-multiccd`; omitted options preserve backend defaults. The benchmark
records requested and effective settings because Warp clamps small tolerances.
Independent Warp references additionally declare
`--warp-reference-initialization cpu_compiled|warp_recomputed`. The latter
executes public `mujoco_warp.set_const_0` before the initial forward, matching
Newton's GPU-derived constant initialization. Both protocols remain separate
results; CPU source constants are not silently overwritten to obtain a pass.
