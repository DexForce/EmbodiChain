(precision-assembly)=

# Precision Assembly

The `embodichain.toolkits.assembly` toolkit prepares peg/hole pairs, threaded
parts and gears as portable assets. Each body retains its own collision pieces,
visual surfaces, local frames, mass, center of mass and inertia. Simulation,
GenSim and task integrations can consume the same saved asset.

Use this toolkit when clearance between mating parts must survive asset
preparation. It preserves the generator's convex pieces instead of taking a
single hull around a concave hole or recomputing inertia from overlapping pieces.

## Original work

The optional [mjfit generator](https://github.com/younghyopark/mjfit) accompanies
**[Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse
Parts](https://arxiv.org/abs/2610.23849)**, by Antonia Bronars, Younghyo Park,
Kushal Kedia and Pulkit Agrawal (2026; Bronars and Park contributed equally).
The [authors' project page](https://younghyopark.me/blind-proprioceptive-assembly/)
contains the original paper videos and interactive policy demonstrations.

The paper studies policies that infer contact from joint/action histories.
It introduces procedurally generated assemblies with clearance-preserving
convex geometry. EmbodiChain integrates that asset generator and adds portable
manifests, native configuration preparation and independent measurements.
The passive demonstrations below exercise asset geometry and contact behavior;
they do not reproduce the paper's learned policies or its sim-to-real results.

## Functional entry points

Import portable data and generation functions from
`embodichain.toolkits.assembly`:

| Entry point | Purpose |
| --- | --- |
| `AssemblyAsset`, `AssemblyBody`, `AssemblyGeometry`, `AssemblyJoint` | Represent one asset, its body ownership, visual/collision parts and supported attachments. |
| `build_assembly(config, **overrides)` | Generate one named, YAML-defined or configuration-defined assembly. |
| `sample_assemblies(count, seed=..., family=...)` | Generate a seeded distribution of peg assemblies. |
| `build_many_pegs(count, seed=..., family=...)` | Generate one grid scene with independently sampled plate/peg pairs and distinct body identities. |
| `import_mjcf(xml, ...)` | Compile and import the supported MJCF subset while preserving authored frames and physical properties. |
| `asset.save(path)`, `AssemblyAsset.load(path)` | Write and load a self-contained, checksummed JSON manifest. |
| `asset.content_hash` | Identify asset content, including configuration and provenance. |
| `asset.write_usd(path)` | Export inspectable USD geometry, collision ownership, mass properties and supported hinge trees. |
| `load_mjfit()`, `MJFIT_REVISION` | Inspect and verify the optional generator's pinned provenance. |

Import native preparation explicitly from
`embodichain.toolkits.assembly.native`:

| Entry point | Purpose |
| --- | --- |
| `prepare_native_configs(asset, directory, ...)` | Prepare rigid bodies and hinges together; optionally prepare file-backed rigid appearance. |
| `to_rigid_config(body, ...)` | Convert one free/fixed body to a compound rigid configuration. |
| `to_rigid_configs(asset, ...)` | Convert an entirely free/fixed asset; reject hinges explicitly. |
| `to_articulation_configs(asset, directory, ...)` | Prepare supported world-mounted hinges as actual articulations with generated URDF/visual files. |

The toolkit root does not import the simulation runtime. Manifest loading
requires neither mjfit nor MuJoCo. The explicit native adapter creates
`RigidObjectCfg` and `ArticulationCfg` objects; `SimulationManager` owns their
runtime lifecycle. See the [Toolkits API reference](../../api_reference/embodichain/embodichain.toolkits.rst).

## Installation

Install EmbodiChain normally. Install the pinned provider separately when
**generating** assemblies:

```bash
pip install .
pip install 'mjfit @ git+https://github.com/younghyopark/mjfit.git@5833f465896e1e06bc4be4f003be847f5183a4db'
```

There is no `assembly` installation extra. SDK profiles using `newton[sim]`
obtain compatible MuJoCo/MuJoCo Warp versions through Newton. Other profiles
need MuJoCo explicitly to compile MJCF. Loading prepared manifests does not
compile MJCF. USD export additionally needs the SDK's USD Python bindings.

## Generate, save and reuse an asset

```python
from pathlib import Path
from embodichain.toolkits.assembly import AssemblyAsset, build_assembly

output = Path("outputs/assembly")
asset = build_assembly("circle", clearance=0.0001)
manifest = asset.save(output / "circle.json")
restored = AssemblyAsset.load(manifest)
assert restored.content_hash == asset.content_hash
restored.write_usd(output / "circle.usda")
```

Named configurations include `circle`, `star`, `usb_c`, `m10` and `spur_pair`.
`seed` records generation provenance; fixed configurations remain deterministic.
Use the sampler when a random distribution is required:

```python
from embodichain.toolkits.assembly import sample_assemblies, build_many_pegs

assets = sample_assemblies(8, seed=7, family="convex")
grid = build_many_pegs(9, seed=0, family="primitive")
grid.save("outputs/assembly/many_pegs.json")
```

A many-pegs grid is **one physical scene**, not nine independent simulation
environments. Its manifest records each cell's sampled configuration and pose.

## Prepare native configurations

```python
from embodichain.toolkits.assembly.native import prepare_native_configs

configs = prepare_native_configs(
    restored,
    "outputs/assembly/native",
    backend="newton",
    source_contact=True,
    file_visuals=True,
)
```

Pass each returned free/fixed configuration to `sim.add_rigid_object()` and
each hinge configuration to `sim.add_articulation()`, then call `sim.prepare()`.
A body remains one physical owner with multiple collision pieces. File-backed
OBJ/MTL appearance combines visible surfaces only; it does not fill holes,
replace colliders or change mass properties. An empty visual list remains hidden.

`source_contact=True` preserves supported MuJoCo contact parameters on the
Newton route. The Default backend requires an explicit
`source_contact=False` comparison. This changes the contact model and should
not be interpreted as an equivalent precision profile.

MJCF import is bounded to world-attached bodies with no joint, one free joint,
or one unlimited hinge. Hinge URDF preparation requires a body-centered anchor.
This is not a general MJCF scene/constraint importer.

## Demo gallery

These are **EmbodiChain native RGB recordings**, captured with a Camera matching
the native window. They use SDK `0.6.0rc2`, Newton/MuJoCo Warp, FastRT, a 1 ms
physics interval, one solver substep and source contact parameters. Physics is
slowed only for playback. The gear comparison explicitly selects CG because
the source's PGS setting is unsupported by MuJoCo Warp.

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/drop-peg.jpg" aria-label="Centered peg falling into a circular hole">
      <source src="../../_static/assembly/drop-peg.mp4" type="video/mp4">
      Your browser does not support embedded MP4 video.
    </video>
    <figcaption><b>Peg insertion:</b> approximately 10 mm insertion. <a href="../../_static/assembly/drop-peg.mp4">Download MP4</a>.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/many-pegs.jpg" aria-label="Nine independently sampled peg assemblies seating together">
      <source src="../../_static/assembly/many-pegs.mp4" type="video/mp4">
      Your browser does not support embedded MP4 video.
    </video>
    <figcaption><b>Many pegs:</b> 9/9 sampled pairs reach their expected seat. <a href="../../_static/assembly/many-pegs.mp4">Download MP4</a>.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/thread-pair.jpg" aria-label="M10 bolt rotating and advancing into its nut">
      <source src="../../_static/assembly/thread-pair.mp4" type="video/mp4">
      Your browser does not support embedded MP4 video.
    </video>
    <figcaption><b>M10 thread:</b> 1.802 mm advance; lead relative error 0.62%. <a href="../../_static/assembly/thread-pair.mp4">Download MP4</a>.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/gear-pair.jpg" aria-label="A driven gear transmitting motion to its passive follower">
      <source src="../../_static/assembly/gear-pair.mp4" type="video/mp4">
      Your browser does not support embedded MP4 video.
    </video>
    <figcaption><b>Spur gears, CG:</b> transmission-ratio relative error 0.18%. <a href="../../_static/assembly/gear-pair.mp4">Download MP4</a>.</figcaption>
  </figure>
</div>

### Reproduce the recordings

A native window needs a supported display/runtime; MP4 export needs FFmpeg.
Use a distinct output directory per run:

```bash
embodichain run-example sim/assembly/visual_demo --config circle --clearance .0001 --duration .2 --playback-speed .1 --native-window --native-record --renderer fast-rt --output outputs/assembly/circle
embodichain run-example sim/assembly/visual_demo --config many_pegs --count 9 --duration .25 --playback-speed .1 --native-window --native-record --renderer fast-rt --output outputs/assembly/many
embodichain run-example sim/assembly/visual_demo --config m10 --duration .5 --playback-speed .2 --native-window --native-record --renderer fast-rt --output outputs/assembly/thread
embodichain run-example sim/assembly/visual_demo --config spur_pair --solver cg --duration .5 --playback-speed .2 --native-window --native-record --renderer fast-rt --output outputs/assembly/gears
```

Add `--peg-offset .012` to the circle command for the blocked-rim control.
Add `--collision-colors` to inspect each convex piece. `--record gif` or
`--record mp4` uses the separate offline renderer of recorded physical poses;
`--native-record` captures the SDK-rendered RGB used above.

Other installed entry points are `sim/assembly/asset_gallery`, `drop_peg`,
`thread_pair`, `gear_pair` and `many_pegs`. For example:

```bash
embodichain run-example sim/assembly/asset_gallery --configs circle usb_c m10 spur_pair --collision-colors --output outputs/assembly/gallery
embodichain run-example sim/assembly/drop_peg --backend mujoco_cpu
```

### DexUni validation

Select DexUni explicitly through the installed demo CLI. Omission of
`--newton-solver` preserves the existing `mujoco_warp` profile:

```bash
embodichain run-example sim/assembly/visual_demo --backend newton --newton-solver dexuni --config circle --clearance .0001 --duration .2 --native-window --native-record --renderer fast-rt --output outputs/assembly/dexuni_circle
embodichain run-example sim/assembly/gear_pair --backend newton --newton-solver dexuni --solver cg --duration .5
```

`--newton-solver` selects the Newton solver family (`mujoco_warp`, `dexuni`).
`--solver` still selects the MuJoCo algorithm (`pgs`, `cg`, `newton`). The family
selector is rejected on the Default and independent MuJoCo reference backends.
`--use-mujoco-cpu` explicitly selects the single-world CPU integration inside
either Newton family; it never requests an automatic fallback.

Both families receive the same iteration limit, source algorithm/integrator,
contact capacity, and explicit tolerance/MultiCCD overrides. DexUni forwards
these under the SDK's `mujoco_options`. `source_contact=True` remains the default.
The traces retain `newton_solver`, `active_solver` (the MuJoCo algorithm), and
SDK `solver_diagnostics`, including its actual scene dispatch and step count.
The window/trace executor label identifies DexUni and the measured dispatch.

The following recordings use the formal CLI with the same manifests, camera
views, source contacts and controls as the gallery. The actual SDK diagnostic
is `features.backend="pure_mujoco"`, with VBD disabled. The assembly benchmark
requires this dispatch to honor its configured MuJoCo contact/algorithm profile;
missing diagnostics or a VBD/coupled dispatch produce an explicit unsupported
result. These checks do not qualify VBD-only or rigid/soft coupling.

<video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/dexuni.jpg" aria-label="DexUni native RGB comparison: peg, many pegs, thread and gears">
  <source src="../../_static/assembly/dexuni.mp4" type="video/mp4">
  Your browser does not support embedded MP4 video.
</video>

DexUni recordings, clockwise from top left: peg, many pegs, gears, thread.
[Download the DexUni MP4](../../_static/assembly/dexuni.mp4).
The circle insertion was approximately 10 mm and the many-pegs result was 9/9.
M10 advanced 1.794 mm with 0.19% lead error, compared with 0.62% above; gear-ratio
error was 0.15%. These routes are not numerically identical.

## Measurement and current limits

The separate `embodichain_benchmarks.assembly.precision` module exposes
`PrecisionRunCfg`, `compare_usd_geometry` and `run_passive_probe`.
`embodichain_benchmarks.assembly.probes` owns independent constrained clearance
probes. Select the family programmatically through `PrecisionRunCfg.newton_solver`:

```python
from embodichain_benchmarks.assembly.precision import PrecisionRunCfg, run_passive_probe

result = run_passive_probe(
    restored,
    PrecisionRunCfg(backend="newton", newton_solver="dexuni", duration=0.2),
)
```

The source matrix runner can compare both families in one invocation:

```bash
python scripts/benchmark/assembly/precision.py --configs circle --backends newton --routes direct --newton-solvers mujoco_warp dexuni --duration .2
python scripts/benchmark/assembly/precision.py --configs circle --backends newton --routes direct --newton-solvers mujoco_warp dexuni --probe S2 --directions 4 --duration .3
```

`--newton-solver` is also accepted as a singular alias by the matrix CLI.
When independent reference/Default backends are included, their cells run once;
the Newton solver sweep applies only to Newton cells. Reports group requested
families separately, display actual SDK dispatch independently, and retain failed
attempts in their requested profile. Each cell owns a separate preparation
directory. An unresolved/failed dispatch is labeled `unavailable`.

`run_native_probe(..., newton_solver="dexuni")` exposes the same family choice
for constrained S2–S4 probes. Native S0 remains explicitly unavailable because
there is no read-only collision-forward query. MuJoCo algorithm, CPU and
calibration CLI overrides apply to passive probes; constrained matrix calls
reject these overrides rather than ignore them.

The [recording metadata](../../_static/assembly/recordings.json) contains asset
hashes, profiles, camera parameters, measurements and collision counts.
A 0.3 s, four-direction S2 check at 100 μm clearance returned identical rows
for the published MuJoCo Warp baseline and both new family selections. Two
directions exceeded the 10 μm wall-error budget (worst error approximately
18.49 μm), so the report correctly retained `failed`. Solver selection does not
relax the accuracy threshold or turn a completed run into precision acceptance.

Behavioral success does not certify micro-clearance accuracy or solver capacity:
live native contact-overflow accounting remains unavailable, and native
precision qualification stays `unverified`.

SDK `0.6.0rc1` and `0.6.0rc2` drop child colliders when importing disabled static
USD compound bodies. EmbodiChain rejects that loss. Use direct native
configurations for the runtime demos; exported USD remains available for
inspection. Hybrid recording crashed under the tested remote Xvfb profile;
the measured recordings use FastRT.

## Citation

When using the upstream generator or its associated research, cite the original
work as requested by [mjfit](https://github.com/younghyopark/mjfit#citation):

```bibtex
@misc{bronars2026blindpreciseassemblyproprioceptive,
  title={Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts},
  author={Antonia Bronars and Younghyo Park and Kushal Kedia and Pulkit Agrawal},
  year={2026},
  eprint={2610.23849},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.23849}
}
```

mjfit is MIT licensed. This integration pins its source revision and records
that provenance in generated manifests. The clips on this page were recorded
in EmbodiChain; original research media remains linked on the authors' project
page.
