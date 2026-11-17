(precision-assembly)=

# Precision Assembly

The precision assembly toolkit generates peg/hole pairs, threaded parts and
gears for simulation. It preserves each body's collision pieces, visual
surfaces, local frames and mass properties, including the clearance between
mating parts.

## Capabilities

- **Asset generation** — build named configurations such as `circle`, `star`,
  `usb_c`, `m10` and `spur_pair`, or sample a seeded distribution of parts.
- **Portable manifests** — save geometry, physical properties and generation
  provenance in checksummed JSON files.
- **Native preparation** — convert free/fixed bodies into compound rigid
  objects and supported hinges into articulations.
- **USD export** — write geometry, collider ownership and mass properties for
  inspection and interchange.
- **Physical examples** — run peg, thread and gear demonstrations with DexUni,
  and measure their behavior with independent assembly probes.

## Installation

Install EmbodiChain, then install the pinned mjfit provider to generate assets:

```bash
pip install .
pip install 'mjfit @ git+https://github.com/younghyopark/mjfit.git@5833f465896e1e06bc4be4f003be847f5183a4db'
```

Loading saved manifests requires neither mjfit nor MuJoCo. USD export requires
the SDK's USD Python bindings.

## Python API

### Generate and save assets

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

Sample individual assemblies or build a grid scene:

```python
from embodichain.toolkits.assembly import build_many_pegs, sample_assemblies

assets = sample_assemblies(8, seed=7, family="convex")
grid = build_many_pegs(9, seed=0, family="primitive")
grid.save("outputs/assembly/many_pegs.json")
```

A grid is one scene containing multiple plate/peg pairs. Use multiple
simulation environments when independent arenas are needed.

### Prepare simulation objects

```python
from embodichain.toolkits.assembly.native import prepare_native_configs

configs = prepare_native_configs(
    restored,
    output / "native",
    backend="newton",
    source_contact=True,
    file_visuals=True,
)
```

Add each free/fixed configuration with `sim.add_rigid_object()` and each hinge
configuration with `sim.add_articulation()`, then call `sim.prepare()`.
Native preparation preserves the original collision pieces and mass properties.

| Entry point | Purpose |
|---|---|
| `build_assembly()` | Generate one named or custom assembly. |
| `sample_assemblies()` / `build_many_pegs()` | Sample separate assets or compose a grid scene. |
| `import_mjcf()` | Import supported MJCF bodies and physical properties. |
| `AssemblyAsset.save()` / `.load()` | Write and read a checksummed manifest. |
| `AssemblyAsset.write_usd()` | Export a USD asset. |
| `prepare_native_configs()` | Prepare mixed rigid bodies and hinges. |
| `to_rigid_config()` / `to_rigid_configs()` | Prepare free/fixed bodies. |
| `to_articulation_configs()` | Prepare supported hinges. |

See the [Toolkits API reference](../../api_reference/embodichain/embodichain.toolkits.rst)
for complete signatures and data types.

## Command-Line Examples

These examples select the Newton backend with DexUni explicitly:

```bash
# Peg insertion.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config circle --clearance .0001 --duration .2

# Nine sampled assemblies in one scene.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config many_pegs --count 9 --duration .25

# M10 thread advancement.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config m10 --duration .5

# Gear transmission with the CG algorithm.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config spur_pair --solver cg --duration .5
```

| Option | Purpose |
|---|---|
| `--config` | Select the assembly configuration. |
| `--clearance` | Set mating clearance in meters. |
| `--collision-colors` | Display individual collision pieces. |
| `--peg-offset .012` | Add a lateral offset for a blocked-rim example. |
| `--native-window` | Open the native simulation window. |
| `--native-record --renderer fast-rt` | Capture native RGB; requires a native window and FFmpeg. |
| `--output` | Select a separate output directory for each run. |

`--newton-solver` selects the solver family; `--solver` selects its MuJoCo
algorithm. The asset-only gallery is available through
`embodichain run-example sim/assembly/asset_gallery`.

### Demo Gallery

The DexUni recording below shows peg insertion, multiple peg pairs, gears and
thread advancement, clockwise from the top left. It uses native RGB capture
with source contact parameters. These are passive physical examples; learned
robot policies are covered separately by the blind-assembly tasks.

<video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/dexuni.jpg" aria-label="DexUni examples: peg insertion, many pegs, gears and thread advancement">
  <source src="../../_static/assembly/dexuni.mp4" type="video/mp4">
  Your browser does not support embedded MP4 video.
</video>

[Download the DexUni recording](../../_static/assembly/dexuni.mp4).
Asset hashes, camera settings and measurements are available in the
[recording metadata](../../_static/assembly/recordings.json).

<details>
<summary>MuJoCo Warp reference recordings</summary>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/drop-peg.jpg" aria-label="MuJoCo Warp peg insertion reference">
      <source src="../../_static/assembly/drop-peg.mp4" type="video/mp4">
    </video>
    <figcaption>Peg insertion.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/many-pegs.jpg" aria-label="MuJoCo Warp many-pegs reference">
      <source src="../../_static/assembly/many-pegs.mp4" type="video/mp4">
    </video>
    <figcaption>Nine sampled assemblies.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/thread-pair.jpg" aria-label="MuJoCo Warp M10 thread reference">
      <source src="../../_static/assembly/thread-pair.mp4" type="video/mp4">
    </video>
    <figcaption>M10 thread advancement.</figcaption>
  </figure>
  <figure style="margin: 0;">
    <video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/gear-pair.jpg" aria-label="MuJoCo Warp gear transmission reference">
      <source src="../../_static/assembly/gear-pair.mp4" type="video/mp4">
    </video>
    <figcaption>Gear transmission with CG.</figcaption>
  </figure>
</div>
</details>

## Measurement

Run a passive measurement from Python:

```python
from embodichain_benchmarks.assembly.precision import PrecisionRunCfg, run_passive_probe

result = run_passive_probe(
    restored,
    PrecisionRunCfg(backend="newton", newton_solver="dexuni", duration=0.2),
)
```

The source benchmark runner also supports passive and constrained probes:

```bash
python scripts/benchmark/assembly/precision.py \
    --configs circle --backends newton --routes direct \
    --newton-solvers dexuni --duration .2

python scripts/benchmark/assembly/precision.py \
    --configs circle --clearances .0001 --backends newton --routes direct \
    --newton-solvers dexuni --probe S2 --directions 4 --duration .3
```

Reports retain requested settings, actual solver dispatch, measurements and
failed attempts. Constrained probes use four integration substeps per requested
interval and score every physical sample in the settled window.

## Notes and Limitations

- MJCF import supports world-attached fixed/free bodies and single unlimited
  hinges; native hinge preparation requires a body-centered anchor.
- `source_contact=True` preserves source contact parameters on Newton. Default
  backend comparisons require `source_contact=False` and use a different
  contact model.
- Use direct native configurations for the examples. Static compound USD
  imports that lose child colliders are rejected.
- The DexUni measurements use the `pure_mujoco` dispatch. They do not qualify
  VBD or rigid/soft coupling. Native overlap-only S0 queries and live contact
  overflow accounting are unavailable; precision qualification remains
  `unverified` even when behavioral probes pass.

## References

The optional [mjfit generator](https://github.com/younghyopark/mjfit) accompanies
[Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts](https://arxiv.org/abs/2610.23849).
See the [authors' project page](https://younghyopark.me/blind-proprioceptive-assembly/)
for the original research demonstrations. mjfit is MIT licensed; generated
manifests record the pinned provider revision.

When using the generator or its associated research, cite:

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
