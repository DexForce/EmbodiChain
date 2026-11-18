(precision-assembly)=

# Precision Assembly

The precision assembly toolkit prepares peg/hole pairs, threaded parts, and
gears for contact-rich simulation with DexUni. It preserves each part's
collision geometry, visual surfaces, local frames, mass, center of mass, and
inertia so that small clearances survive asset preparation.

DexUni is available through EmbodiChain's Newton physics backend. The examples
below select DexUni explicitly and use the same portable assembly assets for
peg insertion, thread engagement, and gear transmission.

## Capabilities

- **Assembly generation** — creates named or configuration-defined peg/hole
  pairs, threaded parts, and gears using the optional mjfit generator.
- **Clearance preservation** — retains individual convex collision pieces
  around mating surfaces and holes.
- **Portable assets** — saves self-contained JSON manifests and exports USD
  geometry with body ownership and physical properties.
- **Native preparation** — converts assets into rigid-body and hinge
  configurations for simulation.
- **DexUni demonstrations** — runs single-peg, many-peg, thread, and gear
  examples with native rendering and optional video recording.

## Installation

Install EmbodiChain and the pinned mjfit provider to generate assemblies:

```bash
pip install .
pip install 'mjfit @ git+https://github.com/younghyopark/mjfit.git@5833f465896e1e06bc4be4f003be847f5183a4db'
```

Loading an existing manifest does not require mjfit or MuJoCo. Native simulation
requires a runtime with DexUni support; native video recording also needs a
supported display and FFmpeg.

## Python API

Use `embodichain.toolkits.assembly` to generate and save reusable assets:

```python
from pathlib import Path

from embodichain.toolkits.assembly import AssemblyAsset, build_assembly

output = Path("outputs/assembly")
asset = build_assembly("circle", clearance=0.0001)
manifest = asset.save(output / "circle.json")
restored = AssemblyAsset.load(manifest)
restored.write_usd(output / "circle.usda")
```

Named configurations include `circle`, `star`, `usb_c`, `m10`, and `spur_pair`.
Use `sample_assemblies()` for seeded peg sampling or `build_many_pegs()` for a
scene containing multiple plate/peg pairs. Native configuration preparation is
available through `embodichain.toolkits.assembly.native.prepare_native_configs()`.
See the [Toolkits API reference](../../api_reference/embodichain/embodichain.toolkits.rst)
for the public interfaces.

## Command-Line Examples

Select DexUni with `--backend newton --newton-solver dexuni`:

```bash
# Insert a peg into a circular hole with 100 μm clearance.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config circle --clearance .0001 --duration .2 \
    --native-window --native-record --renderer fast-rt \
    --output outputs/assembly/dexuni_circle

# Seat nine independently sampled peg/hole pairs in one scene.
embodichain run-example sim/assembly/visual_demo \
    --backend newton --newton-solver dexuni \
    --config many_pegs --count 9 --duration .25 \
    --native-window --native-record --renderer fast-rt \
    --output outputs/assembly/dexuni_many

# Simulate thread engagement and gear transmission.
embodichain run-example sim/assembly/thread_pair \
    --backend newton --newton-solver dexuni --duration .5
embodichain run-example sim/assembly/gear_pair \
    --backend newton --newton-solver dexuni --solver cg --duration .5
```

`--newton-solver dexuni` selects the solver family. The gear example also uses
`--solver cg` to select the contact solver algorithm. Add `--collision-colors`
to a visual demo to inspect individual convex pieces.

## Demo

The recording shows DexUni running peg insertion, multiple peg/hole pairs,
thread engagement, and gear transmission. Clockwise from the top left:
peg, many pegs, gears, and thread.

<video controls playsinline preload="none" style="width: 100%;" poster="../../_static/assembly/dexuni.jpg" aria-label="DexUni precision assembly: peg insertion, many pegs, threads, and gears">
  <source src="../../_static/assembly/dexuni.mp4" type="video/mp4">
  Your browser does not support embedded MP4 video.
</video>

[Download the DexUni MP4](../../_static/assembly/dexuni.mp4).

These passive demonstrations illustrate geometry and contact behavior. They
use DexUni's rigid-body `pure_mujoco` dispatch and do not demonstrate
rigid/soft coupling or a learned assembly policy.

## References

The optional [mjfit generator](https://github.com/younghyopark/mjfit) accompanies
[Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts](https://arxiv.org/abs/2610.23849)
by Antonia Bronars, Younghyo Park, Kushal Kedia, and Pulkit Agrawal.
See the [authors' project page](https://younghyopark.me/blind-proprioceptive-assembly/)
for the original research and policy demonstrations.
