# End-effector assets and migration

Concrete hand and gripper assets live beside other robot models in
`embodichain_robots`, under their manufacturer/model namespace. `Robot` describes
an independently controlled runtime entity; EEF describes a role in an assembly.
The same BrainCo hand model therefore serves its standalone preset and W1 mounts.
The core continues to own downloads, caches, generic configuration and runtime
control. Host presets or embodiments own mounting transforms, prefixes, control
group mappings and task-specific TCPs.

## Canonical asset imports

The previous `embodichain.data.assets.eef_assets` module and its aggregate
`embodichain.data.assets` class exports have been removed. Import download
classes from these canonical modules, or use `get_data_path()` with the existing
asset ID. No compatibility forwarding module is provided.

| Asset ID / class | Canonical import module | Resource page |
| --- | --- | --- |
| `DH_PGC_140_50` | `embodichain_robots.dh_robotics.pgc_140_50.assets` | [PGC-140-50](dh_robotics/pgc_140_50.md) |
| `DH_PGC_140_50_M` | `embodichain_robots.dh_robotics.pgc_140_50.assets` | [PGC modification](dh_robotics/pgc_140_50.md) |
| `DH_PGI_140_80` | `embodichain_robots.dh_robotics.pgi_140_80.assets` | [PGI-140-80](dh_robotics/pgi_140_80.md) |
| `DH_AG95` | `embodichain_robots.dh_robotics.ag95.assets` | [AG95 modification](dh_robotics/ag95.md) |
| `ZH_CTM2F110` | `embodichain_robots.changingtek.ctm2f110.assets` | [CTM2F110](changingtek/ctm2f110.md) |
| `BrainCoHandRevo1` | `embodichain_robots.brainco.revo1.assets` | [Revo 1](brainco/revo1.md) |
| `InspireHand` | `embodichain_robots.inspire.hand.assets` | [INSPIRE hand](inspire/hand.md) |
| `Robotiq` | `embodichain_robots.robotiq.assets` | [Shared Robotiq archive](robotiq/assets.md) |
| `Robotiq2F85` | `embodichain_robots.robotiq.robotiq_2f85.assets` | [Separate 2F-85 archive](robotiq/robotiq_2f85.md) |
| `WheelTecFA2F` | `embodichain_robots.wheeltec.fa2f.assets` | [FA2F asset](wheeltec/fa2f.md) |
| `PikaGripper` | `embodichain_robots.agilex.pika.assets` | [AgileX Pika](agilex/pika.md) |
| `WujiHand` | `embodichain_robots.wuji.hand.assets` | [Wuji Hand](wuji/hand.md) |
| `SharpaWave` | `embodichain_robots.sharpa.wave.assets` | [Sharpa Wave 01](sharpa/wave.md) |

The existing asset IDs, class names, archive URLs, checksum validation and cache
prefixes are preserved. All entries remain in the `eef` data category and use
the existing `embodichain.assets` entry point. Pika retains its configured mirror
and official Hub fallback order. Catalog discovery does not construct datasets.

```python
from embodichain.data import get_data_path
from embodichain_robots.agilex.pika.assets import PikaGripper

# Existing references still work; no cache migration is required.
urdf_path = get_data_path("PikaGripper/PikaGripper.urdf")
```

```bash
embodichain data list --category eef
embodichain data download --name PikaGripper
```

## Provenance boundaries

Each owning source directory includes an archive-specific `README.md` recording
manufacturer affiliation, claimed model, model-match status, asset author and
publication source, current distribution archive/digest, known modifications,
license evidence and remaining questions. These records ship with the wheel.

Manufacturer product references establish affiliation; they do not prove that
EmbodiChain's URDF or meshes are manufacturer publications. Asset-specific
licenses remain unverified where the archives contain none. Python Apache
headers are not evidence of asset licensing. In particular:

- `DH_PGC_140_50_M` **and** `DH_AG95` record DexForce connector/finger changes.
- Pika belongs under AgileX; its preparation metadata names a local ROS source
  package but does not pin a verified public upstream commit.
- `InspireHand` cites RH56BFX, while its exact packaged CAD match is unverified.
- WHEELTEC affiliation is known, while FA2F remains an asset identifier with an
  unverified formal commercial model.
- Missing modification records do not establish an original or unmodified asset.

## Standalone hands and mounted tools

`WujiHandCfg` adds a standalone left/right hand with 20 independent
source joints; see [Wuji Hand](wuji/hand.md) for control and physical defaults.

`SharpaWaveCfg` adds a standalone left/right hand with 22 independent
source joints; see [Sharpa Wave 01](sharpa/wave.md) for control and physical defaults.

`BrainCoRevo1Cfg` is a standalone joint-controlled hand preset. It uses
six active joints and retains the four source mimic followers. Its native
model specification is also consumed by W1, whose installation calibration and
existing configuration interface remain W1-owned. See [Revo 1](brainco/revo1.md)
for variants, commands and physical defaults, and [W1](dexforce/w1.md) for the
host assembly.

The remaining asset-only packages expose download assets rather than `RobotCfg`
classes. Resolve their URDFs and configure their host assembly explicitly. A
control group named `eef` does not turn a dexterous hand into a parallel gripper
or imply an open/close service, finger IK, retargeting or grasping strategy.

## Rendered illustrations

The linked manufacturer/model pages show native OfflineRT renders for the listed
asset IDs, including standalone and asset-only hands and both models in the shared
Robotiq archive. Each image has a neighboring JSON manifest with source URDF
SHA256, source reference, image SHA256, camera, root transform, joint positions,
lighting and renderer settings.

The default presentation uses 1600 × 1080 output, 1 sample per pixel, DLSS offscreen
reconstruction at quality 2, 48 warmup frames, tone mapping off, environment
emission intensity 80 and three neutral rectangular studio lights. Meshes and
materials are taken from the current archives. AG95, both models in the shared
Robotiq archive and WHEELTEC FA2F use tone mapping, emission intensity 20 and
quarter-intensity studio lights to retain detail in their white parts. Their
authored black materials are preserved. Presentation root transforms
place the geometry above the ground; no physics steps are taken. Pika uses
the documented display-only removal of zero inertials on its two empty fixed
frames so the native importer can load its unchanged visual geometry. These are
asset illustrations, not evidence of physical calibration or simulated grasp
success.
