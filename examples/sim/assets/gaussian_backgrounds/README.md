# Standardized captured Gaussian backgrounds

The three V1 packages are published under
[DexForceAI/embodichain_data/scene_assets/gaussian_backgrounds_v1](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/3ec5b1fb7d320599c54f07bc20d4c49d570a2e2b/scene_assets/gaussian_backgrounds_v1).
`sources.lock.json` pins the original providers and bytes;
`release.lock.json` pins the published archive and AssetRevision identities.

| Asset reference | Original data | Data license declaration |
| --- | --- | --- |
| `RoboSplatterOfficeGS/asset.usda` | RoboSplatter office, 2,209,646 splats | Apache-2.0 dataset card |
| `DiscoverseLab3GS/asset.usda` | DISCOVERSE lab3, 351,605 splats | Not separately declared |
| `RoboSimGSTabletopGS/asset.usda` | RoboSimGS single-view scene, 1,052,066 splats | MIT dataset card |

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import UsdSourceCfg, VisualObjectCfg

background = VisualObjectCfg(
    uid="background",
    source=UsdSourceCfg(
        file_path=get_data_path("RoboSplatterOfficeGS/asset.usda"),
        root_prim_path="/Asset",
        source_prim_path="/Asset/Visual",
    ),
    source_pose_mode="preserve",
)
```

The shared dataset registry downloads and verifies an immutable Hub snapshot,
uses the project download lock, and extracts into a version-specific cache.
An official Hub URL is retained after a configured mirror. All ZIPs include
directory entries before files for the Open3D extraction path.

## Package contract

```text
asset.usda                    # default prim /Asset; metres, Z-up
asset.json                    # AssetRevision 2.0-draft inventory/provenance
source/original.ply           # byte-identical upstream capture
source/provenance.json        # provider, revision, hash, conventions, licence
visual/scene.ply              # lab3 only: separately decoded standard PLY
NOTICE.md / LICENSE.txt
reports/normalization.json
reports/dexsim-packaging.json
reports/dexsim-registry.json
```

The entry references PLY through `/Asset/Visual`. DexSim recognizes this native
external format; plain OpenUSD without a PLY plugin can emit an unresolved-file
format warning. The package is admitted by the qualified DexSim parser.

Office's source is Y-up: its +90° X rotation is authored once in USD, with no
rewriting of point positions, covariance or spherical-harmonic coefficients.
Do not repeat that rotation in a scene config. The other captures already use
Z-up. Original metric origins are retained, including RoboSimGS's tabletop
near Z=0. Lab3's compressed source is preserved and decoded into a separate
Inria-compatible file. Its decoder retains its upstream MIT notice.

These are **visual-only backgrounds**: no enclosing collider, reconstructed
physics, robot, task object, light or camera is added. Task hosts supply their
own qualified collision geometry. AssetRevision is package metadata with an
incomplete quality report and no certification; neither packaging nor native
RGB admission establishes industrial READY or metric Gaussian depth.

Use `RenderCfg.for_gaussian(gaussian={"apply_display_srgb": False})` for these
display-referred colors. The flag is fixed before World creation and shared
by all Gaussian assets in that World.

The data licenses remain upstream. In particular, DISCOVERSE's model repository
does not provide an asset license; this package retains `license: null` and does
not infer the software repository's MIT license for the model. Package-level
notices take precedence over any dataset-wide default for these third-party bytes.

## Build and validate a new revision

From a qualified Python 3.12 / DexSim environment:

```bash
python scripts/tools/prepare_gaussian_scene_assets.py --output /tmp/gs-build
# Or reuse verified original PLYs, named as recorded in sources.lock.json:
python scripts/tools/prepare_gaussian_scene_assets.py \
  --source-dir /path/to/original-plys --output /tmp/another-gs-build
```

The tool never replaces existing outputs or changes input files. It creates a
visual-only USD using OpenUSD and invokes the SDK's existing scheme-one exporter
for complete inventory/provenance metadata. This offline build tool depends on
`dexsim.kit.simready._usd_package.export_processed_usd`; runtime downloads do not.
Archive identities and timestamps are new on each build; the source bytes and
normalization operations are pinned and independently hashed. The generic SDK
converter is deliberately not used because it also generates an enclosing box.

The publication includes actual native render previews in `validation/`.
The complete physical consumer is
[the Franka dual-camera Task Program example](../../../gym/gaussian_pick_place/README.md).
