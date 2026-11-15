# Captured Gaussian demo scenes

`../gaussian_scenes.py` downloads the fixed entries in `gaussian_scenes.json`,
checks SHA256 before loading, and saves 960×540 RGB, view-Z depth (`.npy`), and
same-frame capture metadata. Captures are cached under `EMBODICHAIN_DATA_ROOT`
using the shared dataset download lock. Binaries are not included in the wheel
or repository. Download URLs, source identities, checksums and cameras travel
with the example manifest.

From the repository root, using the environment containing the qualified DexSim build:

```bash
PYTHONPATH=. python examples/sim/gaussian_scenes.py --scene garden
PYTHONPATH=. python examples/sim/gaussian_scenes.py --scene library-usd
```

Use `--download-only` to populate the cache without starting rendering;
`--data-root` and `--output` select cache and output destinations. Linux native
Hybrid offscreen initialization requires the desktop session's working DISPLAY
and X authority, even though no interactive window opens.

`--group-translation 0.35 0 0.1` applies one arena-local offset to the imported
visual and static collision entities together. The demo constructs an explicit
`SceneView` from the facades returned by `add_usd()` and calls
`apply_group_transform`; it does not merge their physical or rendering types.
Capture JSON includes source-prim membership and before/after poses. The fixed
camera makes the scene's movement visible. Use `delta @ current_pose` for a
general group rotation/translation; runtime operations retain manager/Spawn
ownership and do not create another scene controller.

- **Garden**: the classic outdoor Garden reconstruction, mirrored by
  [kishimisu/3d-gaussian-splatting-webgl](https://huggingface.co/kishimisu/3d-gaussian-splatting-webgl/blob/d9bd1af40261b318a95d0f5882228f332a4c1afd/garden.ply).
  4,325,323 Gaussians, degree-3 SH, 1,072,681,636 bytes. The source is an Inria
  binary PLY. The table/potted-plant view uses the provider's target and a
  corrected image-up vector. The [original 3DGS project](https://github.com/graphdeco-inria/gaussian-splatting)
  and capture-provider terms apply; the Python source license does not license
  this capture. Only download metadata is distributed here.
- **Library USD**: DexSim's `gaussian-library` acceptance package, 2,000,000
  degree-0 Gaussians. Trace: `tools/ci/asset_loading_manifest.json` →
  `run_asset_loading.py:fetch_case` → base URL in
  `python/dexsim/utility/resource_url.txt` → the manifest's fixed archive.
  The archive and every dependency are checksum checked. The corporate NAS
  URL requires the internal network; redistribution permission is not inferred.
  `asset.usda` references `source/original.ply` and contains a coarse enclosing
  collider. `SimulationManager.add_usd()` loads the actual package unchanged;
  the viewing camera compensates for the capture's orientation. The box is
  acceptance-corpus geometry, not a navigable collision reconstruction.

Both presets use one Arena and Hybrid rendering with OptiX mesh denoising and
native CUDA Gaussian raster. The shared default ground is hidden per camera.
Gaussian depth is the first accepted contribution's view-Z, with invalid
background represented as infinity in the NPY; JSON reports use null for
non-finite values. These are rendering demos, not physical relighting or
navigation qualification.
