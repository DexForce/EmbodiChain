# YAM i2rt

`embodichain_robots.YamI2rtCfg` configures two six-joint arms and two
independently driven fingers per hand. The shared robot catalog ID is `YamI2rt`;
control groups are `right_arm`, `left_arm`, `right_eef`, and `left_eef`.

Like Franka, the default configuration resolves a registered external asset
through `get_data_path("YamI2rt/yam_i2rt.urdf")`. It downloads the prepared
`robot_assets/YamI2rtV1.zip` bundle when the cache is empty. Predownload with:

```bash
python -m embodichain data download --name YamI2rt
```

The registry uses `EMBODICHAIN_DOWNLOAD_PREFIX` and falls back to the official
`DexForceAI/embodichain_data` Hugging Face dataset. The prepared download needs
no CoACD installation. **During this Draft PR, publication is pending a
maintainer upload.** Until publication, prepare from source in a checkout:

```bash
python -m pip install coacd==1.0.14 trimesh==4.12.2 scipy==1.18.1
python scripts/tools/prepare_yam_i2rt_assets.py
```

The script downloads pinned robosimstudio sources, generates the URDF and
convex fingertip collision meshes, verifies every SHA-256, and preserves the
upstream MIT license and provenance. Its default destination is
`<data_root>/YamI2rt/`; registry downloads extract to
`<data_root>/extract/YamI2rt/`. Both locations are recognized. The default data
root is `~/.cache/embodichain_data/`. Set `EMBODICHAIN_DATA_ROOT` before imports
to change the cache root, or pass an explicit `fpath` to a prepared URDF.
Assets are excluded from the repository and wheel. `--source-dir` supports
preparation from an offline copy of the pinned source tree.

For maintainers, `--archive /path/to/YamI2rtV1.zip` packages the verified files
with fixed ZIP metadata, their MIT license, NOTICE and provenance. The registered
archive MD5 is `23601ea820ff3e6260336c4962ef01a5`. Repository attribution is
retained in `NOTICE` and `licenses/YamI2rt-MIT.txt`; meshes remain external.

The complete source-named dual-arm URDF is one `arm` component with an identity
transform; the existing component loader returns that source directly.
Simulation and `build_pk_serial_chain()` share the same URDF. The numerical
solver builds local six-joint FK/IK chains from each arm base to its TCP.
The preset requests Newton gravity compensation to match its source MJCF.

The handwritten routing deployment is documented in
`extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/README.md`.
It requires a DexSim build with cable-state support. Intermediate grasp tests
do not establish routing success across seeds or repeated resets.
