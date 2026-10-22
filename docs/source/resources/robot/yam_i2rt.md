# YAM i2rt

`embodichain_robots.YamI2rtCfg` configures two six-joint arms and two
independently driven fingers per hand. The shared robot catalog ID is `YamI2rt`;
control groups are `right_arm`, `left_arm`, `right_eef`, and `left_eef`.

From a repository checkout, prepare the external assets before constructing
this configuration:

```bash
python -m pip install coacd==1.0.14 trimesh==4.12.2 scipy==1.18.1
python scripts/tools/prepare_yam_i2rt_assets.py
```

The script downloads pinned robosimstudio sources, generates the URDF and
convex fingertip collision meshes, verifies every SHA-256, and preserves the
upstream MIT license and provenance. Its default destination is
`~/.cache/embodichain_data/YamI2rt/`. Set `EMBODICHAIN_DATA_ROOT` before imports
to change the cache root, or pass an explicit `fpath` to a prepared URDF.
Assets are excluded from the repository and wheel. `--source-dir` supports
preparation from an offline copy of the pinned source tree.

Simulation and `build_pk_serial_chain()` share the same URDF. The numerical
solver builds local six-joint FK/IK chains from each arm base to its TCP.
The preset requests Newton gravity compensation to match its source MJCF.

The handwritten routing deployment is documented in
`extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/README.md`.
It requires a DexSim build with cable-state support. Intermediate grasp tests
do not establish routing success across seeds or repeated resets.
