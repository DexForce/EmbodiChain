# Wuji Hand assets

`WujiHand` registers the official left/right hand descriptions as an `eef`
asset. Each hand has 20 active joints. This is an asset-only integration;
it does not register a RobotCfg, controller, tactile model or RL task.

- **Manufacturer and original publisher:** [wuji-technology](https://github.com/wuji-technology).
- **Pinned source:** [`5c4dfa16917355d073e2e1d7e3d03897b6cf9508`](https://github.com/wuji-technology/wuji-description/tree/5c4dfa16917355d073e2e1d7e3d03897b6cf9508).
- **Model:** Wuji Hand, left and right hands.
- **Asset license:** MIT; the original Wuji Technology LICENSE is retained.
- **Preparation:** The official `hand/body` URDF, MJCF and mesh bytes are unchanged. Their relative directory layout is retained beneath the asset root. This bundle selects the original Wuji Hand; Hand 2, glove, soft-pad and ROS variants are excluded.
- **Bundle:** `eef_assets/WujiHandV1.zip`; cache prefix `WujiHandV1`.
- **Archive MD5:** `898dec6518c33b96cc5a769c19fd9c14`.
- **Archive SHA256:** `3cd3bd4c31dd81fdf50725c1adde745c5dd0a71297d2c2ee8658a3a5da32c524`.
- **Evidence:** `manifest.json` records source paths, upstream Git blob IDs,
  original/prepared SHA256 hashes and joint order. `ATTRIBUTION.md` records
  the preparation changes. `UPSTREAM_README.md` preserves publisher guidance.

The downloader uses `EMBODICHAIN_DOWNLOAD_PREFIX` first and the official
`DexForceAI/embodichain_data` Hub as fallback. The versioned archive is prepared
locally; publication is required before its first public download.

```python
from embodichain.data import get_data_path

right_urdf = get_data_path("WujiHand/urdf/right.urdf")
left_urdf = get_data_path("WujiHand/urdf/left.urdf")
right_mjcf = get_data_path("WujiHand/mjcf/right.xml")
```

```bash
embodichain data download --name WujiHand
```

The URDF mesh paths resolve relative to the selected model. Native joint names,
limits, transforms and inertials are preserved. The packaged MJCF is an
upstream resource; EmbodiChain's standard articulation interface consumes URDF.
Descriptions and declared dynamics do not establish hardware calibration or
contact-rich manipulation success in either simulation backend.
