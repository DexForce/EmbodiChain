# Wuji Hand assets

`WujiHand` registers the official left/right hand descriptions as an `eef`
asset. Each hand has 20 active joints. The companion `WujiHandCfg` provides a
fixed-base, joint-controlled standalone hand with left/right variants. Tactile models and RL tasks are not bundled.

- **Manufacturer and original publisher:** [wuji-technology](https://github.com/wuji-technology).
- **Pinned source:** [`5c4dfa16917355d073e2e1d7e3d03897b6cf9508`](https://github.com/wuji-technology/wuji-description/tree/5c4dfa16917355d073e2e1d7e3d03897b6cf9508).
- **Model:** Wuji Hand, left and right hands.
- **Asset license:** MIT; the original Wuji Technology LICENSE is retained.
- **Preparation:** The official `hand/body` URDF, MJCF and mesh bytes are unchanged. Their relative directory layout is retained beneath the asset root. This bundle selects the original Wuji Hand; Hand 2, glove, soft-pad and ROS variants are excluded.
- **Bundle:** `eef_assets/WujiHandV2.zip`; cache prefix `WujiHandV2`.
- **Archive MD5:** `d00115e853cfe0e9c4b096c26740354b`.
- **Archive SHA256:** `9c8178296dfe5ffc5922eb5ddbc349309818807581a370b0aeae946697d7be6a`.
- **Evidence:** `manifest.json` records source paths, upstream Git blob IDs,
  original/prepared SHA256 hashes and joint order. `ATTRIBUTION.md` records
  the preparation changes. `UPSTREAM_README.md` preserves publisher guidance.

The downloader uses `EMBODICHAIN_DOWNLOAD_PREFIX` first and the official
`DexForceAI/embodichain_data` Hub as fallback. The versioned archive is
[published on Hugging Face](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/0540f7a2eb383cf3f24e85069f5af2b369890541/eef_assets/WujiHandV2.zip).

```python
from embodichain.data import get_data_path

right_urdf = get_data_path("WujiHand/right.urdf")
left_urdf = get_data_path("WujiHand/left.urdf")
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

V2 adds root-level native URDF variants with root-relative mesh paths and
without the MuJoCo compiler block, which the native SDK does not consume.
Original URDFs under `urdf/` and all mesh/material/physical values are retained.
The preset uses these portable variants; the original URDF and MJCF paths
remain available for other consumers.
