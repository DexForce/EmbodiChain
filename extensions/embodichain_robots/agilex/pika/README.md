# AgileX Pika Gripper asset

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `PikaGripper`

- **Manufacturer / brand:** AgileX Robotics (松灵机器人); [official product or brand reference](https://github.com/agilexrobotics/pika_sdk).
- **Asset-declared model:** Pika Gripper.
- **Model matching:** The official AgileX SDK supports Pika Gripper. It establishes product affiliation, not the source of this packaged URDF.
- **Original asset production / publication:** Partially recorded. Packaged `asset_meta.json` names the ROS package `pika_gripper_description` from a local `pika_ros` checkout. No public upstream URL, source commit, original model author or manufacturer publication has been verified.
- **Current distribution:** EmbodiChain's [`eef_assets/PikaGripper.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/PikaGripper.zip); MD5 `8a9d0779e7dbfa7bf5d4713175df014e`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `PikaGripper`.
- **Known modifications:** The packaged `README.md` and `asset_meta.json` describe EmbodiChain preparation: relative mesh paths, attachment frames, generic link/joint names, 0–0.05 m jaw coordinates, FINGER2 mimic linkage, and DAE visual/STL collision mesh separation. Individual author identity is unverified. See also [registration commit a714a0c](https://github.com/DexForce/EmbodiChain/commit/a714a0c3aefd648aeb79217fe0987e18622a189d).
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current ZIP member list, `README.md`, `asset_meta.json` and URDF were inspected. No asset LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("PikaGripper/PikaGripper.urdf")
```

## Native display preparation

The unchanged URDF declares zero mass and inertia on the empty fixed frames
`GRIPPER_ROOT_LINK` and `ADAPTER_LINK`. Direct standalone native articulation
import rejects those inertials. Documentation renders use a temporary display
copy with only those inertial elements removed; the archived geometry, meshes,
materials and download identity remain unchanged. This preparation is not a
physical calibration or a standalone simulation preset. The existing
`dexforce_w1_030_pika` embodiment owns mounted simulation. See the
[Pika documentation](../../../../docs/source/resources/robot/agilex/pika.md)
for the preparation recipe and render manifest.
