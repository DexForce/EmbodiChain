# dh_robotics / ag95 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `DH_AG95`

- **Manufacturer / brand:** DH-Robotics (大寰机器人); [official product or brand reference](https://www.dh-robotics.com/product/ag).
- **Asset-declared model:** AG95, DexForce modified version.
- **Model matching:** The AG product family is documented by the manufacturer; the exact CAD/URDF release has not been matched.
- **Original asset production / publication:** Unverified. No original model author, upstream repository/revision or manufacturer URDF publication has been established.
- **Current distribution:** EmbodiChain's [`eef_assets/DH_AG95.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/DH_AG95.zip); MD5 `34b6f3c2f649697ea7f12814b6a50529`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `DH_AG95`.
- **Known modifications:** DexForce modified the connector and finger, as stated in the [historical asset-class docstring](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) and its introduction in [commit 88d0721](https://github.com/DexForce/EmbodiChain/commit/88d07216694fcd0797a396196ea3cf16fa1746b5). No finer-grained modification log has been located.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF were inspected. No README, LICENSE, COPYING or NOTICE file was found; URDF contents did not establish the original model author. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("DH_AG95/dh_ag95.urdf")
```
