# changingtek / ctm2f110 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `ZH_CTM2F110`

- **Manufacturer / brand:** ChangingTek Robotics (知行机器人); [official product or brand reference](https://en.changingtek.com/diandong/159).
- **Asset-declared model:** CTM2F110.
- **Model matching:** The official product page names CTM2F110. The packaged URDF has not been matched to a manufacturer-published CAD/URDF release.
- **Original asset production / publication:** Unverified. No original model author, upstream repository/revision or manufacturer URDF publication has been established.
- **Current distribution:** EmbodiChain's [`eef_assets/ZH_CTM2F110.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/ZH_CTM2F110.zip); MD5 `0e7c3310425609797fe010b2a76fe465`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `ZH_CTM2F110`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF were inspected. No README, LICENSE, COPYING or NOTICE file was found; URDF contents did not establish the original model author. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("ZH_CTM2F110/ZH_CTM2F110.urdf")
```
