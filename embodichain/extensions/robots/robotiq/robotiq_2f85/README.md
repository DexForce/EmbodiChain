# robotiq / robotiq_2f85 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `Robotiq2F85`

- **Manufacturer / brand:** Robotiq; [official product or brand reference](https://robotiq.com/products/adaptive-grippers).
- **Asset-declared model:** 2F-85, separate legacy archive.
- **Model matching:** The official product page documents 2F-85. The exact packaged URDF release is unverified; this archive remains separate from `Robotiq_V2.zip`.
- **Original asset production / publication:** Unverified. No original model author, upstream repository/revision or manufacturer URDF publication has been established.
- **Current distribution:** EmbodiChain's [`eef_assets/Robotiq2F85.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/Robotiq2F85.zip); MD5 `53ecbf2c953f43f1134aa7223e592292`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `Robotiq2F85`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF were inspected. No README, LICENSE, COPYING or NOTICE file was found; URDF contents did not establish the original model author. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("Robotiq2F85/Robotiq2F85.urdf")
```
