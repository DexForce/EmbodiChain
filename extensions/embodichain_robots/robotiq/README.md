# robotiq assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `Robotiq`

- **Manufacturer / brand:** Robotiq; [official product or brand reference](https://robotiq.com/products/adaptive-grippers).
- **Asset-declared model:** 2F-85 and 2F-140 shared archive.
- **Model matching:** The official product page documents both models. The packaged URDFs have not been matched to a manufacturer source revision.
- **Original asset production / publication:** Unverified. No original model author, upstream repository/revision or manufacturer URDF publication has been established.
- **Current distribution:** EmbodiChain's [`eef_assets/Robotiq_V2.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/Robotiq_V2.zip); MD5 `9bcbf92946b200c903938d59323e7b2c`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `Robotiq`.
- **Known modifications:** The registered archive changed to `Robotiq_V2.zip` in [commit 68a84f0](https://github.com/DexForce/EmbodiChain/commit/68a84f0f9c7e28e314bae9105330f395ac567d7e). A complete asset modification history has not been located.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF were inspected. No README, LICENSE, COPYING or NOTICE file was found; URDF contents did not establish the original model author. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf")
```

The second model resolves as `Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf`.
