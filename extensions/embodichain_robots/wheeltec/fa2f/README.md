# wheeltec / fa2f assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `WheelTecFA2F`

- **Manufacturer / brand:** WHEELTEC (轮趣科技); [official product or brand reference](https://www.wheeltec.net/).
- **Asset-declared model:** FA2F asset identifier; formal product model unverified.
- **Model matching:** The official site confirms the brand. No official FA2F model reference has been located. The URDF robot name is `gripper_L180W101`; this is not treated as verified manufacturer nomenclature.
- **Original asset production / publication:** Unverified. No original model author, upstream repository/revision or manufacturer URDF publication has been established.
- **Current distribution:** EmbodiChain's [`eef_assets/WheelTecFA2F.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/WheelTecFA2F.zip); MD5 `feaf13f25b1c6ce58d011b1f2fa72f58`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `WheelTecFA2F`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF were inspected. No README, LICENSE, COPYING or NOTICE file was found; URDF contents did not establish the original model author. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match (including the formal FA2F model name).
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("WheelTecFA2F/WheelTecFA2F.urdf")
```
