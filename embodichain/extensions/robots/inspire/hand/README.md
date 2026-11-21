# inspire / hand assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `InspireHand`

- **Manufacturer / brand:** INSPIRE ROBOTS (因时机器人); [official product or brand reference](https://en.inspire-robots.com/product/rh56bfx/).
- **Asset-declared model:** InspireHand asset family; RH56BFX referenced by the original code.
- **Model matching:** The original code references RH56BFX and the manufacturer documents that model. The exact packaged URDF-to-product match is unverified, so the namespace remains `inspire.hand`.
- **Original asset production / publication:** Unverified. The URDF includes the generic SolidWorks-to-URDF exporter credit and build information. This credits the exporter tooling, not the robot model author or publisher.
- **Current distribution:** EmbodiChain's [`eef_assets/InspireHand.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/InspireHand.zip); MD5 `c60132a6f03866fb021cca5b6d72845e`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `InspireHand`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF comments were inspected. No README, LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("InspireHand/InspireLeftHand/InspireLeftHand.urdf")
```

The same archive also contains `InspireRightHand/InspireRightHand.urdf` and the existing joint-data files.
