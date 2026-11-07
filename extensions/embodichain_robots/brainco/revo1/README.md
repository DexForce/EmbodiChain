# BrainCo Revo 1 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `BrainCoHandRevo1`

- **Manufacturer / brand:** BrainCo (强脑科技); [official product or brand reference](https://www.brainco-hz.com/docs/revolimb-hand/revo1/parameters.html).
- **Asset-declared model:** Revo 1, left and right hand.
- **Model matching:** The official documentation confirms the Revo 1 family and six active joints. The packaged hand revision/SKU and original CAD/URDF release are unverified.
- **Original asset production / publication:** Unverified. The URDF includes the generic SolidWorks-to-URDF exporter credit and build information. This credits the exporter tooling, not the robot model author or publisher.
- **Current distribution:** EmbodiChain's [`eef_assets/BrainCoHandRevo01.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/BrainCoHandRevo01.zip); MD5 `ff9ac77e7e1493fd32d40c87fecbee6c`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `BrainCoHandRevo1`.
- **Known modifications:** No original asset modification record has been located. W1-specific mounting transforms belong to the W1 adapter; independent-hand drive values belong to its simulation preset.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF comments were inspected. No README, LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf")
```

The same archive contains `BrainCoRightHand/BrainCoRightHand.urdf`. Both hands retain their native asset identifiers.
