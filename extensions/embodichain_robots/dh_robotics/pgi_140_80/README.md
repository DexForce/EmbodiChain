# dh_robotics / pgi_140_80 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `DH_PGI_140_80`

- **Manufacturer / brand:** DH-Robotics (大寰机器人); [official product or brand reference](https://en.dh-robotics.com/product/pgia).
- **Asset-declared model:** PGI-140-80.
- **Model matching:** The official page documents the PGIA family. The legacy asset ID remains PGI-140-80; its exact product revision and CAD match are unverified.
- **Original asset production / publication:** Unverified. The URDF includes the generic SolidWorks-to-URDF exporter credit and build information. This credits the exporter tooling, not the robot model author or publisher.
- **Current distribution:** EmbodiChain's [`eef_assets/DH_PGI_140_80.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/DH_PGI_140_80.zip); MD5 `05a1a08b13c6250cc12affeeda3a08ba`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `DH_PGI_140_80`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF comments were inspected. No README, LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("DH_PGI_140_80/DH_PGI_140_80.urdf")
```
