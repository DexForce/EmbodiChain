# DH-Robotics PGC-140-50 assets

Manufacturer references establish product affiliation separately from asset provenance. Unknown entries below are intentional; ownership migration does not assert that the packaged URDFs or meshes are manufacturer originals.

## `DH_PGC_140_50`

- **Manufacturer / brand:** DH-Robotics (大寰机器人); [official product or brand reference](https://www.dh-robotics.com/product/pgc).
- **Asset-declared model:** PGC-140-50.
- **Model matching:** The product family is documented by the manufacturer; the exact CAD/URDF release has not been matched.
- **Original asset production / publication:** Unverified. The URDF includes the generic SolidWorks-to-URDF exporter credit and build information. This credits the exporter tooling, not the robot model author or publisher.
- **Current distribution:** EmbodiChain's [`eef_assets/DH_PGC_140_50.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/DH_PGC_140_50.zip); MD5 `c2a642308a76e99b1b8b7cb3a11c5df3`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `DH_PGC_140_50`.
- **Known modifications:** No asset modification record has been located. This does not establish an unmodified manufacturer release.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF comments were inspected. No README, LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("DH_PGC_140_50/DH_PGC_140_50.urdf")
```

## `DH_PGC_140_50_M`

- **Manufacturer / brand:** DH-Robotics (大寰机器人); [official product or brand reference](https://www.dh-robotics.com/product/pgc).
- **Asset-declared model:** PGC-140-50, DexForce modified version.
- **Model matching:** The base product family is documented by the manufacturer. `_M` identifies the packaged modification, not a manufacturer SKU.
- **Original asset production / publication:** Unverified. The URDF includes the generic SolidWorks-to-URDF exporter credit and build information. This credits the exporter tooling, not the robot model author or publisher.
- **Current distribution:** EmbodiChain's [`eef_assets/DH_PGC_140_50_M.zip`](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/eef_assets/DH_PGC_140_50_M.zip); MD5 `3a9ab5f32639e03afb38dc033b44bb62`. The configured `EMBODICHAIN_DOWNLOAD_PREFIX` selects the mirror; the asset ID and cache prefix remain `DH_PGC_140_50_M`.
- **Known modifications:** DexForce modified the connector and finger, as stated in the [historical asset-class docstring](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py). No finer-grained modification log or source revision has been located.
- **Asset license:** Unverified. No asset-specific license was found in this archive. The Apache 2.0 header in `assets.py` covers the Python download implementation and is not evidence of a URDF or mesh license.
- **Inspection evidence:** The current archive member list and extracted URDF comments were inspected. No README, LICENSE, COPYING or NOTICE file was found. The [historical registry](https://github.com/DexForce/EmbodiChain/blob/a714a0c3aefd648aeb79217fe0987e18622a189d/embodichain/data/assets/eef_assets.py) records the unchanged download identity.
- **Pending verification:** Original CAD/URDF author and upstream revision, complete modification history, asset license, and exact product/CAD release match.
- **Reviewed:** 2026-11-01.

Resolve the model through the stable asset reference:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
```
