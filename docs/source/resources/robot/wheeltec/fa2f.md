# WHEELTEC FA2F asset

WHEELTEC brand affiliation is recorded, but an official product model corresponding to the asset identifier FA2F has not been verified. Here `fa2f` names the existing asset, not a confirmed commercial SKU.

<figure style="text-align: center; margin: 20px auto; max-width: 960px;">
  <img src="../../../_static/robots/eef/wheeltec_fa2f.png" alt="Rendered source visual geometry for WheelTecFA2F/WheelTecFA2F.urdf" width="1600" height="1080" style="display: block; width: 100%; height: auto;"/>
  <figcaption><b>WheelTecFA2F/WheelTecFA2F.urdf</b></figcaption>
</figure>

[Render manifest](../../../_static/robots/eef/wheeltec_fa2f.json).

The images use the archive's visual meshes and authored materials in native
OfflineRT. They show a zero-coordinate display pose with a presentation root transform,
without stepping physics. Rendering settings and the source URDF checksum are
recorded per image. See [EEF rendering conventions](../end_effectors.md#rendered-illustrations).

This page's illustrations enable tone mapping and use emission intensity 20
with quarter-intensity studio lights to retain white-part detail. Source
materials are preserved; the exact settings are recorded in each manifest.

## Resolve and preview

Canonical download-class import:

```python
from embodichain_robots.wheeltec.fa2f.assets import WheelTecFA2F
```

Use the stable asset reference for normal loading:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("WheelTecFA2F/WheelTecFA2F.urdf")
```

```bash
embodichain data download --name WheelTecFA2F
embodichain preview-asset --asset_path /absolute/path/to/WheelTecFA2F.urdf --viser
```

Pass the resolved absolute `urdf_path` to `preview-asset`. Asset listing and
class lookup do not download the archive; resolving its files may download on
first use. The remaining source models in this archive are listed in the image
captions above.

## Ownership and provenance

The download class lives in `extensions/embodichain_robots/wheeltec/fa2f/assets.py` and registers through the
existing `embodichain.assets` catalog with category `eef`. This asset-only
package does not introduce a standalone robot preset. Host assembly owns the
mounting transform, naming prefix, control-group mapping and task-specific TCP.

The [asset provenance record](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_robots/wheeltec/fa2f/README.md)
separately records manufacturer affiliation, model matching, asset production,
archive distribution, modifications and license status. Missing upstream or
license evidence stays explicitly unverified. The Python implementation's
Apache header is not an asset license statement.
