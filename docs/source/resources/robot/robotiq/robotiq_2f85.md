# Robotiq 2F-85 standalone archive

This asset has a separate archive, checksum and cache from `Robotiq`. No equivalence between the two 85 mm URDF revisions is assumed. The manufacturer family is known; the packaged model publication history is unverified.

<figure style="text-align: center; margin: 20px auto; max-width: 960px;">
  <img src="../../../_static/robots/eef/robotiq_2f85.png" alt="Rendered source visual geometry for Robotiq2F85/Robotiq2F85.urdf" width="1600" height="1080" style="display: block; width: 100%; height: auto;"/>
  <figcaption><b>Robotiq2F85/Robotiq2F85.urdf</b></figcaption>
</figure>

[Render manifest](../../../_static/robots/eef/robotiq_2f85.json).

The images use the archive's visual meshes and authored materials in native
OfflineRT. They show a zero-coordinate display pose with a presentation root transform,
without stepping physics. Rendering settings and the source URDF checksum are
recorded per image. See [EEF rendering conventions](../end_effectors.md#rendered-illustrations).

## Resolve and preview

Canonical download-class import:

```python
from embodichain_robots.robotiq.robotiq_2f85.assets import Robotiq2F85
```

Use the stable asset reference for normal loading:

```python
from embodichain.data import get_data_path

urdf_path = get_data_path("Robotiq2F85/Robotiq2F85.urdf")
```

```bash
embodichain data download --name Robotiq2F85
embodichain preview-asset --asset_path /absolute/path/to/Robotiq2F85.urdf --viser
```

Pass the resolved absolute `urdf_path` to `preview-asset`. Asset listing and
class lookup do not download the archive; resolving its files may download on
first use. The remaining source models in this archive are listed in the image
captions above.

## Ownership and provenance

The download class lives in `extensions/embodichain_robots/robotiq/robotiq_2f85/assets.py` and registers through the
existing `embodichain.assets` catalog with category `eef`. This asset-only
package does not introduce a standalone robot preset. Host assembly owns the
mounting transform, naming prefix, control-group mapping and task-specific TCP.

The [asset provenance record](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_robots/robotiq/robotiq_2f85/README.md)
separately records manufacturer affiliation, model matching, asset production,
archive distribution, modifications and license status. Missing upstream or
license evidence stays explicitly unverified. The Python implementation's
Apache header is not an asset license statement.
