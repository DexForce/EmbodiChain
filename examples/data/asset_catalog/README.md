# Local rigid asset catalog

Asset providers prepare packages before a task environment is constructed.
They do not run online generation or infer missing task capabilities.

Arrange each asset as a directory containing `asset.catalog.json`, its mesh,
and all local mesh/texture dependencies. For a 4 cm block whose origin is at
its bottom center, a catalog can be:

```json
{
  "descriptor": {
    "asset_id": "block-04",
    "category": "block",
    "capabilities": ["antipodal_grasp", "support_surface"],
    "bounds_m": {"lower": [-0.02, -0.02, 0], "upper": [0.02, 0.02, 0.04]},
    "support_region": {"lower": [-0.02, -0.02, 0.04], "upper": [0.02, 0.02, 0.04]},
    "normalized": true,
    "provenance": {
      "capabilities": {"kind": "declared", "source": "asset author"},
      "support_region": {"kind": "declared", "source": "asset author"}
    }
  },
  "visual_path": "mesh.obj"
}
```

`normalized: true` explicitly declares rigid geometry in meters with local
right-handed Z-up coordinates. The provider verifies visual mesh bounds against
the declaration. This is not automatic unit conversion or proof of dynamic
stability. Support regions must be horizontal rectangles; container regions are
explicit usable interior boxes, never inferred from exterior bounds. Task-level
matching and simulation acceptance remain separate checks.

```python
from embodichain.data.asset_catalog import (
    AssetQuery, DirectoryAssetProvider, ExplicitAssetProvider, PreparationContext,
)

provider = DirectoryAssetProvider("/absolute/path/to/assets")
# Alternatively restrict selection to particular package directories or manifests:
# provider = ExplicitAssetProvider(["/absolute/path/to/assets/block-04"])

candidates = provider.candidates(AssetQuery(category="block"), limit=10)
prepared = provider.materialize(
    candidates[0],
    context=PreparationContext(cache_dir="/absolute/path/to/asset-cache"),
)
print(prepared.visual_path, prepared.content_sha256)
```

Queries rank exact matches first but retain other categories within the candidate
budget so the task matcher can apply explicit category aliases. Listing never
claims task compatibility. Preparation raises a diagnostic for missing metadata,
unknown manifest fields, missing files/textures, external references, symlinks,
invalid geometry, or corrupted cache content.

The optional `collision_path` defaults to `visual_path`. Distinct collision meshes
can be prepared, but the initial Task Program materializer requires a single mesh
because its `MeshCfg` exposes one model path. Supported formats are OBJ, GLB,
glTF, PLY, and STL. Include all referenced resources inside the package directory;
for OBJ textures, name them in the MTL file and quote filenames containing spaces.

Existing SimReady packages can add `"simready_metadata": "asset.json"` to the
catalog. The adapter imports the existing category when absent from the explicit
descriptor and retains source provenance. Bounds, normalized coordinates,
capabilities, and usable regions still require explicit annotations. A SimReady
status flag alone does not establish those properties.

The cache must be outside the source package. Every file in a package contributes
to its content identity and is copied into a digest-named snapshot, including
textures and metadata. Source edits produce a new snapshot; modified existing
cache snapshots are rejected. `PreparedAsset.provenance` records the package
root and each relative file digest for offline replay and downstream copying.

External providers use the same prepared package contract. Call
`verify_prepared_asset(prepared)` before consuming or copying a snapshot; it
verifies the recorded files, manifest metadata and content digest without checking
which provider produced them. The canonical content digest is SHA-256 over UTF-8
`json.dumps({"descriptor": descriptor.to_dict(), "files": file_digests},
sort_keys=True, separators=(",", ":"), allow_nan=False)`.
