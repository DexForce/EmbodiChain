# SimReady Asset Pipeline

The SimReady asset pipeline converts raw mesh archives into normalized simulation assets. It ingests a source mesh, preserves or bakes visual materials, cleans mesh topology, estimates real-world scale and semantics with multimodal LLMs, and exports assets that can be loaded directly in EmbodiChain simulations.

## Quick Start

Run the pipeline on a single asset directory:

```bash
python -m embodichain.gen_sim.simready_pipeline.cli.start \
    --input_dir /path/to/raw_mesh_folder \
    --output_root /path/to/output_folder \
    --category YourCategory
```

Preview the generated SimReady mesh:

```bash
python -m embodichain preview-asset \
    --asset_path /path/to/sim_ready_asset_or_usd_asset \
    --asset_type rigid
```

## Prerequisites

The full pipeline uses Blender, trimesh, pyrender, and an OpenAI-compatible multimodal chat completions endpoint. Install EmbodiChain with the `gensim` extra and enable both the EmbodiChain package index and Blender package index.

Install from PyPI with `uv`:

```bash
uv pip install "embodichain[gensim]" \
    --extra-index-url http://pyp.open3dv.site:2345/simple/ \
    --trusted-host pyp.open3dv.site \
    --extra-index-url https://download.blender.org/pypi/
```

Install from source with `uv`:

```bash
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
uv pip install -e ".[gensim]" \
    --extra-index-url http://pyp.open3dv.site:2345/simple/ \
    --trusted-host pyp.open3dv.site \
    --extra-index-url https://download.blender.org/pypi/
```

Install from PyPI with `pip`:

```bash
pip install "embodichain[gensim]" \
    --extra-index-url http://pyp.open3dv.site:2345/simple/ \
    --trusted-host pyp.open3dv.site \
    --extra-index-url https://download.blender.org/pypi/
```

Install from source with `pip`:

```bash
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
pip install -e ".[gensim]" \
    --extra-index-url http://pyp.open3dv.site:2345/simple/ \
    --trusted-host pyp.open3dv.site \
    --extra-index-url https://download.blender.org/pypi/
```

Set the OpenAI-compatible LLM credentials before running the pipeline, or configure them in `embodichain/gen_sim/simready_pipeline/configs/gen_config.json`. Environment variables override the JSON config.

OpenAI API example:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Gemini API example:

```bash
export OPENAI_API_KEY="your-gemini-api-key"
export OPENAI_MODEL="gemini-3.5-flash"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

## Processing Flow

The command above runs the full parser sequence:

- **Ingest**: finds the first parseable mesh (`.glb`, `.gltf`, `.obj`, `.ply`, `.stl`), archives the raw input, and writes a canonical `asset_source/asset.obj`.
- **Visual processing**: by default, Blender remeshes the source mesh, unwraps UVs, and bakes diffuse and normal textures. With `--simple`, ingest uses trimesh only and skips Blender remesh/bake.
- **Inspection**: detects whether the normalized source is a mesh, articulation, or scene.
- **Geometry processing**: cleans topology and applies Blender decimation to the canonical mesh.
- **SimReady finalization**: renders multi-view images, uses the LLM to infer object orientation, physical dimensions, and semantics, then exports `asset_simready/asset_simready.obj`.
- **Physics and USD export**: infers physics properties and writes a USD package when possible.
- **Internal preview assets**: generates thumbnails and internal metadata for asset browsing.

## Output Layout

Each processed asset is written under a generated asset ID:

```text
simready_car/
`-- <asset_id>/
    |-- asset_archive/          # Raw source directory copy
    |-- asset_source/           # Canonical normalized source mesh and textures
    |   |-- asset.obj
    |   |-- asset.mtl
    |   |-- diffuse.png
    |   `-- normal.png
    |-- asset_simready/         # Final oriented and scaled mesh
    |   `-- asset_simready.obj
    |-- asset_usd/              # USD export
    `-- asset.json              # Metadata, geometry, semantics, physics, and paths
```

Use `asset_simready/asset_simready.obj` or `asset_usd/` for simulation preview and downstream scene construction.

## Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input_dir` | Directory containing the raw asset files. | **required** |
| `--output_root` | Directory where processed assets are written. | **required** |
| `--category` | Category hint passed into the pipeline, such as `car`, `bowl`, or `chair`. | **required** |
| `--simple` | Use trimesh-only ingest and skip Blender remesh/bake during ingest. Geometry cleanup later in the pipeline still uses Blender. | `False` |

## Configuration

Pipeline hyperparameters live in `embodichain/gen_sim/simready_pipeline/configs/gen_config.json`.

### Ingest

```json
"ingest": {
  "canonical_asset_name": "asset.obj",
  "unprocessed_formats": [".urdf", ".usd"],
  "parseable_mesh_formats": [".glb", ".gltf", ".obj", ".ply", ".stl"]
}
```

This section controls source file discovery and the canonical output mesh name.

### Mesh Processing

```json
"mesh_processing": {
  "trimesh_ingest": {
    "scene_mesh_strategy": "first",
    "mtl_name": "asset.mtl",
    "visual": {
      "default_face_color": [128, 128, 128, 255],
      "pbr_base_color_only": true
    },
    "export": {
      "include_normals": true,
      "include_color": true,
      "include_texture": true,
      "write_texture": false
    }
  },
  "blender_remesh_bake": {
    "remesh": {
      "voxel_size": 0.01,
      "min_voxel_size_ratio": 0.005,
      "use_smooth_shade": true
    },
    "decimate": {
      "ratio": 0.9
    },
    "uv": {
      "angle_limit": 66.0,
      "island_margin": 0.02
    },
    "bake": {
      "texture_size": 2048,
      "diffuse_texture_name": "diffuse.png",
      "normal_texture_name": "normal.png",
      "cage_extrusion_ratio": 0.05
    }
  },
  "blender_cleanup_decimate": {
    "enabled": true,
    "cleanup": {
      "merge_dist": 0.00001,
      "remove_non_manifold": true,
      "triangulate": false
    },
    "simplify": {
      "ratio": 0.5,
      "weld_distance": 0.0001,
      "collapse_triangulate": true
    }
  },
  "simready_finalize": {
    "render_resolution": 1024
  }
}
```

`trimesh_ingest` controls the lightweight ingest path. It does not perform mesh decimation; it normalizes visual materials and exports OBJ/MTL files.

`blender_remesh_bake` controls the default ingest path when `--simple` is not provided. It remeshes the raw mesh, decimates it, unwraps UVs, and bakes textures.

`blender_cleanup_decimate` controls the later geometry parser stage. It uses Blender mesh operators and the Blender Decimate modifier to clean and simplify the canonical mesh.

`simready_finalize` controls rendering used by the LLM-driven orientation and scale estimation stage.

### LLM

```json
"llm": {
  "openai_compatible": {
    "api_key": "",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "default_query": {}
  }
}
```

This section configures the multimodal LLM used for object classification, orientation selection, dimension inference, semantic annotation, and physics inference. Any provider that supports the OpenAI-compatible chat completions API can be used by changing `api_key`, `model`, `base_url`, and optional `default_query` parameters.

For Gemini, use the same config shape:

```json
"llm": {
  "openai_compatible": {
    "api_key": "your-gemini-api-key",
    "model": "gemini-3.5-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "default_query": {}
  }
}
```

For Azure-style OpenAI-compatible endpoints that require an API version query parameter, use `default_query`:

```json
"llm": {
  "openai_compatible": {
    "api_key": "your-api-key",
    "model": "gpt-4o",
    "base_url": "https://dex-gpt4.openai.azure.com/openai/deployments/gpt-4o",
    "default_query": {
      "api-version": "2025-01-01-preview"
    }
  }
}
```

## Default vs Simple Ingest

The default command uses Blender during ingest:

```bash
python -m embodichain.gen_sim.simready_pipeline.cli.start \
    --input_dir /path/to/raw_mesh_folder \
    --output_root /path/to/output_folder \
    --category YourCategory
```

Use `--simple` when you want faster trimesh-only ingest:

```bash
python -m embodichain.gen_sim.simready_pipeline.cli.start \
    --input_dir /path/to/raw_mesh_folder \
    --output_root /path/to/output_folder \
    --category YourCategory \
    --simple
```

The simple mode only affects the ingest step. The downstream geometry parser still uses Blender cleanup and decimation unless `mesh_processing.blender_cleanup_decimate.enabled` is set to `false`.

## See Also

- [Asset Preview](../interaction/preview_asset.md): Load generated meshes and USD assets in the simulator.
- [Installation](../../quick_start/install.md): Install EmbodiChain with Blender and rendering dependencies.
- [Toolkits](../toolkits/index.rst): Other asset preparation utilities.
