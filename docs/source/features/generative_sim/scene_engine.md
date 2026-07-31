# Scene Engine

The Scene Engine converts one tabletop-scene image into a scene-only export. It
identifies a table and visible assets, generates their meshes, refines their
layout, settles them under gravity, and writes an EmbodiChain scene export.

## Quick Start

Install EmbodiChain with the generative-simulation dependencies. See
[Installation (gensim extra)](../../quick_start/install.md#optional-generative-simulation-gensim).

Prepare a Scene Engine JSON config, then run:

```bash
embodichain scene-engine \
    --image /path/to/scene.png \
    --output_root /path/to/scene_output \
    --config /path/to/scene_engine_config.json
```

Preview the result:

```bash
embodichain preview-scene --output_root /path/to/scene_output
```

Use `--viser` for a browser-based preview, or `--headless` to validate the
export without opening a window:

```bash
embodichain preview-scene \
    --output_root /path/to/scene_output \
    --viser
```

The equivalent module commands are:

```bash
python -m embodichain.gen_sim.scene_engine.cli.start --help
python -m embodichain.gen_sim.scene_engine.cli.preview --help
```

## Requirements and Configuration

The input must be one `.jpg`, `.jpeg`, or `.png` image with one main table and
visible, separate tabletop assets. The pipeline requires an OpenAI-compatible
VLM, an image-segmentation service, and a geometry-generation service.

Pass their settings through `--config`. Keep credentials outside version
control. `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_MAX_ATTEMPTS` override the corresponding LLM settings.

```json
{
  "llm": {
    "openai_compatible": {
      "api_key": "<api-key>",
      "model": "<vision-model>",
      "base_url": "https://example.com/v1",
      "default_query": {},
      "max_attempts": 3
    }
  },
  "image_segmentation": {
    "base_url": "http://segmentation-host:port",
    "timeout_s": 120,
    "max_attempts": 3,
    "health_path": "/health",
    "segment_single_object_path": "/segment_single_object"
  },
  "geometry_generation": {
    "base_url": "http://geometry-host:port",
    "timeout_s": 600,
    "max_attempts": 3,
    "health_path": "/health",
    "generate_objects_path": "/generate_objects"
  }
}
```

The configured endpoint paths must match the deployed services. Geometry uses
one ordered `generate_objects` request for all masks; a single-object scene
uses the same request with one mask.

## Output

Each run refreshes the intermediate stage directories and writes the final
portable export:

```text
<output_root>/
|-- scene_understanding/
|-- scene_segmentation/
|-- scene_generation/
`-- scene_export/
    |-- scene_config.json
    `-- mesh_assets/
        |-- <table-id>/<table-id>.glb
        `-- <asset-id>/<asset-id>.glb
```

`scene_export/scene_config.json` has format
`"embodichain.scene-export/v1"`. It contains the table under `background` and
the settled assets under `rigid_object`; mesh paths are relative to
`scene_export/`.

The internal scene layout is y-up. The exporter copies GLBs unchanged and
converts final positions and rotations to the simulator's z-up convention.
This is a scene-only export, not a `run-env` configuration: it does not define
a robot or task.

## Python API

Use `generate_scene_from_image` to run the full pipeline:

```python
from embodichain.gen_sim.scene_engine.pipeline.generate import (
    generate_scene_from_image,
)

scene = generate_scene_from_image(
    image_path="scene.png",
    output_root="scene_output",
    llm_config_path="scene_engine_config.json",
    image_segmentation_config_path="scene_engine_config.json",
    geometry_generation_config_path="scene_engine_config.json",
)
```
