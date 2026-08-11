# Scene Engine

Scene Engine reconstructs a table-top scene from one image. It identifies the
table and visible objects, segments their masks, generates simulation-ready
meshes, refines the object layout on the table, and exports a scene that can be
loaded by EmbodiChain.

## Quick Start

Install EmbodiChain with the `gensim` extra first; see
[Installation](../../quick_start/install.md#optional-generative-simulation-gensim).

Configure the required services in `embodichain/gen_sim/.env`, then generate a
scene:

```bash
embodichain scene-engine \
    --image /path/to/scene.png \
    --output_root /path/to/scene_output
```

The same command is available through the package entry point:

```bash
python -m embodichain scene-engine \
    --image /path/to/scene.png \
    --output_root /path/to/scene_output
```

## Configuration

Scene Engine reads the LLM, segmentation, image-generation, and
geometry-generation settings from `embodichain/gen_sim/.env`:

```bash
OPENAI_API_KEY="your-api-key"
OPENAI_MODEL="your-model"
OPENAI_BASE_URL="https://api.openai.com/v1"
SCENE_ENGINE_OPENAI_DEFAULT_QUERY="{}"
OPENAI_MAX_ATTEMPTS=3

SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL="http://host:port"
SCENE_ENGINE_IMAGE_SEGMENTATION_TIMEOUT_S=30
SCENE_ENGINE_IMAGE_SEGMENTATION_MAX_ATTEMPTS=3
SCENE_ENGINE_IMAGE_SEGMENTATION_HEALTH_PATH="/health"
SCENE_ENGINE_IMAGE_SEGMENTATION_BY_PROMPT_PATH="/segment_by_prompt"

SCENE_ENGINE_IMAGE_GENERATION_BASE_URL="http://host:port"
SCENE_ENGINE_IMAGE_GENERATION_TIMEOUT_S=120
SCENE_ENGINE_IMAGE_GENERATION_MAX_ATTEMPTS=3
SCENE_ENGINE_IMAGE_GENERATION_HEALTH_PATH="/health"
SCENE_ENGINE_IMAGE_GENERATION_BY_PROMPT_PATH="/generate_image_by_prompt"

SCENE_ENGINE_GEOMETRY_GENERATION_BASE_URL="http://host:port"
SCENE_ENGINE_GEOMETRY_GENERATION_TIMEOUT_S=600
SCENE_ENGINE_GEOMETRY_GENERATION_MAX_ATTEMPTS=3
SCENE_ENGINE_GEOMETRY_GENERATION_HEALTH_PATH="/health"
SCENE_ENGINE_GEOMETRY_GENERATION_OBJECTS_PATH="/generate_multiple_objects"
```

## Processing Flow

- **Scene understanding**: analyzes the image and segments the table and visible objects.
- **Scene generation**: generates meshes, prepares SimReady geometry, detects the table support surface, and refines the table-top layout.
- **Scene export**: copies the final GLBs and writes a portable z-up scene export.

## Output and Preview

The important final outputs are:

```text
scene_output/
|-- scene_understanding/     # Object analysis, masks, and stage JSON
|-- scene_generation/        # Generated, SimReady, and layout-debug artifacts
`-- scene_export/
    |-- mesh_assets/         # Final GLBs
    `-- scene_config.json    # Exported scene description
```

Validate the export without opening a window:

```bash
embodichain preview-scene \
    --output_root /path/to/scene_output \
    --headless
```

For an interactive preview, omit `--headless`. Add `--viser` to publish the
scene through Viser.
