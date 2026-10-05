# Mesh contact affordance

This toolkit estimates task-conditioned contact preferences from a mesh,
object description and task description. It does not validate grasp poses,
force closure, robot reachability or scene collisions, and does not change
Atomic Skill execution or its existing affordance contracts.
An optional `target_part` / `--target-part` adds independent semantic part
segmentation; handle membership is not derived from graspability thresholds.

## Entry points and ownership

- `embodichain/toolkits/mesh_affordance/__init__.py` exports `MeshAffordanceCfg`,
  `MeshAffordanceResult`, `prepare_mesh_affordance`, `score_mesh_affordance`
  and `analyze_mesh_affordance`.
- `python -m embodichain.toolkits.mesh_affordance {run,prepare,score}` owns the
  standalone CLI. `run` accepts `--mesh`, `--object`, `--task`, `--output`.
- `_pipeline.py` owns orchestration and artifact persistence; `_mesh.py` owns
  loading and indexing; `_render.py` runs bpy in a separate Python process;
  `_codex.py` owns Codex CLI invocation and strict model-response validation;
  `_providers.py` resolves provider models and transient local credentials.
- `embodichain/compute/geometry/surface.py` owns pure array computations:
  geodesic surface partitions and area-weighted face-to-vertex transfer.

## Resolution and indexing

An existing local mesh path wins; other paths use the
[data asset resolver](../data-assets/data-assets.md). Triangle OBJ loading
preserves every original `v` entry, including unused and seam vertices; UVs,
normals and materials cannot duplicate/reorder vertices. Nontriangle OBJ
faces are rejected with a triangulation instruction. Other formats require a
single Trimesh; the saved `vertices` array defines the index contract.

Adjacency welds exact duplicate positions only for partition computation.
Geodesic distances penalize bends; disconnected sheets never share a patch.
Non-manifold edges are barriers. Too many components require more patches or
mesh repair. Default partition count is 64.

## Lifecycle and artifacts

1. Prepare a new/empty directory with `geometry.npz` and `evidence.json`.
   Original coordinates are preserved; render coordinates use a recorded
   bounding-box translation and uniform scale, without an assumed up axis.
2. Blender CPU Cycles creates eight views, four paired clay/patch contact
   sheets, patch labels and visibility metadata. The Python executable must
   contain bpy; the default is the caller's Python.
3. `codex exec` uses image attachments and a JSON output schema with either
   `provider="openai"` (default) or `provider="deepseek"`. `model=None` resolves
   to `gpt-6-astra` for OpenAI, or the local config model / `deepseek-flash` for
   DeepSeek. Explicit model IDs take precedence. OpenAI reuses Codex login;
   DeepSeek uses its Responses API via per-invocation Codex provider overrides.
   User Codex config is ignored, tool access is read-only, and the global Codex
   configuration is never modified. Override the CLI executable using
   `codex_executable` / `--codex-executable`.
4. Responses accept a JSON object or one outer Markdown JSON fence; surrounding
   prose and multiple objects are rejected. `response.raw.txt` preserves the raw
   answer and `response.json` stores normalized JSON. The response must cover
   every patch exactly once, with finite scores and
   confidence in `[0,1]`. Invalid responses fail without result publication.
5. Area-weighted incident-face scores become vertex scores. Unreferenced
   vertices have score zero and `valid_mask=False`. The selection mask requires
   valid vertices, score >= 0.7 and confidence >= 0.5 by default.
6. Outputs include `affordance.npz`, colored `affordance.ply`, `heatmap.png`,
   `report.json`, structured model response, exact command, prompt and logs.
7. With a target part, `affordance.npz` also includes `part_scores`,
   `part_confidence`, `part_mask`, `part_face_mask` and patch membership arrays.
   The default part membership threshold is 0.5; selection also requires model
   confidence >= `min_confidence`. `segmentation.png` / `.ply` show the full mesh
   with the part highlighted. `target_part.npz` / `.ply` contain selected whole
   faces; NPZ maps local indices back via `source_vertex_ids` / `source_face_ids`.
   Face-selected boundary vertices can differ from the area-averaged vertex
   mask. An absent part yields empty NPZ arrays and no part PLY.

Scores are patch estimates, not independently predicted vertex values or
calibrated probabilities. Geometry hashes guard evidence reuse. Existing
completed results are never overwritten. Each subprocess has a timeout and
on POSIX its process group is terminated on timeout/interruption.

## DeepSeek configuration and credentials

`provider_config` / `--provider-config` points to a local JSON file containing
`base_url`, `api_key`, and optional `model`. The path is resolved against the
caller's working directory. `.mesh_affordance.local.json` at the repository root
is ignored by Git; the tracked example in `examples/toolkits/mesh_affordance/`
contains a placeholder only. Connection contents never enter `MeshAffordanceCfg`
or its serialization. The prompt includes only mesh evidence fields.

The key is injected into the Codex subprocess environment as
`EMBODICHAIN_DEEPSEEK_API_KEY`, with explicit exclusion from model shell tools.
It is never put in command arguments; failure logs and response output are
redacted if a provider echoes it. Provider config errors avoid file contents.
Known text-only DeepSeek model IDs are rejected before inference. Reports
record the actual provider/model and `harness="codex"`. Switching providers on
the `score` CLI resets the saved model only when no explicit `--model` is given.

## Validation and diagnosis

Run `tests/compute/geometry/test_surface.py` and
`tests/toolkits/mesh_affordance/` for provider-free coverage. Real Blender/Codex
qualification uses the CoffeeCup example in `examples/toolkits/mesh_affordance/`.
Read `blender_*.stderr.log` or `codex.stderr.log` / `codex.stdout.log` for failures.
An unsupported-model/version error requires a compatible Codex binary;
`score` can retry prepared evidence with `--codex-executable` without rerendering.
