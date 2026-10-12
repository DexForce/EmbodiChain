:orphan:

Mesh part segmentation and contact affordance
=============================================

The standalone mesh toolkit uses Blender multiview renders and the Codex CLI
to estimate contact suitability conditioned on an object and task description.
An optional target part produces separate membership scores and a segmented
submesh. Part membership is independent of grasp suitability: a difficult
handle attachment can still belong to the handle.

The default model is ``gpt-6-astra``. Codex authentication and compatible CLI
installation are external prerequisites. Install the ``mesh-affordance`` extra
for geometry and rendering dependencies, or use a Python environment containing
NumPy, SciPy, Trimesh, Pillow and bpy. Blender runs in an isolated subprocess.

Toolkit API
-----------

.. currentmodule:: embodichain.toolkits.mesh_affordance

.. autosummary::

   MeshAffordanceCfg
   MeshAffordanceResult
   analyze_mesh_affordance
   prepare_mesh_affordance
   score_mesh_affordance

.. autoclass:: MeshAffordanceCfg
   :members:

.. autoclass:: MeshAffordanceResult
   :members:

.. autofunction:: analyze_mesh_affordance

.. autofunction:: prepare_mesh_affordance

.. autofunction:: score_mesh_affordance

Indexing and output contracts
-----------------------------

Triangle OBJ input preserves the original ``v`` record order, including unused
vertices, without texture or normal seam duplication. Other formats require one
mesh and use the unprocessed Trimesh vertex order saved in ``geometry.npz``.
Rendering applies only a recorded translation and uniform scale; output geometry
remains in the source coordinate frame.

``affordance.npz`` stores source geometry, vertex scores and confidence, validity
and selection masks, triangle patch IDs and patch scores. With ``target_part``,
it also contains part scores, confidence, vertex and face masks. A part submesh's
local indices map to the source through ``target_part.npz``'s
``source_vertex_ids`` and ``source_face_ids``. Submesh extraction selects whole
triangles; its boundary vertex set can differ from the thresholded vertex mask.

Scores are patchwise model estimates averaged onto vertices using incident
triangle area. They are neither calibrated probabilities nor force-closure,
collision or reachability guarantees. Confidence is also a subjective model
estimate. Existing completed runs are preserved; use a new directory for a new
task. The preparation step does not invoke Codex, and the scoring step can retry
prepared evidence after a model or CLI failure.

Surface computations
--------------------

These pure array routines do not depend on simulation objects, Blender or Codex.
Partitions follow surface adjacency and penalize bends. Exact duplicate positions
are welded only when computing adjacency; disconnected sheets do not share a
patch, and non-manifold edges are barriers.

.. currentmodule:: embodichain.compute.geometry.surface

.. autosummary::

   partition_surface
   face_scores_to_vertices

.. autofunction:: partition_surface

.. autofunction:: face_scores_to_vertices


Provider selection
------------------

Both providers run through the Codex harness. ``provider="openai"`` (default)
uses existing Codex authentication; ``model=None`` selects ``gpt-6-astra``.
``provider="deepseek"`` uses DeepSeek's OpenAI-compatible Responses API and
requires ``provider_config`` pointing to a local JSON file with ``base_url``,
``api_key`` and an optional ``model``. An explicit model overrides the file's
model; otherwise DeepSeek defaults to the vision-capable ``deepseek-flash``.

The credential is passed only through the Codex child environment, excluded
from model shell tools, and omitted from prompts, command arguments and result
configuration. The repository-root ``.mesh_affordance.local.json`` is ignored
by Git. Provider selection uses per-invocation overrides without changing the
user's global Codex configuration. Reports record the effective provider and
model. Known text-only DeepSeek models are rejected before inference.

The decoder accepts a single outer Markdown JSON fence for provider
compatibility, while preserving strict schema and patch-coverage validation.
Raw replies are saved as ``response.raw.txt`` and normalized replies as
``response.json``.
