# Per-Instance Visual Texture Randomization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make \`randomize_visual_material\` safely assign random, unique, or fixed base-color textures to selected object instances across vectorized environments.

**Architecture:** Keep the class-style functor and per-environment material instances. Add a cached DexSim texture-reference path to \`VisualMaterialInst\`, select textures by global environment ID, retrieve only targeted instances, and remove unsafe runtime cleanup.

**Tech Stack:** Python 3.11+, PyTorch, DexSim 0.4.3, pytest, existing \`Functor\`/\`VisualMaterialInst\` APIs.

## Global Constraints

- Preserve default random-with-replacement behavior.
- Use full annotations and \`from __future__ import annotations\` for new Python APIs.
- Preserve Apache headers on modified Python files.
- Never call \`env.clean_materials()\` during an episode.
- Do not modify the unrelated working-tree change in \`embodichain/gen_sim/action_agent_pipeline/generation/action_agent_config_defaults.yaml\`.
- \`RigidObjectGroup\` remains out of scope.

---

## File Map

- Modify \`embodichain/lab/sim/material.py\`: accept a pre-created DexSim texture reference.
- Modify \`embodichain/lab/gym/envs/managers/randomization/visual.py\`: normalize IDs, cache references, select textures, and apply only to targets.
- Create \`tests/gym/envs/managers/test_randomize_visual_material.py\`: fake-backed unit tests and simulator integration coverage.

## Task 1: Add the texture-reference material API

**Files:** \`embodichain/lab/sim/material.py:244-271\`; new manager test file.

**Produces:** \`VisualMaterialInst.set_base_color_texture(texture_ref=...)\`, storing the reference and calling \`set_base_color_map(texture_ref)\` without creating another texture.

- [ ] **Step 1: Write the failing test**

\`\`\`python
def test_set_base_color_texture_uses_texture_ref():
    material = FakeMaterial()
    instance = VisualMaterialInst("instance", material)
    texture_ref = object()
    instance.set_base_color_texture(texture_ref=texture_ref)
    assert material.get_inst("instance").base_color_map is texture_ref
    assert instance.base_color_texture is texture_ref
\`\`\`

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py::test_set_base_color_texture_uses_texture_ref -q
\`\`\`

Expected: failure because \`texture_ref\` is not accepted.

- [ ] **Step 3: Implement the minimal branch**

Add \`texture_ref: object | None = None\`; reject or warn if combined with \`texture_path\` or \`texture_data\`; assign \`self.base_color_texture = texture_ref\` and call the DexSim material-instance setter.

- [ ] **Step 4: Verify**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py::test_set_base_color_texture_uses_texture_ref -q
pytest tests/sim/objects/test_rigid_object.py -q
\`\`\`

Expected: both commands pass.

## Task 2: Add target normalization and texture selection

**Files:** \`embodichain/lab/gym/envs/managers/randomization/visual.py\`; new manager test file.

**Produces:** \`_normalize_env_ids(env_ids, num_envs) -> list[int]\` and \`_select_texture_indices(mode, target_ids, num_textures, texture_indices) -> list[int]\`.

- [ ] **Step 1: Write failing tests**

\`\`\`python
def test_normalize_env_ids_supports_all_input_forms():
    assert _normalize_env_ids(None, 3) == [0, 1, 2]
    assert _normalize_env_ids(torch.tensor([2, 0]), 3) == [2, 0]
    assert _normalize_env_ids([1], 3) == [1]
    assert _normalize_env_ids(slice(None), 3) == [0, 1, 2]

def test_texture_selection_modes():
    assert sorted(_select_texture_indices("without_replacement", [0, 1, 2], 3, None)) == [0, 1, 2]
    assert _select_texture_indices("fixed", [3, 1], 4, {1: 2, 3: 0}) == [0, 2]
    with pytest.raises(ValueError, match="without_replacement"):
        _select_texture_indices("without_replacement", [0, 1], 1, None)
\`\`\`

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py -k "normalize or selection" -q
\`\`\`

Expected: failure because the helpers do not exist.

- [ ] **Step 3: Implement helpers**

Handle \`None\`, tensors, Python sequences, and \`slice(None)\`; validate IDs. Implement \`random\` with replacement, \`without_replacement\` with \`torch.randperm\`, \`cycle\` modulo source count, and \`fixed\` using global \`env_id -> texture_index\`. Raise \`ValueError\` for unknown modes, missing mappings, missing IDs, insufficient sources, or invalid indices.

- [ ] **Step 4: Verify and commit**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py -k "normalize or selection" -q
git add embodichain/lab/gym/envs/managers/randomization/visual.py tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "test: define per-instance texture selection behavior"
\`\`\`

Expected: focused tests pass and the commit contains only Task 2 files.

## Task 3: Integrate targeted assignment into the functor

**Files:** \`embodichain/lab/gym/envs/managers/randomization/visual.py:538-798\`; new manager test file.

**Interface:** Extend \`__call__\` with \`texture_sampling: str = "random"\`, \`texture_indices: Mapping[int, int] | None = None\`, and \`texture_scope: str = "per_material"\`.

- [ ] **Step 1: Write failing tests**

Use fake rigid objects and material instances. Call with \`env_ids=torch.tensor([1, 3])\`; assert only material instances 1 and 3 change. Add a fixed mapping test for \`{1: 2, 3: 0}\` and an articulation \`per_instance\` test asserting all selected links in one environment share one selected texture.

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py -k "partial or fixed_assignment or per_instance" -q
\`\`\`

Expected: failure because the current functor loops its full cached list and has no selection parameters.

- [ ] **Step 3: Implement**

Normalize IDs at call time; build all plans with \`len(target_ids)\` rows; select texture indices by global IDs; retrieve \`RigidObject.get_visual_material_inst(env_ids=target_ids)\` or the articulation equivalent; apply each row to its target. For \`per_instance\`, reuse one index across selected links. Apply metallic, roughness, and IOR in both texture and generated-color branches. Remove the final \`env.clean_materials()\` call.

- [ ] **Step 4: Verify**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py -q
pytest tests/gym/envs/managers -q
\`\`\`

Expected: all focused and manager tests pass.

## Task 4: Cache texture references and add integration coverage

**Files:** \`embodichain/lab/gym/envs/managers/randomization/visual.py:589-607\`; new manager test file.

- [ ] **Step 1: Write failing cache test**

Fake \`create_color_texture\` with a counter; invoke the functor twice and assert each source image is converted exactly once.

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py::test_texture_references_are_created_once -q
\`\`\`

Expected: failure because current assignment recreates a GPU texture from image data each time.

- [ ] **Step 3: Implement and test**

Convert each padded RGBA image once during initialization, store \`self.texture_refs\`, and pass the selected reference through \`_randomize_texture\` and \`_randomize_mat_inst\`. Use canonical resolved source paths as cache keys. Add real four-environment tests for distinct no-replacement assignments, fixed mapping, partial-reset isolation, and a second reset with valid handles.

\`\`\`bash
pytest tests/gym/envs/managers/test_randomize_visual_material.py -q
pytest tests/sim/objects/test_rigid_object.py -q
pytest tests/sim/objects/test_articulation.py -q
\`\`\`

Expected: all selected tests pass; if renderer setup is unavailable, report that limitation while retaining pure tests.

- [ ] **Step 4: Commit**

\`\`\`bash
git add embodichain/lab/gym/envs/managers/randomization/visual.py embodichain/lab/sim/material.py tests/gym/envs/managers/test_randomize_visual_material.py
git commit -m "feat: support per-instance visual texture randomization"
\`\`\`

## Task 5: Final verification and documentation

**Files:** the relevant existing randomization documentation page, if present; no unrelated files.

- [ ] **Step 1:** Document \`texture_sampling\`, \`texture_indices\`, and \`texture_scope\` with the fixed and no-replacement examples from the specification.
- [ ] **Step 2:** Run \`black --check --diff --color ./\` and \`git diff --check HEAD^ HEAD\`; both must pass.
- [ ] **Step 3:** Run \`pytest tests/gym/envs/managers -q\` and \`pytest tests/gym tests/sim/objects -q\`; no new failures may be attributable to this feature.
- [ ] **Step 4:** Review \`git status --short\`, \`git diff HEAD^ --stat\`, and the final feature diff; preserve the pre-existing unrelated user change.
