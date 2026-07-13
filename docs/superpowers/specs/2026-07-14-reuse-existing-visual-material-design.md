# Reuse Existing Visual Material for Material Randomization

- **Date:** 2026-07-14
- **Status:** Design (pending implementation plan)
- **Owner:** yuecideng
- **Related files:** `embodichain/lab/sim/material.py`, `embodichain/lab/gym/envs/managers/randomization/visual.py`, `embodichain/lab/sim/objects/rigid_object.py`, `embodichain/lab/sim/objects/articulation.py`

## 1. Background

`randomize_visual_material` (in `embodichain/lab/gym/envs/managers/randomization/visual.py`) currently randomizes an object's appearance by **creating a brand-new visual material and replacing the object's material with it**:

```python
# visual.py:622-636
mat = env.sim.create_visual_material(cfg=VisualMaterialCfg(base_color=[1,1,1,1], uid=...))
entity.set_visual_material(mat)          # REPLACE semantics -> discards the parsed material
...
self._mat_insts = entity.get_visual_material_inst()
```

Two problems follow:

1. **Original appearance is discarded.** `set_visual_material` calls `MeshObject.set_material(...)` / `Articulation.set_material(link, ...)` with **replace** semantics, so the material dexsim parsed from the mesh/URDF (including embedded textures and PBR values) is thrown away. Randomization always starts from a blank white material, never the asset's own look.
2. **Overhead.** A new `Material` template is created per functor via `create_visual_material` -> `env.create_pbr_material`, and `__call__` ends with `env.clean_materials()` every reset. Per-reset texture uploads (`env.create_color_texture` inside `set_base_color_texture(texture_data=...)`) also accumulate.

The existing code even carries a TODO acknowledging the gap:

```python
# visual.py:622
# TODO: we may need to get the default material instance from the asset itself.
```

## 2. Goals

- **Reuse the object's existing material** (the one dexsim parsed from the mesh/URDF) instead of creating a new one. Do not call `create_visual_material` / `create_pbr_material` on the new path.
- **Preserve the asset's original appearance** as a stable baseline: randomization perturbs *on top of* the original, and an explicit "original texture" tier restores it faithfully across resets.
- **Reduce per-reset overhead**: pre-create textures once, avoid `Env.clean_materials()`.
- **Backward compatible**: existing configs keep working; the old behavior is reachable via a flag.

## 3. Non-goals

- Changing the `VisualMaterial` / `VisualMaterialInst` abstraction for non-randomization callers.
- Per-segment (per-mesh-id) independent randomization tiers (v1 uses a per-object tier applied to all segments).
- Reading back arbitrary material properties that dexsim does not expose (see §4).

## 4. Feasibility — dexsim API constraints

dexsim is a compiled pybind11 extension; its API surface was verified from the `.pyi` stubs at `site-packages/dexsim/cuda/pybind/{engine,models,environment}.pyi`.

**The material system is a thin wrapper over dexsim `Material` (template) / `MaterialInst` (instance).** `VisualMaterial(cfg, mat)` holds a `Material`; `VisualMaterialInst(uid, mat)` looks up the live `MaterialInst` via `mat.get_inst(uid)` (`material.py:108-212`).

Retrieval APIs that make reuse possible:

- `MeshObject.get_material() -> MaterialInst` (`models.pyi:521`) — the material currently attached to a parsed mesh.
- `RenderBody.get_material(mesh_id=0) -> MaterialInst` (`engine.pyi:5193`) — per segment.
- `RenderBody.get_mesh_count() -> int` (`engine.pyi:5203`) — number of material segments.
- `RenderBody.set_material(mesh_id, material_inst)` (`engine.pyi:5322`) — **swap a whole instance onto a segment**.
- `Articulation.get_render_body(link_name) -> RenderBody` (`engine.pyi:472`).
- `MaterialInst.get_template() -> Material` (`engine.pyi:3354`), `Material.create_inst(name) -> MaterialInst` (`engine.pyi:2911`), `Material.get_inst(name)` (`engine.pyi:2950`).
- `set_base_color_map(texture: Texture, ...)` overload (`engine.pyi:3499`) — accepts a pre-created `Texture` object, and `Texture` objects can be shared across `MaterialInst`s.

**Critical limitation — cannot read back textures.** `MaterialInst` getters exist only for `base_color`, `emissive`, `ior`, `roughness`, `get_base_color_map()` (returns a **name string**, not a `Texture`), `name`, `template`. There is **no** getter for `metallic` or for any texture map's `Texture` object; `Texture` is opaque (`get_width/height/name` only); there is **no `find_texture(name)`**. Therefore "snapshot the original texture by reading it back, then restore" is **infeasible**.

**Resolution:** keep a reference to the original `MaterialInst` (never mutate it) and restore the original tier by **swapping the whole instance back** via `RenderBody.set_material(mesh_id, original_inst)`. This sidesteps the readback limitation entirely.

**Cleanup semantics:** `RenderBody.clean_material()` = "reset render body to default material" (per-body); `Env.clean_materials()` = "clean all materials" (registry-level, ambiguous). The new path must **not** call `Env.clean_materials()`.

## 5. Design — Approach: instance-swap + pre-created textures

### 5.1 Architecture & components

**`embodichain/lab/sim/material.py` (small change)**
- Add the ability to set a **pre-created `Texture` object** on `VisualMaterialInst` by extending `set_base_color_texture` with a `texture_obj: Texture | None = None` param. When set, it calls `MaterialInst.set_base_color_map(texture_obj)` directly (no `create_color_texture`, no re-upload). Purpose: library textures are uploaded once at init and reused per reset.
- Other setters are unchanged. The working instance is wrapped normally as `VisualMaterialInst(name, template)`; `template.get_inst(name)` resolves to it, so existing setters work as-is.

**`embodichain/lab/sim/objects/rigid_object.py` & `articulation.py` (new methods)**
- `get_existing_visual_material(env_ids, shared) -> ReuseMaterialState`: for each env (first only if `shared`) and each segment `mesh_id in range(get_mesh_count())`, fetch `original_inst = render_body.get_material(mesh_id)` (kept immutable), get `template = original_inst.get_template()`, create `working = template.create_inst(name)` and wrap as `VisualMaterialInst(name, template)`. Returns per-env segment list `[{original_inst, working_inst, mesh_id}]` plus render-body/segment locating info. The Articulation variant takes `link_names` and uses `get_render_body(link_name)`.
- `apply_render_material_inst(env_idx, mat_inst, mesh_id=...)`: swap a `MaterialInst` onto the given env's render-body segment via `RenderBody.set_material(mesh_id, inst)`.

**`embodichain/lab/gym/envs/managers/randomization/visual.py` (`randomize_visual_material`)**
- New config params: `fallback_to_new: bool = False`, `p_original`/`p_library`/`p_solid` (three-tier probabilities). Backward-compatible derivation from existing `random_texture_prob` when none of `p_*` are set.
- `__init__`: branch on `fallback_to_new`. New path calls `get_existing_visual_material` and pre-creates library `Texture`s. Old path is preserved verbatim.
- `__call__`: branch on `fallback_to_new`. New path does the three-tier swap and does **not** call `Env.clean_materials()`. Old path (including `clean_materials`) is preserved.

**`sim_manager.py`: no change** (the new path does not use `create_visual_material`).

**Responsibility boundary:** object classes only provide "fetch existing material + swap instance" primitives; all randomization logic (sampling, tier probabilities, texture selection) stays in the functor.

### 5.2 Init data flow (new path, `fallback_to_new=False`)

1. Resolve entity (RigidObject/Articulation/plane) — as today.
2. Preload library texture tensors (existing logic, into the sim texture cache).
3. **Pre-create `Texture` objects**: for each library tensor, call `env.create_color_texture(tensor, has_alpha=True)` once and cache at **sim level** keyed by `path+idx` (shared across functors). Functor holds the handles.
4. `entity.get_existing_visual_material(env_ids, shared)` builds `ReuseMaterialState` as in §5.1.
5. Store `self._reuse_state`. Do **not** call `create_visual_material` / `set_visual_material` / `clean_materials`.

### 5.3 Per-reset data flow (new path)

1. Resolve `env_ids`; sample a **per-env tier** with `torch.multinomial([p_original, p_library, p_solid])` (shared mode: sample one, broadcast).
2. Sample the per-env randomization plan (`base_color`/`metallic`/`roughness`/`ior`) — as today; used only by library/solid tiers.
3. For each env `i`, for each segment:
   - **original tier**: `apply_render_material_inst(i, original_inst, mesh_id)` — swap the original back; zero upload, zero mutation.
   - **library tier** (library non-empty): pick a pre-created `Texture`; on `working`: `set_base_color_texture(texture_obj=tex)` + set plan properties (`set_base_color`/`set_metallic`/`set_roughness`/`set_ior`); `apply_render_material_inst(i, working.mat, mesh_id)`.
   - **solid tier**: on `working`: set a 2×2 solid texture + `base_color`; `apply_render_material_inst(i, working.mat, mesh_id)`.
   - library tier with empty library degrades to solid tier.
4. Do **not** call `Env.clean_materials()` (no accumulation: library textures created once at init, working instances created once at init, solid 2×2 upload negligible).

The working instance is reused across resets and mutated in place safely: it is always re-set before being displayed; the original instance is never mutated. In shared mode all envs share one tier/appearance (matching current `is_shared_visual_material` semantics).

### 5.4 Config surface (`FunctorCfg.params`)

| Parameter | Default | Notes |
|---|---|---|
| `entity_cfg` | required | as today |
| `texture_path` | None | as today; library texture source |
| `base_color_range` / `metallic_range` / `roughness_range` / `ior_range` | None | as today |
| `random_texture_prob` | 0.5 | **kept**; used directly in fallback; backward-compat source for three-tier in new mode |
| `fallback_to_new` | False | True = old path (create+replace+clean), fully compatible |
| `p_original` | None | original-tier prob; None -> 0 |
| `p_library` | None | library-tier prob; None -> `random_texture_prob` |
| `p_solid` | None | solid-tier prob; None -> `1 - p_original - p_library` |
| `shared` | None | whether working instances are shared across envs; None -> current default |

**Backward-compatible derivation:** when all three `p_*` are unset -> `p_original=0, p_library=random_texture_prob, p_solid=1-random_texture_prob`. Existing configs then run on the new path with zero changes (gaining slot reuse + pre-created library textures + no `clean_materials`), with the original tier off by default (opt-in via `p_original > 0`). If any `p_*` is set explicitly, use them and normalize (warn if sum != 1).

**Solid-tier implementation detail:** 2×2 solid texture created per reset (4-pixel upload, negligible); if profiling shows cost, switch to a pre-generated palette at init. Library textures (large images) pre-creation is the main optimization.

### 5.5 Edge cases & error handling

| Case | Handling |
|---|---|
| `render_body.get_material(mesh_id)` returns None | dexsim normally always attaches a default; if None, degrade that object to `fallback_to_new` + `log_warning` |
| `get_template()` fails / returns None | cannot create working instance -> degrade to `fallback_to_new` + warning |
| `create_inst(name)` fails | degrade to `fallback_to_new` + warning |
| Multi-segment mesh (`get_mesh_count()>1`) | per-segment original+working; tier is **per-object** (all segments same tier per reset); randomization applied uniformly |
| Articulation `link_names` | only listed links randomized; others keep original; `get_render_body(link)` None -> skip + warning |
| `default_plane` | reuse `plane_mat` instance as original; same three-tier swap; single instance, shared semantics |
| Empty texture library | library tier degrades to solid; if `p_library>0` but empty, `log_info` |
| Probabilities sum != 1 | normalize + `log_warning` |
| `p_*` all unset | backward-compat derivation (§5.4) |
| shared mode | sample a single tier, broadcast (no per-env diversity; matches current shared behavior); non-shared samples per env |
| Object with `cfg.shape.visual_material` pre-set | no conflict — reuse uses whatever is currently on the render body (the pre-set one) |
| `fallback_to_new=True` | old path preserved verbatim (create+replace+clean_materials); zero regression |
| entity not found | early return as today |

**Robustness:** the swap model is robust to the per-env instance independence question (which the stubs cannot confirm). The original instance is immutable; working instances are created per-env by us. Even if dexsim shares one original instance across envs, `RenderBody.set_material` is per-render-body, so swapping onto env0's render body does not affect env1.

## 6. Runtime verification points (confirm during implementation; do not affect the design)

- `Material.create_inst(name)` produces a blank instance vs a copy of the original (swap model works either way).
- `RenderBody.set_material(mesh_id, inst)` per-reset cost (expected: cheap pointer swap).
- `set_base_color_map(pre-created Texture)` behaves as the stub indicates.
- `original_inst.get_template()` returns a usable `Material` at runtime.
- Whether per-env material instances are independent (swap model does not depend on this, but record the actual behavior).

## 7. Testing plan

Tests scaffolded with `/add-test`, placed under `tests/`. dexsim is a compiled C++ extension, so unit tests mock `MeshObject`/`RenderBody`/`Material`/`MaterialInst`; integration smoke tests are gated on dexsim/GPU availability.

**Unit tests (mocked):**
- `get_existing_visual_material`: correctly fetches original, creates+wraps working, multi-segment count, shared vs non-shared instance counts.
- `apply_render_material_inst`: calls `RenderBody.set_material(mesh_id, inst)` with correct args.
- Tier sampling: `multinomial` distribution, shared-mode broadcast, probability normalization + warning.
- Backward-compat derivation: `p_*` all None derives correctly from `random_texture_prob`.
- Pre-created `Texture` shared across functors via sim-level cache (same `texture_path` creates once).
- Degradation: `get_material` None / `get_template` fails -> `fallback_to_new` + warning.

**Integration tests (require dexsim):**
- RigidObject + mesh, new path: assert `create_visual_material` and `Env.clean_materials` are **not** called; render-body material swapped correctly per tier.
- `fallback_to_new=True`: assert old path taken (create+set+clean all called); behavior matches current.
- `p_original>0`: original tier swaps `original_inst` back onto the render body.
- Multi-segment mesh: all segments handled.
- Articulation + `link_names`: only listed links randomized.
- Empty texture library: library tier degrades to solid, no error.
- Cross-reset consistency: instance/texture counts do not grow over repeated resets.

## 8. Alternatives considered

- **Approach 2 — in-place mutation + partial snapshot (no swap).** Wrap the existing `MaterialInst` and mutate in place; snapshot the readable props (`base_color`/`emissive`/`ior`/`roughness`). **Rejected:** `metallic` and all texture maps cannot be read back, so the "original texture" tier is not faithful (the original is lost after the first mutation). Does not satisfy the three-tier requirement.
- **Approach 3 — reload original texture from the mesh file.** Parse the object's mesh file (OBJ/MTL, glTF, URDF material) at init to recover the original texture path, load it once, and restore it for the original tier. **Rejected:** requires parsing multiple mesh formats (fragile), URDF material parsing, and does not handle procedural/file-less meshes; high complexity and risk. The swap model achieves the same faithful original tier far more cleanly.

## 9. Open questions / future work

- Per-segment independent tiers (currently per-object).
- Pre-generated solid-color palette if per-reset 2×2 upload shows up in profiling.
- Exposing `shared` as a first-class documented param once current default is confirmed.
