# Per-Instance Visual Texture Randomization Design

**Date:** 2026-07-14

**Status:** Approved design; awaiting written-spec review

## 1. Overview

Extend `randomize_visual_material` so each replica of a rigid object or
articulation can receive a reliably different base-color texture when
`num_envs > 1`. The extension must also support deterministic assignment by
environment ID and correctly target only the environments being reset.

The current object wrappers already create one `VisualMaterialInst` per
environment. The existing functor randomly selects a texture for each material
instance during full-batch calls, but it does not guarantee uniqueness or offer
an explicit per-environment mapping. Its cached full material list is also
incompatible with partial resets, and it calls DexSim's unsafe runtime material
cleanup API after every application.

## 2. Goals and Non-Goals

### Goals

1. Select a base-color texture independently for each targeted environment.
2. Guarantee distinct source textures for a reset when requested and enough
   textures are available.
3. Permit deterministic texture assignment by global environment ID.
4. Preserve existing random sampling with replacement as the default behavior.
5. Correctly support full resets, partial resets, and interval events.
6. Reuse material and texture resources across resets without invalidating live
   DexSim handles.

### Non-Goals

1. Add randomization of metallic, roughness, normal, or AO texture maps.
2. Support `RigidObjectGroup`; that requires a separately stored and retrievable
   per-environment material-instance collection.
3. Change the default plane's global-material behavior.

## 3. Public Configuration Interface

Keep `randomize_visual_material` as the public functor and retain all existing
parameters. Add the following optional parameters:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `texture_sampling` | `"random" \| "without_replacement" \| "cycle" \| "fixed"` | `"random"` | How folder textures are chosen. |
| `texture_indices` | `dict[int, int] \| None` | `None` | Global `env_id -> texture_index` mapping; required for `"fixed"`. |
| `texture_scope` | `"per_material" \| "per_instance"` | `"per_material"` | Whether an articulation samples one texture per selected link or one texture shared by all selected links in an environment. |

`random` retains current semantics: each target samples independently with
replacement, so duplicate textures are valid. `without_replacement` samples a
permutation and raises `ValueError` when fewer textures than targeted
environments exist. `cycle` assigns textures in source order modulo the source
count. `fixed` looks up every target environment in `texture_indices` and
raises `ValueError` for a missing or out-of-range entry.

For the requested behavior, configure:

```yaml
random_material:
  func: randomize_visual_material
  mode: reset
  params:
    entity_cfg: {uid: bottle}
    texture_path: DexsimMaterials/Bottle
    random_texture_prob: 1.0
    texture_sampling: without_replacement
    texture_scope: per_instance
```

For a deterministic mapping:

```yaml
random_material:
  func: randomize_visual_material
  mode: reset
  params:
    entity_cfg: {uid: bottle}
    texture_path: DexsimMaterials/Bottle
    texture_sampling: fixed
    texture_indices: {0: 2, 1: 0, 2: 3, 3: 1}
    texture_scope: per_instance
```

## 4. Architecture and Runtime Flow

### 4.1 Persistent material and texture ownership

At initialization, the functor creates one PBR material template and assigns a
unique material instance to every rigid-object replica or every selected
articulation link. The functor retains the template and the loaded texture
sources for its lifetime.

Textures are decoded once and converted to DexSim texture references once. The
cache key uses the resolved directory path rather than just its basename to
avoid collisions between unrelated texture folders with the same name.

`VisualMaterialInst.set_base_color_texture` gains a texture-reference input so
assignment can directly call DexSim's material-instance map setter without
creating a new GPU texture on every reset.

### 4.2 Targeted material retrieval

On each call, normalize `env_ids` to a CPU list of global environment IDs.
Accept `None`, tensors, Python sequences, and `slice(None)`. Return immediately
for an empty target set.

Retrieve material instances for exactly those IDs at call time:

- `RigidObject.get_visual_material_inst(env_ids=target_ids)`
- `Articulation.get_visual_material_inst(env_ids=target_ids, link_names=...)`

Do not iterate a material list cached for all environments. This ensures a
partial reset affects only the selected replicas and aligns every sampled plan
row with its material instance.

### 4.3 Sampling and application

1. Build a material-property plan of shape `(num_targets, ...)` for base color,
   metallic, roughness, and IOR.
2. Build texture indices keyed by the selected global environment IDs according
   to `texture_sampling`.
3. Apply scalar PBR properties for every target, independent of whether the
   texture branch or generated-color branch is taken.
4. If `random_texture_prob` selects the folder-texture branch, assign the
   cached texture reference indicated by the plan.
5. Otherwise, assign a generated solid-color texture. Generated colors remain
   per material for backward compatibility; `per_instance` reuses the same
   generated texture across selected articulation links in an environment.

The functor must never call `env.clean_materials()` or
`SimulationManager.clean_materials()` during an episode. DexSim documents that
the lower-level call invalidates live Python-side material and mesh references.
Simulation teardown remains responsible for cleanup.

## 5. Error Handling

- `texture_sampling` is unknown: raise `ValueError`.
- `fixed` without `texture_indices`: raise `ValueError`.
- A fixed mapping lacks a targeted global environment ID: raise `ValueError`.
- A texture index is negative or outside the loaded texture collection: raise
  `ValueError`.
- `without_replacement` has fewer source textures than target environments:
  raise `ValueError`.
- A texture-based strategy has no readable source textures: raise `ValueError`.
- If `random_texture_prob` is outside `[0, 1]`, raise `ValueError`.

## 6. Testing Plan

### Unit tests

1. Normalize `None`, tensor, sequence, and `slice(None)` environment IDs.
2. Verify `random`, `without_replacement`, `cycle`, and `fixed` index plans.
3. Verify validation errors for invalid modes, mappings, source counts, and
   index ranges.
4. Verify a partial target, for example `[1, 3]` of four environments, updates
   only instances 1 and 3.
5. Verify property values are applied whether a folder texture or a generated
   color texture is selected.

### Simulator integration tests

1. Create a four-environment rigid object with four distinct textures and use
   `without_replacement`; assert the assigned texture identities are distinct.
2. Apply `fixed` assignment to a partial reset and assert non-target material
   maps are unchanged.
3. Perform a second reset after a full reset; assert materials remain valid and
   can be updated.
4. For an articulation with multiple selected links, verify `per_instance`
   applies the same texture to each selected link within one environment and a
   different texture in another environment.
5. Confirm cached texture creation happens only during functor initialization,
   not on every reset.

## 7. Files Expected to Change

- `embodichain/lab/gym/envs/managers/randomization/visual.py`
- `embodichain/lab/sim/material.py`
- New focused tests under `tests/gym/envs/managers/`
- Configuration/API documentation for the functor, if a suitable randomization
  reference page exists

## 8. References

- `embodichain/lab/gym/envs/managers/randomization/visual.py`
- `embodichain/lab/sim/material.py`
- `embodichain/lab/sim/objects/rigid_object.py`
- `embodichain/lab/sim/objects/articulation.py`
- `embodichain/lab/gym/envs/managers/event_manager.py`
