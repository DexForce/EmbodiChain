# Task 3 Report

Implemented targeted visual material randomization in `visual.py`.

## Changes

- Added environment-id normalization and texture selection modes (`random`, `without_replacement`, `fixed`, `cycle`).
- Extended `randomize_visual_material.__call__` with `texture_sampling`, `texture_indices`, and `texture_scope`.
- Material instances are retrieved only for selected environment IDs, fixing partial-reset behavior.
- Texture/property plans are applied per selected environment; articulation links share the per-instance selection.
- Metallic, roughness, and IOR properties now apply in both texture and generated-color branches.
- Removed unsafe runtime `clean_materials()` invocation.

## Test Repair

Reworked the integration fakes to match the functor's keyword-based helper API
and isolated class-symbol monkeypatches per test. The tests now exercise actual
`__call__` dispatch for partial resets, fixed global environment mappings,
articulation per-instance link reuse, and the nonempty texture branch.

Final verification:

- `pytest tests/gym/envs/managers/test_randomize_visual_material.py -q`: 7 passed.
