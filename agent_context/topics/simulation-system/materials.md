# Visual materials

`embodichain/lab/sim/material.py` owns `VisualMaterialCfg`, material templates,
native instance wrappers and segment assignment helpers. Its private
`_material_desc_from_cfg()` validates and normalizes both runtime creation and
`spawn/descriptors.py` lowering. Keep material type and emission semantics at
this shared boundary rather than duplicating conversion in callers.

## Identity and state

An unnamed configuration has no shared identity. Spawn derives its name from
the object and normalized appearance; the manager names standalone runtime
creations uniquely. Runtime and Spawn share the definition registry. Explicit
UIDs share only equivalent definitions; conflicting properties fail before
native writes. Equivalent aliases borrow current state without resetting it.
Definitions remain reserved for the World lifetime, including across native
unused-material cleanup. A template retains one wrapper per instance UID. Repeated
creation returns that wrapper without resetting properties; explicit
`set_default_properties()` applies a reset intentionally.

Wrappers retain the exact native instance, including asset-parsed and reuse
instances. Every setter uses that handle. Scalar/color reads query native
state; DexSim's color getter supplies display-sRGB RGB, while configuration
and setters accept RGBA. Alpha cannot be verified through that getter.
Segment binding compares native material IDs rather than names.

## Assignment lifetime

Objects own their original per-segment assignments and selective reset
baselines. Before a topology rebuild, `SpawnScene` captures current assignments
of bound facades and replays them during binding. This preserves both deliberate
runtime replacements and completed restores even when retained native
descriptors still name an earlier replacement. A failed replay retains its
snapshot for preparation retry; successful replay and close release it. Removal
discards that object's pending entries before a same-UID replacement can bind.

VisualObject additionally owns pose/visibility replay. Neither material replay
nor rendering advances physics. Native material handles and texture pools must
outlive their bindings and be released before their owning World.

## Randomization boundary and validation

[Visual randomization](../randomization/randomization.md) owns bounded native
texture pools, selected-environment sampling and tier choices. Imported asset
materials remain reset baselines; working instances are separate handles.
For repeated updates, prefer pre-created textures from the owning simulation
over per-call Tensor uploads.

Primary CPU contracts are in `tests/sim/test_material.py`,
`tests/sim/spawn/test_scene.py` and
`tests/gym/envs/managers/test_randomize_visual_material.py`. Native parity,
selective mutation, stable instance capacity and topology replay live in
`tests/sim/test_material_integration.py`. The
[acceptance plan](../../../docs/architecture/visual-material-acceptance.md)
separates these checks from optical image and longer resource qualification.
