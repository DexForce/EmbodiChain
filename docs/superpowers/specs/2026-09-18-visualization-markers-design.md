# Visualization marker groups (issue #129)

Historical first-stage spec. The current batch/attachment contract supersedes
the single-arena design below; see
[batch delivery plan](../plans/2026-09-18-markers-batch-attachment.md).

The user approved proceeding with the preceding source-backed design. Implement a
render-only marker group abstraction shared by DexSim and Viser, retaining the
legacy axis API. No physics stepping, scene finalization, or rigid-body creation
is permitted during marker publication.

## Contract

`MarkerPrototypeCfg(shape, scale=(1,1,1), color=(1,0,0,1), vertices=None,
faces=None)` describes a reusable unit geometry. Built-ins are box, sphere,
cylinder, capsule, cone, arrow and frame; mesh accepts explicit triangle arrays.
Box/sphere/cylinder/cone have unit bounding dimensions; arrow points along +X;
frame has unit positive axes. Prototype scale is multiplied by instance scale.

`MarkerGroupCfg(name, prototypes, arena_index=-1)` declares one group. -1 means
world coordinates; other values select one arena and interpret translations in
that arena's local coordinates. Groups have stable unique names (duplicate names
raise). Group.update accepts translations (N,3), orientations_xyzw (N,4), scales
(N,3), prototype_indices (N,), colors (N,4), and visible (N,). Omitted arrays
retain their values for unchanged counts; count changes require translations and
reset omitted arrays to documented defaults. Validate all arrays before mutation.
Group.set_visibility, clear, remove do not affect physics. Snapshots own copies.

A `MeshMarkerOverlay` protocol value carries overlay_id, vertices, faces, position,
wxyz, scale, RGBA color, visible, and optional env_id. SceneOverlays.meshes holds
these. Frame prototypes expand into three colored meshes. Viser reuses handles
for pose/color changes, replaces only geometry changes, respects environment
visibility, and removes stale handles. Native adapter consumes the same meshes.

DexSim debug overlay is the preferred native route, with no shadows and no
physics. Expose a supported native API if needed; do not silently render debug
markers into sensor images on older engines. Report an actionable unsupported
capability error. Viser remains usable without native marker support. Existing
legacy axes and existing point/trajectory overlays remain compatible.

## Scope and checks

First stage implements groups, primitives, RGBA, compatibility, native and Viser
adapters, example, focused behavioral tests, and context/API documentation.
Point clouds and polylines retain existing overlay APIs. Streaming optimization,
new GPU instancing machinery, arbitrary USD import, and text are deferred.
Checks: validation and rollback, nontrivial rotations, arena offsets once,
repeated updates without leaks, per-instance/group visibility, count changes,
clear/remove, offscreen overlay defaults, and unchanged simulation clocks.
