# Articulation geometry boundary

Read this when the request needs these details. [Topic overview](atomic-actions.md).

## Articulation geometry boundary

`sample_initial_articulation_geometry()` adapts domain-neutral articulation
facts into the geometry consumed by link affordances. Callers pass an
`ArticulationGeometryProvider`, the explicitly named initial joint state, and
body scale. The adapter owns Open3D surface sampling, target-link-frame mesh
transforms, nearest prismatic/revolute ancestor geometry, and the private
`ObjectSemantics.geometry` keys.

Every non-empty provider link mesh must expose at least one non-degenerate
triangle surface. The adapter rejects vertices-only or fully degenerate links
before merging so the target-link and whole-articulation clouds cannot disagree
about whether a link contributes geometry.

The adapter preserves the complete articulation cloud and separately samples
all non-target link surfaces. `SlideAffordance`, `PressAffordance`, and
`TwistAffordance` use only that non-target cloud to sign a parent-joint axis.
Missing provenance, no non-target samples within twice the target radius, or an
axial offset within the sampling-error confidence bound is directionally
ambiguous; never fall back to the complete cloud because its independently
sampled target surface is noise, not direction evidence.

The adapter returns `ArticulationAffordanceGeometry`; convert it with
`to_object_geometry()` only when constructing `ObjectSemantics`. Keep random
sampling and semantic geometry keys out of `objects/articulation.py`.
