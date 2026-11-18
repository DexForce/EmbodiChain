# Dual-arm RGB sorting acceptance

This scene-first fixture reuses the original RGB block-sorting program unchanged:
right arm moves red, left arm moves blue, green stays stationary. It selects the
existing dual-UR5 profile and open-loop dual-arm execution policy. The two arms
act sequentially; this case does not demonstrate simultaneous manipulation.

Default physics, sensors disabled, B=1. The supported primitive tabletop has its
top at 0.835 m, with the 5 cm blocks resting at 0.86 m. Red/blue nominal poses
are the midpoints of the original reset domains, with the same ±2.5 cm X / ±3 cm
Y ranges. Green is
a fixed reference at its original domain midpoint. Geometry/robot topology stays
fixed. Both source strategies share the same layout and task-path screens.

All original lift (5 cm), XY placement (3.5 / 3 cm), and wait-stable validators
remain unchanged. Independent measured acceptance additionally requires both
objects at table height, both hands released, green within 5 mm of its initial
reference, and stable terminal speeds. Coverage records both manipulated blocks.

See [the comparison guide](../../../../../../../../../scripts/benchmark/expansion/README.scene_first.md).
