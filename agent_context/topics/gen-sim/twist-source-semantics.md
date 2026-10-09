# Twist Source Semantics

`task_engine/_task_program/twist_semantics.py` owns the static source-role contract.
`twist_binding.py` measures the joint angle and grip dimensions from those roles;
`twist_geometry.py` selects the exact declared collider path. No asset SHA, task ID
or mesh-name vocabulary grants capability.

## Declaration

An asset may provide an adjacent `device.twist.json` for `device.usdc`, or its
USD default prim may author custom data under `gen_sim:twistControl`.
The sidecar has exactly these fields:

```json
{
  "schema": "gen_sim.twist-control/v1",
  "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "joint": "/Device/axis",
  "grip": "/Device/moving/contact_surface",
  "pointer": "/Device/moving/indicator",
  "settings": {
    "0": ["/Device/fixed/mark_zero"],
    "73": ["/Device/fixed/mark_target"]
  }
}
```

The example source digest is a placeholder for the actual USD bytes. Embedded
data omits this self-referential digest; its source file hash covers the
declaration. Two present declarations must agree. Setting IDs identify declared
labels, not radians, degrees or a default ordinal-to-angle formula. Multiple
meshes may form one label. No declaration field may supply a scale, target qpos,
control parameter or task-specific policy.

The protocol declares one explicitly selected control per asset deployment.
Multiple joints are not resolved by choosing the first; the joint path is
explicit. Unannotated assets do not receive inferred grip/pointer/label roles.
Future perception may produce reviewed declarations, but is not implemented by
this loader.

## Qualification

The source must be a self-contained metre-authored USD. The selected enabled
revolute joint must connect distinct enabled rigid bodies under one articulation
root. Grip is an enabled moving-body collider, pointer is a distinct moving-body
mesh, and every label mesh belongs to the fixed parent. Paths are absolute and
canonical; label parts cannot be duplicated across setting IDs. Source geometry
and transforms must be finite and static. Collider topology, native name
uniqueness, joint limits, and reachability remain separate checks in the consumers.

Angles come from measured pointer/label directions and the joint axis. Grip
width/depth come from the declared collider. Unknown IDs, missing roles,
conflicts, source drift and invalid ownership reject rather than fall back to a
whole render-link union or a guessed setting.

## Integrity And Delivery

The binding carries both `source_sha256` and `calibration_sha256`, the normalized
role-selection digest. Sidecar source SHA is only a stale-data check, never a
registry key. The resolver uses fresh anonymous USD layers and checks source
and declaration bytes again after qualification. Its raw sidecar snapshot is
the runtime guard baseline; a later read cannot replace that evidence.

Scene source fingerprints include sidecars. Added, removed or edited sidecars
invalidate prepared source fingerprints, and geometry guards detect changes
after assembly. A static binding/geometry check is not native cooked-shape or
physical task qualification.

Sidecars are asset data and must travel with their USD file; copying a USD alone
does not deliver the declaration. Local `gym_project/` assets are not wheel
resources and may be Git-ignored. Embedded declarations are an alternative for
self-contained distribution. Regenerate E8 bundles for lowerer revision 14 and
adapter contract `gen_sim.task_program/2620929c/v9`; prior host contracts and
serialized bindings are intentionally not reused.

An opted-in deployment mass copy additionally pins `mass_lineage_sha256`.
`twist_mass_source.py` qualifies both pristine and runtime USD bytes, sealed roles
and the complete authored-layer transformation. The lineage is not an origin
marker or a capability registry. Geometry guards retain both source snapshots;
native mass verification derives expectations from the pristine copy, not from
an already compensated runtime USD. Ordinary source bindings leave this field
empty. Explicit `twist_mass_source` selection uses the GenSim deployment copy;
bundle generation also selects it automatically when the Lab articulation
configuration lacks `body_scale_mass_policy`. The explicit flag defaults to
false, but does not disable that capability-based fallback. The pristine asset
is never rewritten, and removing the Lab mass-policy field from the deployment
prevents a second conversion.

## Runtime Ownership

GenSim's `TwistFeedbackSession` extends the public `ExecutionSession.tick` and
`revise_current` APIs for one opted-in, effectless, empty-hand Twist invocation
in one environment. The original execution runner, semantic executor, Gym
bridge and command transports remain the execution owners; no private executor
is copied or globally replaced. Default and non-E8 calls retain the original
session path.

The session reads the public phase view only for scheduling identity, never to
assert a held relation. Emitted command snapshots are not acknowledgements:
the unchanged runner stops after rejected transport, and the feedback policy
requires exact native command-target readback before using a continuation seed.
Controller/context snapshots preserve ownership. Revisions consume the first
plan's remaining logical deadline and the existing wall budget; they do not
renew either. Effects, phase gates, held guards and parallel execution are not
supported by this specialization. Correlated results reject rather than being
reused against a new plan generation. These lifecycle checks do not certify
dual-pad grasp or precise printed-setting convergence.
