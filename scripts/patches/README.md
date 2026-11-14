# DexSim assembly capabilities

Current main selects DexSim 0.6.0rc1 on Python 3.12. The installed package
exposes raw contact and prepared-variant descriptor fields; direct native
compound, raw-contact and compatible-variant smoke checks passed without these
historical patches. The notes below retain the earlier source profiles for
reproducible old runs and do not replace current dependency selection.

One current dependency limitation remains: static compound USD import drops
child colliders. EmbodiChain rejects this at the import boundary and recommends
direct manifest construction. The static USD round-trip test retains its full
collider-count oracle with a strict expected failure for exactly 0.6.0rc1.

## Historical runtime archive

Earlier source patches and independent CPU version-comparison material are
preserved in the immutable [assembly runtime archive](https://github.com/DexForce/EmbodiChain/tree/e15c945a8976b06b447f933400d12e5bd936dd3a/scripts).
That snapshot includes the three DexSim descriptor/continuous-joint patches,
the Newton 1.6.0 CPU PGS patch and its source checksums, and the independent
reference worker/requirements under `scripts/benchmark/assembly/`.
They reconstruct historical measurements and are excluded from current wheels.
Use an SDK exposing the requested public descriptor capabilities for current
runs; the historical patches are tied to their recorded source revisions.

The nominal Rizon robot preparation tool belongs to the robot deployment
layer (#984), which also provides its source asset registration. When cascading
this cleanup into that layer, explicitly reintroduce its preserved helper,
robot-frame regression and API page after the merge: Git otherwise propagates
the lower layer's deletion. Their originals remain in the same immutable
snapshot linked above.

## Optional MuJoCo Warp EPA horizon capacity

`mujoco_warp_epa_horizon.patch` (MuJoCo Warp **3.12.0**) and
`mujoco_warp_epa_horizon_3_15.patch` (**3.15.0**) increase the
`MJ_MAX_EPAHORIZON` allocation from 24 to 128 integers per CCD candidate.
The convex and flex collision allocators read this Python constant; EPA's
algorithm, iteration limits, geometry, contact properties and overflow flags
are unchanged. Exhausting the original horizon emits `EPA_HORIZON` (bit 8)
and can discard collision work, so a run that emitted this warning cannot
qualify as an overflow-free training run.

Create a new isolated overlay using the same Python environment as training:

```bash
python scripts/patches/apply_mujoco_warp_epa_horizon.py --output /tmp/embodichain-mjw-epa128
```

The helper selects the matching version profile and checks exact SHA-256 values
of `types.py`, `collision_convex.py`, `collision_flex.py` and `collision_gjk.py`
before copying anything. It applies the included patch to the copy and verifies
that only the constant changed. It writes `provenance.json`, refuses existing
outputs, and never edits the installed dependency. The original third-party
license headers remain in the copied sources. Prepend the printed overlay root
to the existing `PYTHONPATH`, retaining any DexSim overlay, then start a fresh
process. A process that already imported MuJoCo Warp retains its old constant.

The additional convex scratch allocation is
`naccdmax × (128 - 24) × sizeof(int32) = naccdmax × 416 bytes`.
At `naccdmax=4096×256`, that is **416 MiB**; at `12288×256`, **1248 MiB**.
Flex allocation adds its corresponding `capacity × 416 bytes` when used.
This is an explicit memory tradeoff, not a solver accuracy or throughput claim.
Re-run strict native overflow checks, the triggering contact scene, grip/reset
checks and policy evaluation before qualifying a new training profile. The
patch is optional; ordinary installation never applies it automatically.
