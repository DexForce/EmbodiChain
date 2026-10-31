# PhysX 5.6 GPU link measurements

`5.6-gpu-link-measurements.patch` repairs measurement accounting in the public
PhysX 5.6 source at `2264315594478a9aa0bda3464761a666fa107d76`.
It preserves NVIDIA's source license and changes no interface or solver input.

The GPU velocity-output flush must accumulate `mSolverSpatialDeltaVel`, as the
regular flush already does. Without it, a stationary articulated pad resting
on a floor reports gravity acceleration. Partial impulse propagation must
accumulate `mSolverSpatialImpulse` before clearing each link's scratch impulse,
as full propagation already does. Otherwise some contacts disappear from the
incoming-joint-force measurement when constraints span branches/partitions.
The patch touches only these existing measurement accumulators.

## Build and consume

Use a private checkout/output prefix and the CPU dependency's exact PhysX
version and build configuration. DexSim's `PHYSX_VISUAL_DEBUG=ON` configuration
uses CHECKED PhysX, including the GPU library. A RELEASE GPU library cannot be
substituted for that CHECKED dependency. Keep compiler/runtime compatibility
with the intended consumer; this local validation does not publish a new bundle.

Apply from the pinned PhysX Git root:

```bash
git apply --check /path/to/5.6-gpu-link-measurements.patch
git apply /path/to/5.6-gpu-link-measurements.patch
```

Use the upstream `physx/compiler/public` CMake project with
`PX_GENERATE_GPU_PROJECTS=ON`, `PX_GENERATE_GPU_PROJECTS_ONLY=ON`, the matching
`CMAKE_BUILD_TYPE`, and private `PX_OUTPUT_LIB_DIR`/`PX_OUTPUT_BIN_DIR` values.
Select the compiler, CUDA toolkit, and architectures for that build explicitly.
The validation build used Clang 14, CUDA 12.8 and sm_120; its architecture-only
CMake adjustment is independent of the physics patch. Build target `PhysXGpu`.
For a local runtime comparison, place the matching library under the consumer's
`libPhysXGpu_64.so` filename in a private directory and prepend that directory
to `LD_LIBRARY_PATH`. Verify `/proc/self/maps` and the actual library hash.
Do not replace the shared dependency prefix to test this patch.

Rebuilding/staging the SDK can restore its selected prebuilt GPU dependency.
The retained-body OGBench state force measurement requires the patched artifact;
new Python accessors alone do not repair an unpatched PhysX binary.

## Verification

Run `python/test/engine/test_gpu_link_wrench.py` with the selected SDK and GPU
library. Six cases cover an unloaded fixed pad, free fall, applied torque,
one/two static supports, and two dynamic supports. Child frames are rotated;
forces use newtons and torques newton-metres. Contact impulses/moments are an
independent oracle. The dynamic case uses finite velocity differences for its
small residual acceleration rather than assuming exact equilibrium. Original
measurement code fails the static support case by the pad's own weight and the
dynamic support case by several newtons. The patched build passes all six.

The OGBench expert contact replay checks normal force independently at 24 loaded
steps. Maximum relative error after both repairs is 6.65e-7; 21 previously missing
steps recover. The partial-propagation repair leaves 45-control joint positions,
joint velocities and cube poses bit-identical to the preceding arm. This checks
measurement accounting, not MuJoCo/PhysX contact-law equivalence.

Evidence for the current technical validation is in the owning task's execution
ledger and logs; binaries, build trees and machine-specific paths are not part
of this source patch. Also run affected Scene/batch/constraint/filter/clone tests
and the existing GPU articulation benchmark before changing a dependency bundle.

## GPU PGS mimic compliance

`5.6-gpu-pgs-mimic-compliance.patch` applies to the same PhysX source revision.
Apply it together with the link-measurement patch when building the official
Cube GPU recipe. In PGS, joint positions remain fixed during solver iterations.
Reapplying a compliant impulse without its accumulated CFM term drives the
coupling towards a hard constraint as the iteration count grows.

The patch keeps one scalar impulse accumulator per mimic in the existing private
GPU block, resets it during per-step constraint setup, and includes its CFM
residual in the existing helper. The TGS call path and existing velocity-iteration
behavior remain unchanged. It adds no public API or CPU/GPU interface layout;
GPU-private allocation uses its updated block size. The prebuilt CPU solver is
unchanged, so this patch does not certify CPU PGS mimic compliance.

Two independently forced 0.1 kg-m^2 hinges under opposing 0.1 Nm loads give
50 microradians at frequency 200 rad/s at the backend's default iterations.
The unmodified GPU gives an effectively hard coupling. With the source-mapped
frequency, the corrected GPU gives 2.71495969 microradians, versus the pinned
MuJoCo result 2.71492317. All four 250-step TGS control trajectories are
bit-identical to baseline. The focused regression is
`test_gpu_pgs_mimic_preserves_soft_equilibrium` in
`python/test/spawn/test_articulation_properties.py`.

Earlier regression/benchmark labels claiming a 4/100 iteration sweep were
invalid: the public solver/sleep setters used by those harnesses only warn and
do not apply the values. The duplicate regression and sweep labels were removed.
The registered benchmark now measures 25/256 chains at backend-default iterations;
it does not certify an iteration sweep. Native pre-initialization experiments
and EmbodiChain's configured task have separate iteration-setting paths.

## Optional GPU point-anchor linear drives

`5.6-gpu-d6-point-anchors.patch` adds an opt-in variant of the existing D6 linear
spring rows on the same pinned source. Ordinary D6 drives apply actor 0's force
moment at actor 1's joint origin. The optional variant uses each actor's own
authored origin. It retains the drive axes, gains, targets, row count and the
existing GPU D6 descriptor layout. Drive flag bits 0 and 1 retain their original
acceleration/output-force meanings; the private extension uses bit 2.

The constraint/Arena API exposes `supports_d6_linear_drive_point_anchors()` and
`set_d6_linear_drive_point_anchors(name, enabled)`. The capability query checks
the already-loaded GPU module's scalar export
`PxGetDexSimGpuConstraintCapabilities`; it does not load another solver library.
Enabling on CPU or an older GPU library fails explicitly. Disabling restores
ordinary D6 geometry. Gain updates preserve the explicit selection. Existing
constraint vtables, ownership, Scene-thread entry and default behavior remain
unchanged. The CPU extension is not implemented by this GPU patch.

Build and stage the matching CHECKED GPU library as above, together with the
other two patches when using the Cube recipe. SDK staging can restore the
prebuilt dependency, so inspect the actual provider and capability after staging.
`test_d6_linear_drive_point_anchors_match_spatial_impulse` independently checks
the 12-dimensional body impulse for default and selected geometry.
`test_d6_point_anchors_reject_unsupported_cpu_scene` covers the CPU rejection.
These component tests do not establish whole-task MuJoCo equivalence; source
integration, articulated velocity bias, mimic behavior and contacts have their
own contracts and validation.
