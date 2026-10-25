# DexSim assembly capabilities

These patches extend the public Spawn boundary required by assembly assets.
They do not change an installed SDK automatically. Their source baseline is
Dexverse commit `39c817973111a1dea0e716cf7efb45594c134099`.

Apply from that Dexverse checkout, then build/package DexSim using its normal
build process. The three DexSim patches can be applied together; they do not
modify Newton. The separate optional Newton patch is described below.

```bash
git apply --check /path/to/EmbodiChain/scripts/patches/dexsim_assembly_contact.patch
git apply --check /path/to/EmbodiChain/scripts/patches/dexsim_assembly_variants.patch
git apply --check /path/to/EmbodiChain/scripts/patches/dexsim_continuous_joint.patch
git apply /path/to/EmbodiChain/scripts/patches/dexsim_assembly_contact.patch
git apply /path/to/EmbodiChain/scripts/patches/dexsim_assembly_variants.patch
git apply /path/to/EmbodiChain/scripts/patches/dexsim_continuous_joint.patch
```

`dexsim_assembly_contact.patch` adds `NewtonCollisionDesc.solref`, `solimp`, and
`solmix`, forwarding them through Newton's registered MuJoCo custom attributes.
Raw `solref` sets Newton's raw mode and takes precedence over shape `ke/kd`.
This preserves source contact impedance rather than inferring it from contact
stiffness. EmbodiChain validates the parameters and rejects unsupported SDK
builds before materialization.

`dexsim_assembly_variants.patch` adds `ObjectDesc.env_variants`, resolves each
replacement before copying/grouping scene descriptions, and handles Newton
topology rebuilds. It keeps one path per logical asset in every arena and
retains native arena collision isolation. It does not copy unused library
assets into environments. EmbodiChain exposes this through
`RigidObjectCfg.env_variants`; entries are complete configurations, including
initial poses and physical properties. Missing entries use the parent config.
Nested variants, source-USD variants, and the Default backend are rejected.

Newton/MuJoCo Warp still requires homogeneous world topology: body, joint,
collision-shape, and constraint counts/types must match. The patch supports
different compatible primitive dimensions and mass properties in that mode;
it does **not** enable arbitrary convex-piece counts in MuJoCo Warp. Newton's
semi-implicit solver was verified with different piece counts. Arbitrary-count
MuJoCo Warp training requires grouped solver execution or additional native
solver support. Hidden collision padding is not used.

`dexsim_continuous_joint.patch` preserves URDF continuous-hinge limits as
`(-inf, +inf)` when translating native Skeleton metadata. It also routes
continuous joints through Newton's revolute DOF configuration, preventing the
native parser's placeholder `(0, 0)` range from becoming a locked joint. Its
native regression checks MuJoCo's `jnt_limited=False`, infinite source descriptor
limits, and a valid pose beyond one complete turn. Newton internally converts
infinities to its documented unlimited `MAXVAL` sentinel:

```bash
python -m pytest tests/sim/objects/test_articulation.py::test_native_continuous_urdf_joint_preserves_unlimited_rotation --run-gpu -q
```

Validation used an isolated overlay of DexSim `0.5.1rc1` Python sources with the
installed binary and Newton runtime. Neither the SDK checkout nor the installed
package was edited. Focused GPU regressions are:

```bash
python -m pytest tests/sim/objects/test_collision_shapes.py --run-gpu -q
```

They verify a peg passes through a compound aperture, raw contact values reach
MuJoCo Warp's actual solver arrays, distinct per-arena masses/poses and shape
counts, selective reset, and topology rebuild. SDK-dependent tests skip when
the required public descriptor field is absent; unit tests verify that the
corresponding production path raises an actionable capability error.

## Optional Newton CPU PGS profile

See [newton_cpu_pgs.md](newton_cpu_pgs.md) for the exact Newton 1.6.0 source
checksum, isolated application instructions, staging and CPU midphase fixes,
and qualified smoke scope. This does not add PGS support to MuJoCo Warp.
