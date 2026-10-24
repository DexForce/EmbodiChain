# NeRD reference robot assets

`cartpole_single.urdf` and `ant.xml` are unmodified copies from
NVlabs/neural-robot-dynamics, commit
`476dc67fccc6a99be293bf611f897b1c06f13e30`, under
`envs/newton_envs/assets/`. The upstream repository license is reproduced in
`LICENSE.md` (Apache License 2.0; NVIDIA Corporation and affiliates, 2025).

Source: https://github.com/NVlabs/neural-robot-dynamics/tree/476dc67fccc6a99be293bf611f897b1c06f13e30/envs/newton_envs/assets

EmbodiChain translates these assets using the installed Newton public importer,
then passes their physical descriptors through DexSim Spawn. This is a current
Newton teacher profile, not a claim to reproduce legacy Warp 1.5.1 physics.
Imported motor actuator declarations and XML solver options are not installed
into the Spawn scene: the host supplies actual generalized forces and declares
its own solver configuration. Masses, inertia, COMs, joint limits, passive joint
parameters and collision geometry/material parameters are retained.
