# NeRD reference robot assets

`cartpole_single.urdf` and `ant.xml` are unmodified copies from
NVlabs/neural-robot-dynamics, commit
`476dc67fccc6a99be293bf611f897b1c06f13e30`, under
`envs/newton_envs/assets/`. The upstream repository license is reproduced in
`LICENSE.md` (Apache License 2.0; NVIDIA Corporation and affiliates, 2025).

Source: https://github.com/NVlabs/neural-robot-dynamics/tree/476dc67fccc6a99be293bf611f897b1c06f13e30/envs/newton_envs/assets

EmbodiChain interprets these assets using the installed Newton public importer.
The independent data collector finalizes an analytical Newton model directly;
Scene-based neural solver integration is a separate consumer of the same asset
interpretation. Neither path claims to reproduce legacy Warp 1.5.1 physics.

The host supplies generalized forces and its own solver configuration rather
than installing the source motor actuators. Dataset manifests identify the
importer profile, asset hash, actual parameter hashes and solver versions.
`warp_reference` reconstructs measured legacy mass, inertia and contact-parameter
semantics; it does not select the historical simulator or guarantee identical
trajectories across solver versions.
