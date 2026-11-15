# Native physics deployment adapters

`embodichain_solvers` ships in the main `embodichain` distribution. It shares
the existing version/dependencies and `dexsim.newton_solvers` discovery
mechanism. Importing the package does not import Torch, models or the SDK.
Core simulation never imports a concrete adapter.

`physics.nerd.provider` owns the `nerd` Provider and native state/control
binding. Model prediction lives in `embodichain_models.dynamics.nerd`:
`NeRDTransition` combines its injected `NeRDGeometry`, codec, history and
network. Native inference and the optional SDK functional service delegate to
that same transition; only explicit native publication detaches history.

The independent learning environment calls the transition directly. Its
reference factory may reuse the Newton geometry binding with an FK-only model;
it does not construct a native World, Scene or solver. The private Newton FK
gradient adapter lives in `embodichain.lab.sim.diff._newton_fk`, preserving
native layout/velocity conventions and independently owned call storage.
Reusable tensor FK/IK APIs remain in `compute.kinematics` and `learning.layers`.

Deployment accepts the existing direct `ProviderNewtonSolverCfg` or the `nerd`
registered name with `options: {checkpoint: ...}`. It validates the recorded
joint order, COM/velocity convention, contact layout, physical signature, gravity
and prediction timestep. Only homogeneous reference Cartpole/Ant domains are
supported; a solver entry point does not establish general physical coupling.

Refresh editable mappings after adding the extension. Checkpoints/reference
assets remain outside Git and the wheel; preparation and task/benchmark
workflows belong to `examples/learning/nerd`.

`physics.worldparticle.provider` supplies the optional `worldparticle` name.
The independent `WorldParticleModel` owns features, topology, frozen network
and coordinate conversion. The Provider validates native scene, particle count,
dt, gravity and supported capabilities; callers must preserve authored particle
order/topology. It delegates read/force/write/reset
and stream ownership to the SDK's `ParticleInferenceSolver` (SDK 0.6.0rc2).
Main's SDK 0.6.0rc2 pin provides this facility. Older SDKs missing the facility
fail before network loading; deployment also needs qualified model operators.
No core particle inference solver or second model transition is introduced.
WorldParticle currently supports one CUDA particle world with static model
boundaries, without functional gradients, dynamic rigid coupling or CUDA Graph
capture. Its optional operator runtime is documented in the example README;
the main distribution's Torch/SDK pins remain unchanged.
