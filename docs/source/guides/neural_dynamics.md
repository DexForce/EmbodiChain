# Named dynamics solvers and NeRD

NeRD integrates a robot-specific learned state transition into the DexSim Newton
scene lifecycle. Cartpole uses contact-free dynamics; Ant uses fixed candidate
points projected onto a static plane. A model is bound to its ordered joints,
physical signature, gravity, feature schema and time interval.

This feature requires a DexSim build exporting `RegisteredNewtonSolverCfg`, the
`dexsim.newton_solvers` registry and the Scene functional/snapshot APIs. The
development JIT package and a released source-free package are different artifact
types; use the actual tested build identity rather than assuming every 0.5.x wheel
contains these interfaces.

The repository's existing `dexsim_engine==0.5.1rc1` dependency pin does not identify
the extension-enabled development artifact. Until a matching release and pin are
available, deployment requires the tested local DexSim wheel and this EmbodiChain
checkout installed as a pair. Install the ordinary dependencies first, then install
the local wheel and editable checkout with `--no-deps` so the older pin does not
replace it. Record the wheel hash and check the required APIs before running.
This development procedure does not qualify the existing rc1 distribution.

## Register an implementation

Declare one installed entry point whose name becomes the configuration value:

```toml
[project.entry-points."dexsim.newton_solvers"]
my_solver = "my_package.solver:MySolverProvider"
```

The provider registers builder attributes before model finalization and creates
its extension after model, control and contacts exist. It returns a Newton
`SolverBase` implementation, explicit capabilities, and optional functional
services. Provider instances and mutable histories are per World. Install local
development packages with an editable install; a source file alone does not
register an entry point.

```yaml
physics: newton
physics_config:
  solver_cfg:
    solver_type: my_solver
    options: {}
```

There is no separate direct-provider configuration or imperative registration
route. Names must be unique and cannot override built-in aliases. AutoSolver
retains its scene-aware selection; it does not guess which neural model applies.

## Generate, train and export

The Cartpole and Ant workflow configurations live under
`extensions/embodichain_tasks/configs/tasks/classic_control/<robot>/`.
These `dynamics.*.yaml` files configure commands, not Gym deployments.

```bash
embodichain dynamics generate --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.generate.smoke.yaml
embodichain dynamics train --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.train.smoke.yaml
embodichain dynamics export --checkpoint outputs/dynamics/cartpole/smoke/train/best.pt --output outputs/dynamics/cartpole/smoke/export
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.eval.smoke.yaml
```

Use `embodichain dynamics <command> --help` for path/device/seed overrides.
Configuration-file paths resolve relative to the configuration file; explicit CLI
path overrides resolve relative to the working directory. The `full` profiles
carry a much larger training budget. Smoke profiles establish executable flow,
not model accuracy or official benchmark qualification.

Teacher collection owns an ordinary `SimulationManager` and uses its public
articulation/Scene APIs. Cartpole sets `enable_ground_plane: false`: hiding a
ground mesh would leave collisions active. CPU MuJoCo batches use
`separate_worlds: true`. Record the actual teacher configuration, including
ignored solver-specific material settings, with every dataset.

HDF5 shards contain complete trajectories: T+1 states and T controls/contact
features. Only committed shards appear in the manifest. Sampling windows cannot
cross trajectory/reset/split boundaries. Collection and reading use bounded
buffers, and each data-loader worker owns its file handles. Training, export and
offline evaluation do not initialize DexSim.

Training checkpoints retain optimizer, scheduler, RNG and committed sample cursor
for resume. Exported models contain tensor state, normalization buffers and JSON
metadata with checksums. They can be loaded directly by `solver_type: nerd`.
Legacy official Python-object checkpoints require the separate
`convert-checkpoint --trust-pickle` operation and an explicit deployment manifest.

## Functional dynamics and model gradients

`DifferentiableDynamicsEnv` uses graph-connected state and output hooks. Its
functional state contains physical tensors, per-world history and valid history
lengths. Native state publication is detached and does not own autograd storage.
`FunctionalDynamics` advances these states through SimulationManager; do not also
call `World.update` for the same interval.

Torch network operations preserve gradients in evaluation mode. Newton FK uses
an explicit first-order VJP bridge. Tensor sharing between Warp and Torch alone
does not establish a gradient path. Ant's continuous contact geometry has local
derivatives; contact masks, wrap cuts and quaternion signs remain discrete branch
choices. Physical parameters and contact topology are not differentiable in v1.

Resetting selected environments clears their histories without breaking other
rows' graphs or already-computed terminal rewards. Call `detach_state()` at an
intentional rollout boundary, and update model weights only after backward has
completed. Resident model weights are fixed by default; explicitly request
trainable weights or supply parameter tensors for a functional optimization.

NeRD consumes effort controls and rejects unsupported free-base/external forces.
It provides no physical contact impulse/friction-force report. Its trained time
interval is validated; changing `dt` cannot be implemented by scaling a predicted
state delta. Both reference exports use 1/60 s and one neural substep.

## Correctness and official accuracy evidence

Implementation parity, training-step parity, matched-budget training accuracy,
and deployment rollout accuracy are separate gates. A completed short training
run or a finite gradient is not an official accuracy result.

Comparisons pin source commits, assets, data, initialization, batch order,
normalization statistics, optimizer settings and checkpoint-selection criteria.
The official trainer adapter executes original methods in a separate process.
Current-Newton teacher data and official-Warp teacher data occupy separate rows
of the matched-data matrix. Do not attribute teacher changes to the model port.

The historical importers and current Newton produce different masses, inertias
and materials even from identical asset bytes. Official checkpoint conversion
does not prove deployment compatibility. Preserve the historical physical profile
or retrain for the current teacher. The reference data adapter also records an
explicit repair of the official sampler's uninitialized gravity components;
raw arrays remain available and corrected inputs are labeled in the report.

Formal reports require five training seeds, matched declared update budgets,
fixed evaluation trajectories and error-versus-time metrics. Test data never
selects checkpoints. Missing experiments remain explicitly pending. The complete
acceptance contract is maintained in the repository design document
`design/neural-dynamics-solvers.md`.

## Historical Warp comparison

Use `embodichain.learning.dynamics.reference_rollout` in a fresh process to
evaluate either trainer's portable weights with the original environment and
Transformer integrator. Cartpole requires Warp 1.8.0; Ant requires Warp 1.5.1.
Keep these dependencies in separate directories rather than replacing the
installed Newton/Warp runtime. The source checkout must match the pinned main
commit checked by the reference tools.

```bash
PYTHONPATH=/path/to/isolated-warp:/path/to/pinned-nerd-source:$PWD \
  python -m embodichain.learning.dynamics.reference_rollout \
  --spec historical-rollout.json --trust-external-code
```

The JSON spec provides `source`, `model`, `robot`, `output`, `device`, `num_envs`,
`trajectories`, `horizons`, `seed`, `motion_mode` and `initial_states`.
`fixed_policy` currently accepts `policy: {"kind": "pd"}`. Teacher and learned
simulators evaluate the policy on their own states. CPU execution uses a recorded
stream compatibility adapter while retaining original simulation/network math.

Historical physical parameters must match the model's recorded training source.
Older manifests without measured gravity remain pending; do not add a new label
to make them pass. Original/portable network equivalence, teacher-content hashes,
source/trace/model identities, error curves and every failed trajectory are
retained. This path does not make historical weights compatible with the current
Newton asset import.

The training comparison is explicitly a controlled matched-data experiment.
Shared initialization, statistics, sample order and validation selection permit
port comparisons; they do not reproduce every upstream paper-recipe detail.
Full-profile gates inspect the actual immutable model's feature order, FP32
weights, architecture and optimizer settings. Short runs remain short-run
evidence even when all paired metrics pass their numerical thresholds.
