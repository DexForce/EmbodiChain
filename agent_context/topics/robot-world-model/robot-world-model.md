# Robot world model

## Ownership and entry points

`embodichain_models.dynamics.rwm` is the canonical model package. Its `model.py`
owns fixed-period frame-local state increments; `robot_state.py` owns the
RWM state representation and SO(3) transforms. They have no simulation or example
dependency. Shared MLPs/normalizers stay in core `learning.models`; optimizer
updates, run artifacts and recovery stay in [SL](../supervised-learning/supervised-learning.md).

`supervised.py` owns the autoregressive objective and replay-window adaptation.
`predictor.py` retains graph-connected prediction state. `sl_recipe.py` exports
`build_rwm_trainer` for `train-sl`; the same symbol is exported from the model
family. `examples/learning/rwm/` owns controlled corpus generation and acceptance
through `embodichain run-example learning/rwm`; private support is not a model API.
The [RWM guide](../../../docs/source/features/neural_dynamics/rwm.md) owns the
user-facing principles, tensor layout, prediction and replay examples.

## State, graph and data boundaries

Canonical state is position, row-major body-to-world rotation, world linear and
angular velocity, then ordered joint positions and velocities. Increments use
the current body frame with right-multiplied exponential rotation coordinates.
Fixed control period, action semantics and joint order enter model/checkpoint
identity. Normalization changes only from explicitly selected measured training
transitions and is frozen during prediction differentiation.

The objective retains the full multi-step graph and excludes invalid suffix
padding before model/error arithmetic. Predictor reset allocates replacements,
preserving outstanding results and gradients through unselected rows. Freeze
model parameters and normalization for action/state differentiation, and finish
dependent backward work before changing weights.

`replay.py` owns fixed-byte-budget device storage. Append copies detached,
owned physical transitions before reset. Windows/history respect episode
boundaries and ring overwrite; sampling returns owned tensors. Acquisition,
reset and optional replay refill remain caller-owned.

## Validation

`tests/models/dynamics/rwm/` covers state transforms, replay ownership/boundaries,
NaN padding, state/action gradients, model identity and exact offline recovery.
The controlled three-seed fitting protocol and its unchanged acceptance gate
live in `examples/learning/rwm/README.md`; historical reports retain their
recorded source/data hashes. These runs qualify fitting/interfaces, not robot
policy performance.
