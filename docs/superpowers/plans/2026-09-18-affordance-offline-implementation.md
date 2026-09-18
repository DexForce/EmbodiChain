# Offline Affordance collection implementation

The opt-in `ExpertTrajectoryCfg.affordance_augmentation` policy connects legal
Affordance variation to the existing Task Program expert and synchronous
LeRobot recorder. It keeps geometry sampling in Affordances and gives the Gym
host ownership of bounded attempts, measured acceptance, full-batch reset,
selected-row persistence, and run reporting.

## Usage

From the repository root with the simulator/runtime dependencies installed:

```bash
python -m embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.augmentation.yaml --headless
```

This reference uses seed 42, four physical environments/four branches, and a
quota of five confirmed episodes. The final vector batch may commit only one
accepted row. `branches: 1` keeps nominal selection; otherwise the branch count
must equal the physical environment count. `max_batches` bounds total rollout
attempts and `demo_max_attempts` bounds retries of one collection batch.

## Acceptance and outputs

The supported profile is sequential Pick/Place: every touched object ends each
segment at an absolute Place target, with measured target proximity, placing
gripper release, and object stability. Completed bridge execution and current
measured evidence are both required; projected intermediate effects are not
physical grasp certification.

Exactly one synchronous LeRobot save sink is required. Only receipt-confirmed
rows advance the quota after configured outputs finish. Full-batch reset is
preserved; optional trajectory auto-save selects the accepted rows. Row metadata
retains run/batch/attempt, physical row, group/branch, episode/commit identities,
config/source hashes, program identity, and pool-local sampling choices. The
dataset directory receives `affordance_collection_<run_id>.json` with attempts,
acceptance, receipts, and final status.

Each accepted LeRobot episode closes its writers and reloads locally before
returning a receipt. Subsequent episodes use the public resumed-recording path,
preserving existing files. This adds one dataset reload per episode. A failure
after a save begins blocks further writes because persistence may be partial.

## Limits and qualification

No trajectory coverage selection, `GenerationSession`, online refill, async
receipt path, fragment collection, arbitrary measured-call qualification, or
parallel segments are included. Scene-slot-independent RNG is not supported;
upstream candidate pools still depend on physical rows and RNG. Candidate IDs
are pool-local. One-process commit deduplication and sticky post-commit errors
do not provide crash recovery; dataset and optional trajectory outputs do not
form an atomic transaction.

This note describes source/configuration contracts. It does not certify a live
physical run, episode yield, geometric diversity, or persisted multimodal output.
Those claims require separate execution and artifact inspection.

The local simulation smoke run was attempted but failed during startup: the
available DexSim build lacks `WorldConfig.log_startup_info`, required by this
branch. The real LeRobot tests run independently of the simulator and verify
that multiple successive receipts can each be reloaded before recorder shutdown.
