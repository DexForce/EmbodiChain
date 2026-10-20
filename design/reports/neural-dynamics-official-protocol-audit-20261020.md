# NeRD official protocol audit — 2026-10-20

Reference: unmodified `NVlabs/neural-robot-dynamics` commit
`370e89d2b8ce68aa3578aa7812859c681b7502bc`. This audit distinguishes the official
architecture/optimizer-budget profile from controlled matched-training experiments
and literal published-result reproduction. Disclosed differences do not waive the
required official comparison, full budget, five seeds or rollout gates.

## Effective official model profile

`train/cfg/{Cartpole,Ant}/transformer.yaml` and the saved pretrained-model configs
agree on these numerical architecture fields:

| Field | Cartpole | Ant |
|---|---|---|
| Ordered inputs (canonical names) | `states_embedding:4`, `joint_f:2` | `states_embedding:29`, `contact_normals:75`, `contact_points_1:75`, `contact_depths:25`, `joint_f:8`, `gravity_dir:3` |
| Output dimension | 4 | 29 |
| Transformer embedding | 192 | 384 |
| Layers / heads / block size | 6 / 12 / 32 | 6 / 12 / 32 |
| History and training window | 10 / 10 | 10 / 10 |
| Prediction-head hidden dimensions | `[64]` | `[64]` |
| Transformer bias / dropout | false / 0.0 | false / 0.0 |
| Normalize input / output | true / true | true / true |
| Effective output tanh | false | false |
| State frame | world scalar joint coordinates | body, anchor at every step |
| Embedding / prediction / orientation | identical / relative / quaternion | identical / relative / quaternion |

The original input name is `joint_acts`; `joint_f` is the explicit transport rename.
Feature **order** is part of the network contract, not just names and widths.
Input encoders are identity (no hidden layers); the head is Linear/ReLU with no
head LayerNorm. Transformer MLP uses GELU and a 4× expansion. These choices are
fixed in the implementation rather than separately configurable in the migrated
model. Transformer `bias:false` does not mean the prediction head has no bias.
The source YAML nests `output_tanh:false` under `network.model`, while the source
model reads a top-level network flag with a false default; effective behavior is
false in either case.

A full-profile gate should inspect **hash-verified actual model artifacts** for
all these fields. `comparison._json(sort_keys=True)` sorts the plan's nested
`input_dims` map, so plan dictionary iteration is not authoritative feature order.
The initial-model export's `model_config.input_dims` preserves actual order.
Actual floating weights/normalizers must be FP32 for the official numeric profile;
JSON architecture fields alone do not prove dtype.

## Effective official optimizer and budget

| Contract | Expected |
|---|---|
| Algorithm / optimizer | SequenceModelTrainer / Adam |
| Batch size | 512 full windows |
| Adam defaults | betas `(0.9,0.999)`, eps `1e-8`, weight decay `0`, amsgrad false |
| Gradient clipping | enabled, norm limit `1.0` |
| Updates per minibatch | one backward and one Adam step; no accumulation/AMP |
| Learning-rate endpoints | `1e-3`, `1e-4`; linear epoch schedule |
| Source epochs / updates per trained epoch | 1000 / 5000 |
| Actual optimizer updates | 4,995,000 (epoch zero evaluates without training) |
| Migrated schedule interval / offset | 5000 / 1 |
| Validation timing / maximum batches | every 5000 updates / 100 batches |
| Loss | mean MSE after dividing prediction and target by `sqrt(output_var + 1e-5)` |

For zero-based optimizer step `s`, the matching rate is
`r=(floor(s/5000)+1)/1000`, `lr=1e-3*(1-r)+1e-4*r`. The first actual update uses
**0.0009991** and the final update uses **0.0001009**; the loop never trains at the
nominal epoch-1000 endpoint `1e-4`.

Current full workflow YAMLs match the translated architecture and numerical
optimizer/budget fields above. `output_tanh:false` is supplied by the model config
default. Enforce the fields, rather than accepting an arbitrary architecture or
zero learning rate merely because it runs 4,995,000 updates. Checkpoint frequency,
worker count and device name are execution choices, not architecture gates. Five
predeclared paired seeds are a project qualification requirement; the saved
supervised source checkpoint config itself records seed zero.

## Controlled shared-protocol differences

These are currently explicit adaptations, not equivalent implementations of every
outer-loop behavior in `VanillaTrainer.train()`:

| Aspect | Pinned original implementation | Current matched adapter |
|---|---|---|
| Training sample order | DataLoader shuffle, `drop_last=True`; iterator behavior governs permutations | Shared explicit batch-index artifact, generated with replacement by default |
| Iterator exhaustion | `one_epoch` rebinds an exhausted iterator locally; outer iterator is not replaced | Durable completed-update cursor and exact shared index rows |
| Normalizer weighting | Overlapping full windows, batches `max(512,B)`, final partial batch dropped | Overlapping full windows including final partial batch unless exact shared stats supplied |
| Normalizer estimator | RunningMeanStd FP32 with initial count `1e-4`, mean zero, variance one | Streaming FP64 population moments, then FP32 frozen buffers |
| Validation order | Shuffled loader, full batches only, up to 100 batches, persistent/exhaustion behavior | Same first up-to-100 validation batches for both trainers, partial final batch included |
| Initial checkpoint candidacy | Epoch-zero validation may select untrained initialization as best | First selection after a trained validation interval |
| Validation collections | Saved checkpoint configs contain separate experimental and passive datasets, each with its own best checkpoint | One canonical validation split and one minimum normalized-MSE selector |

For strict sample-order reproduction, capture the original loader's actual emitted
indices; an approximate "epoch shuffle" is insufficient because of its exhaustion
behavior. Shared mean/variance artifacts can make both trainers use exactly the
same normalization, but their origin and estimator must be recorded. The normalizer
count prior `1e-4` is distinct from the normalization denominator epsilon `1e-5`.

These controlled experiments can establish migration correctness/noninferiority
under identical data and optimizer conditions. Matching architecture and update
budget alone does **not** establish literal original validation/statistics/sampling
reproduction or equivalence to published paper numbers. Further protocol work or
explicit evidence remains necessary for those claims; no data-size floor or
alternate acceptance threshold is introduced by this audit.

Source recipe identity must also be explicit: current `train/cfg` paths name 1M
training transitions, while saved pretrained configs name 5M for Cartpole and 10M
for Ant, plus experimental/passive validation files. These labels are separate
from the required immutable dataset fingerprints and do not justify assuming a
single universal "official dataset".

## New source-generation provenance

New calls to `generate_official_dataset` emit
`physics_contract_version: official-warp-physical-v2` and preserve:

- `source_physical_arrays.gravity`: the actual original model gravity vector in
  the source world frame.
- `teacher_origin.gravity`, `gravity_magnitude`, `up_axis` and contract version.
- `source_generator_files`: SHA256 identities for the original sampler, common
  sampling definitions, environment/contact code and robot-specific environment.
- `source_generator_sha256`: SHA256 of the canonical sorted generator-file map.
- `generation_adapter_sha256`: the exact EmbodiChain reference-wrapper file hash.

The source vector remains Y-up even when canonical state transport rotates to Z-up.
The measured original environment magnitude is 9.81: its explicit
`Environment.gravity=-9.81` overrides the Warp Model fallback of 9.80665. This is
a measured original-runtime value, not an inferred Newton setting. Actual
CPU-generation tests inspect the vector, magnitude, up-axis,
file hashes and their preservation in canonical metadata.

Existing smoke datasets/checkpoints are **not rewritten or relabeled**. Old
artifacts lacking measured gravity retain that provenance gap; they cannot acquire
a v2 physical contract by changing a model manifest annotation.
