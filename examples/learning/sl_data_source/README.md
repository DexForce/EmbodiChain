# Causal data sources and supervised learning

This CPU-only example sends a resident `TensorDataSource` through
`SequenceBatchSource` into the common SL trainer. It needs no simulator, SDK,
dataset download or producer process. Two explicit encoders illustrate the
interface:

```bash
embodichain run-example learning/sl_data_source --encoder bc --output outputs/sl_data_bc
embodichain run-example learning/sl_data_source --encoder transition --output outputs/sl_data_transition
```

The `bc` encoder uses two measured observations as inputs and two current/future
actions as targets. The transition encoder uses measured history and the current
drive as inputs; the next observation is a target only. Both consume the
source's padding/loss masks explicitly. This is a synthetic dimensionless drive
contract and a tiny MLP, not a qualified robot BC policy or NeRD effort recipe.
Recipes must validate real action representations, units, timing and frames;
position commands cannot be relabeled as generalized efforts.

Every encoder is an explicit pure function returning `SupervisedBatch`. Give it
a persisted `encoder_version` that changes with its behavior or captured
configuration. Keep returned tensors in the current owned context or newly
allocated storage. Do not reuse mutable encoding buffers. The adapter reserves
`metadata['sequence_source']` for the schema, frozen window selection, identity
and original validity/loss masks plus provenance; that context omits observations
and actions. Only the encoder selects model inputs and targets.

The example sets `immutable=True`: the resident CPU payload and semantic
metadata must never be mutated, and delivery quotas must stay disabled.
Construction hashes actual content in bounded blocks; exact resume also
revalidates content, schema, window and encoder identity. Reusing a handwritten
dataset ID does not authorize different data. Storage paths, cache files and
read-batch diagnostics do not change semantic identity. Language dictionaries
and non-storage metadata do. Use the same update budget and recipe when resuming
an interrupted run from its latest `last.pt` with `--resume FILE`.

For an already-started live ODS engine, the caller can construct the same bridge:

```python
from embodichain.data_pipeline.datasets import OnlineDataSource, SequenceSpecCfg
from embodichain.learning.sl import SequenceBatchSource

source = OnlineDataSource(started_engine, schema=explicit_schema)
training_source = SequenceBatchSource(
    source,
    SequenceSpecCfg(history_length=2, prediction_horizon=2),
    encoder=encode_measured_inputs_and_targets,
    encoder_version="my-explicit-robot-recipe-v1",
)
# Pass training_source to SupervisedTrainer. The caller still owns the engine.
```

The caller owns start/stop, bounded waits, refill and producer recovery. The
bridge performs one sampling call without retries. Quotas charge successful
delivery, so an encoding or optimizer failure does not refund them. Live buffers
and budgeted resident sources can train and save model/optimizer snapshots, but
`trainer.resume()` rejects them before changing model, optimizer, sampler, RNG
or output admission state. Restoring RNG cannot restore buffer contents or
delivery credits. A fresh run may explicitly initialize model weights from a
trusted snapshot; that is not exact continuation. The same resident bridge also
accepts `LeRobotDataSource` after its caller-owned bounded SDK load.
