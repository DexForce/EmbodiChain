# Contact history and substeps

`ContactSensor.create_history()` registers sensor-owned `ContactHistory` objects
and enables substep sampling automatically. Tasks select body IDs, counterpart
IDs and thresholds; locomotion self-contact thresholds come from task reward
configuration. Unknown counterpart IDs (`-1`) require explicit opt-in and do
not identify ground.

`BaseEnv` starts a control interval, then calls `SimulationManager.update()`
once with an `after_substep` observer. The manager invokes the observer after
each world update, preserving its preparation, profiling and camera-capture
boundary. Observation reads reuse the sampled sensor buffer.

CUDA queries use `fetch_async()` and device-resident counts. Sparse Warp kernels
process valid contact rows, using precomputed sorted environment/actor indices;
no environment-by-contact-by-body tensor is allocated. Per-body force, contact
flags and timing buffers persist between samples. `current_air_time` measures
an unfinished flight; `last_air_time` retains its completed duration on landing.
Selective reset clears only the selected rows.

`dropped_contacts` reports query loss plus separately counted scatter overflow
across the control interval. Query counts stay on the device until this diagnostic
is read. A standalone sensor without histories or observers reports the latest
update. `latest_sample_dropped_contacts` separately reports the latest fetch's
query and scatter loss, including a standalone refresh before an observer's
first control interval. Query output buffers belong to each sensor's own query.
Evidence consumers must reject loss from either diagnostic; use the maximum
rather than adding overlapping counts.
See `tests/sim/sensors/test_contact_history.py`, `test_contact_query_sensor.py`
and `tests/gym/envs/test_substep_sensors.py`.

## Sampling observers

`ContactSensor.add_sampling_observer` attaches a named, instance-owned read-only
observer to this same lifecycle. It receives begin, before-fetch, after-sample and
selected-row reset notifications. The empty default performs no observer copies
or callbacks. Hooks run outside the sensor's internal CUDA capture; observers
inside an external capture fail before their callbacks run. Existing history
sampling and device kernels keep their ownership and ordering.
Removal can supply `expected=observer` to retain a replacement registered under
the same name. Owners that close during an in-flight token snapshot must make
their delayed callbacks no-ops.

`lab/sim/sensors/contact_sampling.py` snapshots and revalidates each request
before acquisition. It validates channel shape/dtype and finite-row masks once,
then privately transfers the fresh channel tensors into evidence batches. Each
batch still owns its environment IDs and all frame alignment checks remain.
Public batch/frame construction and snapshots always revalidate and copy mutable
inputs; this path does not cache validation by tensor/object identity.
The sampler adapts sampled buffers
to the shared `lab/sim/evidence.py` frame/batch types. It retains the validity mask,
native actor IDs, Arena frames and impulses, with explicit unsupported, nonfinite,
saturated or source-drift errors. Native and runtime observation revisions remain
different identities. The provider's output device follows its request.

`lab/sim/sensors/contact_timeline.py` owns bounded per-environment packed
history, driven by an explicit host control clock. Keep it in the environment's
sensor suite so `BaseEnv.step` drives the public observer port. It records no task
verdict and reads no private active-bridge field. Selected resets clear only those
rows; missing/partial intervals and loss remain visible. The contact-row limit is
a source-compatible soft threshold: at most one complete native batch can cross
it; the following sample is unavailable. Reset uses the sensor's index conversion
and aliases, deduplicating rows before advancing generations. Frontends own report
schema wrappers and semantic actor labels. The CPU test
`tests/gym/envs/task_program/test_contact_evidence.py` exercises the actual public
step boundary, frame ownership, native stamps and selective reset behavior.

`lab/sim/sensors/support_observation.py` owns native load-patch acquisition. Its explicit
`SupportObservationRegistry` reuses an exact native pair across bridge consumers;
same-UID native replacements require release and rebind. Registry cleanup detaches
only owned hooks and leaves simulation-owned resources alive. The pure COM patch
calculation lives in `compute/geometry/contact_support.py`.

Support results separate a physical boolean from availability/error masks.
Global acquisition failures clear every row's witness, while selected resets
advance only those row generations. A positive witness survives sleeping contacts
only while both poses and native identities remain qualified. Complete empty
samples do not prove an absence of support over a stability window. Every frame
uses a real acquisition clock; `bind_clock` separately enables revision-1 window
qualification. A legacy observation does not acquire strong metadata merely by
sharing that owner. TaskSpec evaluation remains outside Lab.

Strong frames and metadata use the bound environment clock plus the measured
substep offset. The observer fixes and checks its mapping from the real
simulation acquisition clock when the first control interval begins. Explicit
warmup may create a nonzero offset; unobserved advancement that changes the
mapping invalidates the window. Legacy acquisition keeps the simulation clock
and does not gain strong qualification. These domains must not be mixed in one
evidence window.
