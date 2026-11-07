# Contact history and substeps

The canonical implementation is `lab/sim/sensors/contact/`: `sensor.py`,
`history.py`, `sampling.py`, `timeline.py` and private `_warp/` adaptation kernels.
Use `sensors.contact` or its implementation submodules. The old flat
`contact_sensor` and `contact_history` modules have been removed. The parent
`sensors` registry still exports `ContactSensor` and its configuration.

`ContactSensor.create_history()` registers sensor-owned `ContactHistory` objects
and enables substep sampling automatically. Tasks select body IDs, counterpart
IDs and thresholds; locomotion self-contact thresholds come from task reward
configuration. Unknown counterpart IDs (`-1`) require explicit opt-in and do
not identify ground.

`BaseEnv` starts a control interval, then calls `SimulationManager.update()`
once with an `after_substep` observer. The manager invokes the observer after
each world update, preserving its preparation, profiling and camera-capture
boundary. Observation reads reuse the sampled sensor buffer.

`ContactSensor` deduplicates raw reads and history reductions separately using
`SimulationManager.physics_step_index` and the query's current actor binding.
Reading a raw sample before its history callback does not consume the history
update. Query metadata is checked even on cache hits so topology rebuilds cannot
retain old contacts. Cached CUDA reads wait for the producing stream. Caller-owned
CUDA graph capture bypasses raw/history deduplication and records the query read
and reductions on the capture stream. Capture invalidates eager cache keys rather
than marking unexecuted work as a completed sample; replay does not execute Python
cache bookkeeping. Callers own replay scheduling and establish stream dependencies before capture
and replay. Capture skips eager producer waits and never records its stream as
a completed producer, including control-interval initialization and reset.

Sensor reset clears only selected rows and invalidates this cache; it does not
reset the physics solver. An explicit `update()` may therefore refresh a valid
settled snapshot without advancing history, as scene preparation requires.
It cannot recompute solved contacts after physical state writes; the backend's
contact-availability rules still apply. Empty selections do not invalidate the
cache. `BaseEnv` retains its ordinary acquisition and substep scheduling.

CUDA queries use `fetch_solver_report_async()` and device-resident counts. Sparse Warp kernels
process valid contact rows, using precomputed sorted environment/actor indices;
no environment-by-contact-by-body tensor is allocated. Per-body force, contact
flags and timing buffers persist between samples. `current_air_time` measures
an unfinished flight; `last_air_time` retains its completed duration on landing.
Selective reset clears only the selected rows. Force reduction uses only
components marked available by the sample's normal/friction validity flags.
Geometry-only rows can count as contact when no force threshold is requested;
they do not create measured force or satisfy a positive force threshold.
Native friction is the tangential impulse on actor 0, while the normal points
toward actor 1: actor-0 force is `(-normal * impulse + friction) / dt` and
actor 1 receives the opposite force.

`dropped_contacts` reports query loss plus separately counted scatter overflow
across the control interval. Query counts stay on the device until this diagnostic
is read. A standalone sensor without histories or sampling observers reports the
latest update. `latest_sample_dropped_contacts` separately reports the latest
query and scatter loss, including a refresh before the observer's first control
interval. Its device counters are updated by the existing CUDA scatter kernel,
including graph replay and empty-contact samples; CPU uses a scalar copy.
Full reset clears both diagnostics; selective reset retains query-wide loss.
`evidence_dropped_contacts` reads their packed counters once and reduces them
with host int64 sums. The standard sampler rejects loss from either diagnostic
and uses their maximum because the counts overlap; custom sources retain the
separate-counter fallback. Query output buffers belong to each sensor's query.
See `tests/sim/sensors/test_contact_history.py`, `test_contact_query_sensor.py`
and `tests/gym/envs/test_substep_sensors.py`.

## Sampling observers

`ContactSensor.add_sampling_observer` attaches a named, instance-owned read-only
observer to the same lifecycle. It receives begin, before-sample, after-sample and
selected-row reset notifications. With no observers, the update path does not
enumerate subscriptions, construct token generators or enter callback loops.
Hooks run outside the sensor's internal CUDA capture; external capture with
observers fails before callbacks run. Observer tokens and after-sample callbacks
run once per physics step alongside history reduction. A raw sample fetched
earlier in the same step is reused; repeated reads do not advance history or
repeat callbacks. Removal can supply `expected=observer` to retain a replacement
registered under the same name. Closed owners make delayed callbacks no-ops.

`lab/sim/sensors/contact/sampling.py` snapshots and revalidates each request
before acquisition. It validates channel shape/dtype and finite-row masks once,
then privately transfers fresh channel tensors into evidence batches. Every
batch owns its environment IDs and frame alignment checks remain complete.
Public batch/frame constructors and snapshots still validate and copy mutable
inputs; validation is never cached by tensor or object identity.

The sampler adapts buffers to the shared `lab/sim/evidence.py` frame/batch types.
It retains validity masks, native actor IDs, Arena frames and impulses, with
explicit unsupported, nonfinite, saturated or source-drift errors. Native and
runtime observation revisions remain different identities. The output device
follows the request.
Signed contact distance is opt-in; an empty request retains the five default
channels. A missing distance invalidates only that requested channel.

`lab/sim/sensors/contact/timeline.py` owns bounded per-environment packed history,
driven by an explicit host control clock. Keep it in the environment's sensor
suite so `BaseEnv.step` drives the public observer port. It records no task verdict
and reads no private active-bridge field. Selected resets clear only those rows;
missing/partial intervals and loss remain visible. The contact-row limit is a
source-compatible soft threshold: at most one complete native batch can cross it;
the following sample is unavailable. Reset uses the sensor's index conversion
and aliases, deduplicating rows before advancing generations. Frontends own
report schema wrappers and semantic actor labels. The CPU test
`tests/gym/envs/task_program/test_contact_evidence.py` exercises the actual public
step boundary, frame ownership, native stamps and selective reset behavior.

`contact/observation.py` collects explicit B=1 joint position/velocity
and packed valid contacts through the same sensor hooks. It uses actual
simulation time and separates native contact revision from its acquisition
sequence. Initial captures explicitly request a refresh; substeps reuse the
completed sensor sample. The observer never declares action success or task
acceptance; integrations own occurrence and episode binding.

Window generations reject delayed callbacks after reset, replacement or close.
Records are append-only within a generation; `snapshot(start)` returns an owned
suffix, with `snapshot()` selecting the full window. Limits mark
overflow explicitly, and missing/nonfinite channels preserve unavailability
independently.

`lab/sim/sensors/support_observation.py` owns native load-patch acquisition. Its explicit
`SupportObservationRegistry` reuses an exact native pair across bridge consumers;
same-UID native replacements require release and rebind. Registry cleanup detaches
only owned hooks and leaves simulation-owned resources alive. The pure COM patch
calculation lives in `compute/geometry/contact_support.py`.

Support results separate a physical boolean from availability/error masks.
Global acquisition failures clear every row's witness, while selected resets
advance only those row generations. A positive witness survives sleeping contacts
only while both poses and native identities remain qualified. Callback captures
and direct reads share the actual physics-step identity: a cached contact cannot
re-establish a witness after a same-step pose change. Complete empty
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

`qualified_support_heights` adapts an already qualified native COM/load patch;
it returns the highest accepted contact or NaN for an unavailable row. It does
not create a physical window or choose a Task Program verdict. Geometric world
mesh envelopes are numerical Compute inputs, independent of this contact proof.
