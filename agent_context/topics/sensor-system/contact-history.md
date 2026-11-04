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

CUDA queries use `fetch_async()` and device-resident counts. Sparse Warp kernels
process valid contact rows, using precomputed sorted environment/actor indices;
no environment-by-contact-by-body tensor is allocated. Per-body force, contact
flags and timing buffers persist between samples. `current_air_time` measures
an unfinished flight; `last_air_time` retains its completed duration on landing.
Selective reset clears only the selected rows.

`dropped_contacts` reports query loss plus separately counted scatter overflow
across the control interval. Query counts stay on the device until this diagnostic
is read. A standalone sensor without histories or sampling observers reports the latest update.
`latest_sample_dropped_contacts` separately reports the latest query and scatter
loss even when interval counters are retained. Its device counters are updated
by the existing CUDA scatter kernel, including graph replay, without another
copy launch. CPU uses a scalar copy to avoid a per-row scatter branch. Full reset clears both; selective
reset retains query-wide loss. Evidence rejects loss from either diagnostic and
uses their maximum, since the counts overlap. `evidence_dropped_contacts` reads
the packed interval/latest counters once and reduces them with host int64 sums.
The standard sampler uses this combined diagnostic; custom sources without it
retain the separate-counter fallback.
See `tests/sim/sensors/test_contact_history.py`, `test_contact_query_sensor.py`
and `tests/gym/envs/test_substep_sensors.py`.

## Sampling observers

`ContactSensor.add_sampling_observer` attaches a named, instance-owned read-only
observer to this same lifecycle. It receives begin, before-sample, after-sample and
selected-row reset notifications. With no observers, the update path does not
enumerate subscriptions, construct token generators or enter callback loops. Hooks run outside the sensor's internal CUDA capture; observers
inside an external capture fail before their callbacks run. Existing history
sampling and device kernels keep their ownership and ordering.

Observer tokens and after-sample callbacks run once per physical step, alongside
history reduction. A raw sample fetched earlier in the same step is reused;
repeated reads do not advance history or repeat callbacks.

`lab/sim/sensors/contact/sampling.py` adapts sampled buffers
to the shared `lab/sim/evidence.py` frame/batch types. It retains the validity mask,
native actor IDs, Arena frames and impulses, with explicit unsupported, nonfinite,
saturated or source-drift errors. Native and runtime observation revisions remain
different identities. The provider's output device follows its request.

`lab/sim/sensors/contact/timeline.py` owns bounded per-environment packed
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
