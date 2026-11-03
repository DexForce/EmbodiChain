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

`task_program/integrations/simulation/contact_evidence.py` adapts sampled buffers
to the existing physical-evidence frame/batch types. It retains the validity mask,
native actor IDs, Arena frames and impulses, with explicit unsupported, nonfinite,
saturated or source-drift errors. Native and runtime observation revisions remain
different identities. The provider's output device follows its request.

`contact_timeline.py` in that integration owns bounded per-environment packed
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
