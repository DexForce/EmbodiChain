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

`ContactSensor` owns sample caching using the manager's public
`physics_step_index`. Ordinary Gym observation reads call `sensor.update()`;
the sensor queries at most once for each newly observed physics step. Repeated
reads reuse its buffer. Creation and reset suppress SDK queries until the next
physics step, since solver reset invalidates the current contact snapshot.
Selected reset rows are invalidated while untouched rows retain their last
sample; an empty reset leaves cache eligibility unchanged. Standalone and replay
updates use the same sensor logic without Gym-specific sampling state.

Histories retain per-substep sampling. Raw contact refresh and history reduction
track their completed steps separately so reading a raw sample cannot suppress
its history update, and duplicate callbacks do not advance history twice.
Ordinary sensors do not acquire a decimation-factor query overhead.

CUDA queries use `fetch_async()` and device-resident counts. Sparse Warp kernels
process valid contact rows, using precomputed sorted environment/actor indices;
no environment-by-contact-by-body tensor is allocated. Per-body force, contact
flags and timing buffers persist between samples. `current_air_time` measures
an unfinished flight; `last_air_time` retains its completed duration on landing.
Selective reset clears only the selected rows.

`dropped_contacts` reports query loss plus separately counted scatter overflow
across the control interval. Query counts stay on the device until this diagnostic
is read. A standalone sensor without histories reports the latest update.
See `tests/sim/sensors/test_contact_history.py`, `test_contact_query_sensor.py`
and `tests/gym/envs/test_substep_sensors.py`.
