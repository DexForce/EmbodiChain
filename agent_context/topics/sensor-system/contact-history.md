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
Selective reset clears only the selected rows. Force reduction uses only
components marked available by the sample's normal/friction validity flags.
Geometry-only rows can count as contact when no force threshold is requested;
they do not create measured force or satisfy a positive force threshold.
Native friction is the tangential impulse on actor 0, while the normal points
toward actor 1: actor-0 force is `(-normal * impulse + friction) / dt` and
actor 1 receives the opposite force.

`dropped_contacts` reports query loss plus separately counted scatter overflow
across the control interval. Query counts stay on the device until this diagnostic
is read. A standalone sensor without histories reports the latest update.
See `tests/sim/sensors/test_contact_history.py`, `test_contact_query_sensor.py`
and `tests/gym/envs/test_substep_sensors.py`.
