# Contact history and substeps

`ContactSensor.create_history()` registers sensor-owned `ContactHistory` objects
with explicit body IDs, optional counterpart IDs, and a contact-force threshold.
`get_actor_ids(uid, link_names)` resolves Spawn identities in environment order.
Unidentified counterpart IDs (`-1`) are accepted only through an explicit option;
they do not establish that the counterpart is ground.

With `track_substeps=True`, BaseEnv calls `begin_control_step()` once, advances
physics, then calls `update_physics_step(dt)` after each substep. Regular
observation reads do not advance history. Current sample, interval occurrence,
peak force and first-contact flags remain distinct. `current_air_time` measures
an unfinished flight; `last_air_time` retains the completed duration on landing.
Sensor reset clears only selected rows. See `tests/sim/sensors/test_contact_history.py`
and `tests/gym/envs/test_substep_sensors.py`.
