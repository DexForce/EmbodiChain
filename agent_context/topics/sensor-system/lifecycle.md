# Sensor lifecycle contracts

Read when changing topology rebuilds or contact adaptation. For entry points and
validation, return to the [sensor overview](sensor-system.md).

## Camera rebuilds

Camera views and their shared targets belong to the existing Spawn Scene.
`SceneBuilder.prepare_arenas()` supplies its lightweight render Scene before
physical finalization, so camera creation must not finalize an incomplete Newton
model. Creation/removal and parent attachment use Scene batches; image buffers
retain the existing environment and left-then-right eye layout.

`SimulationManager.prepare()` detaches parented camera views through
`Scene.detach_render_objects()` before committing a topology rebuild. This also
clears retained physical-parent paths, allowing semantic resolution to retry
after binding rather than replaying stale render-node identities. After scene
binding, the manager resolves new parent nodes and reapplies configured
extrinsics through `attach_to_parent_nodes()`. Preparation without topology
changes leaves existing attachment intact.

This separates three owners: the manager schedules detachment/reattachment,
`attachment.py` resolves semantic asset/link identity, and camera classes apply
the concrete render-node relationship. Missing/ambiguous registered links are
configuration failures; missing per-arena render topology is a runtime failure.

Camera and light facades cache weak native selections only for one Scene
topology revision and reselect their stable handles after finalization/rebuild.
Pose/parameter getters return owned tensors; later reads or writes must not
change earlier results. Camera reset restores selected configured offsets while
preserving the current lens. Removing cameras precedes releasing their empty
shared group; light removal and manager teardown release Scene-owned light
handles before closing the Scene.

## Contact queries

`SimulationManager.add_sensor()` prepares the scene and passes its explicit
owner to `ContactSensor`. The sensor resolves configured rigid/link targets to
Spawn handles and creates one query. Preserve these query/buffer contracts:

- `filter_need_both_actor` chooses query `match="all"` or `"any"`.
- Capacity is bounded per Arena as well as globally, so a busy environment
  cannot consume the entire batch's rows.
- Query positions are arena-local and each row carries `env_ids`; either actor
  may be the selected target and the other may be global.
- Spawn semantic path/link identity preserves targets across Newton rebuilds.
- `user_ids`, `item_user_ids`, `filter_by_user_ids()` and `get_actor_info()` share
  the query's backend-neutral actor identity. Normals point from actor 0 to 1.
- Only validity/counts reset each update. Data outside valid rows is undefined.

`contact_capabilities` distinguishes geometry from normal/friction impulse
support. Do not infer force support from sensor construction alone. Inspect
`supports_contact_sensor` in `embodichain/lab/sim/physics/` for the current
backend/solver/device matrix: Newton AutoSolver may change the answer during
preparation, so the manager checks again afterwards.

The sensor retains backend-emitted rows rather than reducing an actor pair to
one contact. Force-reporting paths apply the query's positive-impulse filtering;
geometry-only paths may provide valid rows with zero impulse. Solver settings
can change manifold density. Fixed sensor capacity may truncate rows; increasing
capacity does not normalize manifold semantics across backends. Consumers that
interpret missing contacts as a physical miss must first check the device-local
`dropped_contacts_mask`: scatter overflow is row-local, while unattributed native
query overflow conservatively invalidates every row. Standalone updates report
the latest query; history-enabled sensors retain the current interval's losses.

PhysX Direct GPU cannot identify static counterparts from its raw contact
buffer and uses actor ID `-1` for those rows. To observe arbitrary static
geometry there, monitor the dynamic/link side with `filter_need_both_actor=False`.
Default CPU and Newton can identify registered static shapes. Keep this
limitation at the query boundary rather than substituting render IDs.
