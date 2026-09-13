# Contact Sensor

```{currentmodule} embodichain.lab.sim.sensors
```

## Configuration

The {class}`ContactSensorCfg` class defines the configuration for contact sensors. It inherits from {class}`~SensorCfg` and enables filtering and monitoring of contact events between specific rigid bodies and articulation links in the simulation. The same API works with the Default (DexSim adapter) and Newton physics backends.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rigid_uid_list` | `List[str]` | `[]` | List of rigid body UIDs to monitor for contacts. |
| `articulation_cfg_list` | `List[ArticulationContactFilterCfg]` | `[]` | List of articulation link contact filter configurations. |
| `filter_need_both_actor` | `bool` | `True` | Whether to filter contact only when both actors are in the filter list. If `False`, contact is reported if either actor is in the filter. |
| `max_contacts_per_env` | `int` | `64` | Maximum number of contacts per environment that the sensor can handle. |

The sensor forwards `max_contacts_per_env` to DexSim as a per-Arena query
quota. If the global contact buffer is also full, DexSim distributes retained
rows across Arenas before filling later rows from the same Arena.

## Articulation Contact Filter Configuration

The `ArticulationContactFilterCfg` class specifies which articulation links to monitor for contacts.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `articulation_uid` | `str` | `""` | Unique identifier of the articulation (robot or articulated object). |
| `link_name_list` | `List[str]` | `[]` | List of link names in the articulation to monitor. If empty, all links are monitored. |

## Usage

You can create a contact sensor using `sim.add_sensor()` with a `ContactSensorCfg` object.

### Code Example

```python
from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg, ArticulationContactFilterCfg
import torch

# 1. Define Contact Filter Configuration
contact_filter_cfg = ContactSensorCfg()

# Monitor contacts for specific rigid bodies
contact_filter_cfg.rigid_uid_list = ["cube0", "cube1", "cube2"]

# Monitor contacts for specific articulation links
contact_filter_art_cfg = ArticulationContactFilterCfg()
contact_filter_art_cfg.articulation_uid = "UR10_PGI"
contact_filter_art_cfg.link_name_list = ["finger1_link", "finger2_link"]
contact_filter_cfg.articulation_cfg_list = [contact_filter_art_cfg]

# Only report contacts when both actors are in the filter list
contact_filter_cfg.filter_need_both_actor = True

# Set maximum contacts per environment
contact_filter_cfg.max_contacts_per_env = 128

# 2. Add Sensor to Simulation
contact_sensor: ContactSensor = sim.add_sensor(sensor_cfg=contact_filter_cfg)

# 3. Update and Retrieve Contact Data
sim.update(step=1)
contact_sensor.update()
contact_report = contact_sensor.get_data()

# Access contacts for a specific environment using is_valid mask
env_id = 0
env_valid_mask = contact_report["is_valid"][env_id]
env_contact_positions = contact_report["position"][env_id][env_valid_mask]

# Or get all valid contacts across all environments
valid_mask = contact_report["is_valid"]
all_valid_positions = contact_report["position"][valid_mask]  # Shape: (total_valid_contacts, 3)

# 4. Filter contacts by backend-neutral contact actor IDs
filter_user_ids = torch.as_tensor(
    [
        actor_id
        for actor_id in contact_sensor.item_user_ids.tolist()
        if contact_sensor.get_actor_info(actor_id).path.endswith("/cube2")
        or contact_sensor.get_actor_info(actor_id).link_name == "finger1_link"
    ],
    dtype=torch.int32,
    device=sim.device,
)
# Filter for specific environments
filter_contact_report = contact_sensor.filter_by_user_ids(filter_user_ids, env_ids=[env_id])

# 5. Visualize Contact Points
contact_sensor.set_contact_point_visibility(
    visible=True,
    rgba=(0.0, 0.0, 1.0, 1.0),  # Blue color
    point_size=6.0,
    env_ids=[env_id],  # Optional: visualize only specific environments
)
```

## Observation Data

Retrieve contact data using `contact_sensor.get_data()`. The data is returned as a dictionary of tensors on the specified device.

| Key | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `position` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Contact positions in arena frame (world coordinates minus arena offset). |
| `normal` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Unit normal vectors pointing from actor 0 toward actor 1. |
| `friction` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Tangential contact impulse applied to actor 0. Availability is reported by `contact_capabilities.friction`. |
| `impulse` | `torch.float32` | `(num_envs, max_contacts_per_env)` | Backend contact impulse magnitudes. Default CPU reports the total impulse norm; Direct GPU and force-reporting Newton solvers report normal impulse magnitude. |
| `distance` | `torch.float32` | `(num_envs, max_contacts_per_env)` | Signed contact separation (negative means penetration). |
| `user_ids` | `torch.int32` | `(num_envs, max_contacts_per_env, 2)` | Pair of query-local, backend-neutral contact actor IDs. The legacy field name is retained; resolve IDs with `get_actor_info()`. |
| `is_valid` | `torch.bool` | `(num_envs, max_contacts_per_env)` | Boolean mask indicating which contact slots contain valid data. Use this mask to filter out unused slots. |

**Note**: Use the `is_valid` mask to access only valid contacts:
```python
# Get all valid contacts across all environments
valid_mask = contact_report["is_valid"]
valid_positions = contact_report["position"][valid_mask]  # Shape: (total_valid_contacts, 3)

# Or access per-environment
env_id = 0
num_valid = contact_report["is_valid"][env_id].sum().item()
env_positions = contact_report["position"][env_id, :num_valid]
```

Force-capable backends preserve every backend-emitted contact row that passes
the positive-impulse filter. Default CPU uses a total-impulse norm threshold of
`1e-7`; Default Direct GPU and force-reporting Newton solvers use the same
threshold on normal impulse. Geometry-only Newton solvers retain all candidate
rows with zero impulse. The sensor does not synthesize a common contact
manifold: solver options such as MuJoCo-Warp's `enable_multiccd` control how
many points the backend emits for a geometry pair. Multiple contact points and
shape pairs can map to the same actor pair. The fixed-size numeric buffers are
not cleared every update; values where `is_valid=False` are unspecified and
may be left over from an earlier update.

## Additional Methods

- **`get_actor_info(actor_id)`**: Resolve a contact actor ID to its Spawn path, articulation link, Arena, and environment ID.
- **`contact_capabilities`**: Report whether geometry, normal impulse, and friction impulse are available for the active backend/solver.
- **`filter_by_user_ids(item_user_ids, env_ids=None)`**: Filter contact report by contact actor IDs. The method name is retained for compatibility. Optionally filter by specific environment IDs.
- **`set_contact_point_visibility(visible, rgba, point_size, env_ids=None)`**: Enable/disable visualization of contact points with customizable color and size. Optionally visualize only specific environments.

Newton MuJoCo-Warp exposes contact forces, so both impulse fields are available. Other supported Newton rigid solvers currently expose contact geometry with zero-valued impulse fields. MuJoCo CPU mode does not expose device contact buffers, and DexUni does not currently publish rigid contacts through `ContactQuery`; those modes are therefore unsupported by this sensor.

Default Direct GPU reports static counterparts with actor ID `-1` because its raw contact buffer does not expose their object identity. To monitor a dynamic body or articulation link against arbitrary static geometry, select the dynamic/link object and set `filter_need_both_actor=False`. Default CPU and Newton can identify registered static shapes.
