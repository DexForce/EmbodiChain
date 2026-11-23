# Robot

```{currentmodule} embodichain.lab.sim
```

{class}`~objects.Robot` extends {class}`~objects.Articulation` with named
control parts, part-specific kinematic solvers, and proprioceptive observations.
Use the {doc}`sim_articulation` interface for common link, joint, and drive
properties.

## Configuration

Run `embodichain list-robot` to find installed presets. Use a catalog ID with
`resolve_robot_cfg()` when a preset already describes the model and
calibration.

```python
from embodichain.robot_registry import resolve_robot_cfg

cfg = resolve_robot_cfg("FrankaPanda")
```

Catalog resolution may download model assets. See
{doc}`/guides/robot_catalog` for variants, overrides, and the control summary.
For a {class}`~cfg.RobotPresetCfg` with backend-specific alternatives,
`sim.add_robot(cfg)` selects one complete alternative from the active
physics configuration.

Use {class}`~cfg.RobotCfg` for an explicitly configured robot.

| Field | Meaning |
| :--- | :--- |
| `fpath` or `source` | Single URDF, or a selected USD articulation with `UsdSourceCfg`. |
| `urdf_cfg` | Optional assembly of multiple URDF components. |
| `control_parts` | Mapping of part names to joint names or regex patterns. Each part is resolved into runtime joint order. |
| `solver_cfg` | Solver configuration for kinematics. For named control parts, supply a mapping keyed by part name for the parts that need solvers. |
| `sensors`, `enable_sensor` | Mounted sensor configurations and the standalone acquisition switch. Parent links bind at preparation. |

Official single-arm presets typically use `arm` and `eef`. Dual-arm
presets typically use `left_arm`, `right_arm`, `left_eef`, and `right_eef`.
Custom control-part names are supported. Keep the asset's link and joint
names in configuration selectors.

### Setup & Initialization

This headless example uses the repository's one-joint hinge URDF as a small
robot. Run it from the repository root. The `arm` group gets a kinematic
solver, and its joint position target is initialized from measured state.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, RobotCfg
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

urdf = "examples/sim/sensors/state/hinge.urdf"
cfg = RobotCfg(
    uid="hinged_robot",
    fpath=urdf,
    init_pos=(0.0, 0.0, 0.3),
    control_parts={"arm": ["hinge"]},
    solver_cfg={
        "arm": PytorchSolverCfg(
            urdf_path=urdf,
            root_link_name="base",
            end_link_name="arm",
            num_samples=1,
        )
    },
)
sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    robot = sim.add_robot(cfg)
    sim.prepare()
    arm_qpos = robot.get_qpos(name="arm")  # (1, 1)
    robot.set_qpos(arm_qpos, name="arm", target=True)
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Robot Class

### Control Parts

Use `robot.get_joint_ids(name="arm")` to find the columns for one part in
the complete joint tensor. Pass `remove_mimic=True` to retrieve its
independent joint IDs. With named state getters, the returned columns
follow the resolved part order.

In a prepared scene with the example robot, the following holds the
current arm pose in environment zero. The input retains the leading
environment dimension.

```python
target = robot.get_qpos(name="arm")[0:1].clone()
robot.set_qpos(target, name="arm", env_ids=[0], target=True)
sim.update(step=1, render_final_step=False)
```

Use `set_qvel()` for velocity targets and `set_qf()` for generalized effort
where the drive and solver support those commands. For a direct joint-state
write, pass `target=False` to `set_qpos()` or `set_qvel()`.

### Kinematics (Solvers)

For a named part, FK and IK use its configured solver. Check
`robot.get_solver("arm")` before relying on a solver for a preset. Joint
positions and seeds use the selected solver's joint order.

#### Forward Kinematics (FK)

`compute_fk(qpos=..., name="arm")` returns the configured tool pose in the
arena-local frame. For a single target per environment, the default shape is
`(E, 7)` in `xyz + xyzw` order. Pass `to_matrix=True` for `(E, 4, 4)`.

```python
arm_qpos = robot.get_qpos(name="arm")
tool_pose = robot.compute_fk(qpos=arm_qpos, name="arm", to_matrix=True)
```

#### Inverse Kinematics (IK)

Give `compute_ik()` a goal in the same arena-local frame and a seed for
each selected environment. The default result contains one solution row
per environment and a success mask. Apply the returned targets only to
successful rows.

This example constructs a reachable goal from a nearby joint position for
the one-joint robot above.

```python
seed = robot.get_qpos(name="arm")
goal = robot.compute_fk(qpos=seed + 0.1, name="arm", to_matrix=True)
success, solution = robot.compute_ik(pose=goal, joint_seed=seed, name="arm")
env_ids = success.nonzero(as_tuple=True)[0].tolist()
if env_ids:
    robot.set_qpos(solution[env_ids], name="arm", env_ids=env_ids, target=True)
```

### Proprioception

`robot.get_proprioception()` returns a batched `TensorDict` with `qpos`,
`qvel`, and `qf` in the robot's joint order. Use the named getters when an
observation needs one control part.

### Advanced API

| Task | Interface |
| :--- | :--- |
| Read effective position, speed, and effort limits | `get_qpos_limits(name=...)`, `get_qvel_limits(name=...)`, `get_qf_limits(name=...)` |
| Change supported physical limits for a part | `set_qpos_limits(..., name=...)`, `set_qvel_limits(..., name=...)`, `set_qf_limits(..., name=...)` |
| Tighten the search range for planning or IK | Configure `SolverCfg.user_qpos_limits`. The solver intersects them with the effective physical position limits. |
| Reset an episode | Use `sim.reset_objects_state(env_ids=...)` to coordinate the robot and the rest of the scene. |

## Related Topics

- {doc}`/guides/robot_catalog` and {doc}`/resources/robot/index` for concrete
  robot configurations.
- {doc}`sim_articulation` for joint drives, combined state writes, and
  physical properties.
- {doc}`Robot tutorial </tutorial/robot>` for rendered simulation and URDF
  assembly examples.
