# Preview a Robot

`embodichain preview-robot` loads one catalog robot or a standalone robot
configuration. The default frontend is the native simulation window.
`python -m embodichain preview-robot` is equivalent.

## Catalog selection

```bash
embodichain list-robot
embodichain preview-robot --robot FrankaPanda
embodichain preview-robot --robot URRobot --variant ur5e
embodichain preview-robot --robot DualArmRobot --variant ur5 --viser
embodichain preview-robot --robot DexforceW1 --variant v025/DH_PGC --viser
embodichain preview-robot --robot DexforceW1 --variant v030/bare --viser
```

IDs are exact catalog names, including installed third-party providers.
`--variant` uses the variant name displayed by `list-robot`. Omitting it
preserves the robot's existing construction defaults.

W1 lists V021/V022/V025 hand and bare-body profiles, plus the standard V030
bare body. See {doc}`/resources/robot/dexforce_w1` for version-specific solvers,
TCPs and optional attachments.

`--robot-params` accepts a JSON object of robot configuration overrides:

```bash
embodichain preview-robot --robot DexforceW1 --variant v025/BrainCo \
    --robot-params '{"uid":"inspection_robot"}' --viser
```

The precedence is catalog defaults, explicit variant defaults, file fields,
then CLI robot parameters. These are top-level field overrides; each
configuration class retains its existing nested decoding rules.

## Robot configuration files

`--robot-config` accepts JSON, YAML, or YML and is mutually exclusive with
`--robot`. The file root contains `RobotCfg` fields:

```yaml
uid: custom_arm
fpath: ./arm.urdf
root_props:
  fixed_base: true
control_parts:
  arm: ["joint[1-6]"]
solver_cfg:
  arm:
    class_type: PytorchSolver
    root_link_name: base_link
    end_link_name: tool0
```

Joint and link names must match the chosen asset. Local asset paths resolve
relative to the configuration directory. Registered asset references continue
to use the shared asset registry. URDF assembly components and solver URDF
paths support the same local-path resolution.

An optional top-level `class_type` selects a catalog configuration rather
than generic `RobotCfg`:

```yaml
class_type: URRobot
uid: custom_ur
robot_type: ur5
```

```bash
embodichain preview-robot --robot-config /path/to/robot.yaml --viser
```

`--variant` requires a catalog selection, either `--robot` or file
`class_type`. Files contain robot configuration fields; Gym deployments and
embodiment components keep their existing loaders.

## Interaction

In the native window, press **I** to activate or toggle the existing managed
IK Gizmos, then drag a control part's target. In Viser, drag its Gizmo directly.
Only defined parts with complete IK chain/TCP metadata are registered, and
Gizmo control writes the part's independent joint targets. Registration does
not activate native IK or change the robot's targets. Automatic Gizmos retain
the current single-environment requirement and default DexSim IK backend.

With `--viser`, the existing joint panel exposes independent scalar joints.
It supports joint limits, numeric inputs for continuous/unbounded joints, and
reset to the initial pose. Parts without IK metadata remain controllable
through the joint panel. Slider commands and Gizmo writes are processed on the
simulation thread; subsequent steps adopt the latest drive targets so an IK
pose is not overwritten by an old slider pose. When both arrive in one step,
Gizmo processing follows slider processing.

```bash
# Browser joint controls without IK Gizmos
embodichain preview-robot --robot ur5 --viser --no-gizmo

# Gizmos without the joint panel
embodichain preview-robot --robot ur5 --viser --no-joint-control
```

The native frontend and Viser are mutually exclusive. Native preview requires
a working display and renderer. Common `--viser-host`, `--viser-port`,
`--physics`, `--device`, and renderer arguments are available in
`preview-robot --help`.

## Headless checks and exit

```bash
# Load and validate, then exit without stepping
embodichain preview-robot --robot ur5 --headless --renderer no-render

# Load and execute ten physics steps with Newton
embodichain preview-robot --robot ur5 --physics newton \
    --headless --renderer no-render --max-steps 10
```

Pure headless mode exits after loading unless `--max-steps` is supplied.
Native and Viser previews continue until Ctrl+C or their step limit.
Every run owns and cleans up its simulation; existing robot-specific module
demonstrations remain available.
