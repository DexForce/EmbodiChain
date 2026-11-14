# External joint encoder sensor

This example is a separate Python distribution. It registers
`ExampleJointEncoder` through an `embodichain.sensors` entry point and consumes
only `SensorContext.state.get_joint_state()`. It does not modify EmbodiChain's
built-in modules or sensor dictionaries and does not access native runtime objects.

Install the current EmbodiChain checkout in your supported Python environment,
then install this example before starting the application:

```bash
python -m pip install -e .
python -m pip install --no-deps ./examples/sim/sensors/external_joint_encoder
```

With an existing `SimulationManager` named `sim` and a robot or articulation
registered as `robot`, create the sensor from Python:

```python
from embodichain_joint_encoder_example import JointEncoderCfg

sensor = sim.add_sensor(JointEncoderCfg(uid="joint_encoder", asset_uid="robot"))
sensor.update()
positions = sensor.get_data()["qpos"]
```

Or use the same registry through YAML:

```python
from pathlib import Path
import yaml
from embodichain.lab.sim.sensors import SensorCfg

path = Path("examples/sim/sensors/external_joint_encoder/sensor.yaml")
cfg = SensorCfg.from_dict(yaml.safe_load(path.read_text()))
sensor = sim.add_sensor(cfg)
```

The same declaration also works in an independent Gym `sensor` list or an
embodiment component's independent sensor suite. Mounted non-camera sensors,
unified sampling periods and non-image rollout-buffer descriptions belong to
the next runtime-contract milestone; this example does not require them.

Outputs are `(num_instances, dof)` tensors on the context device, in the
articulation's joint order: angular joints use radians and rad/s, and linear
joints use metres and m/s. Acquisition does not step physics. The provider
returns owned snapshots; reset accepts list or Tensor environment selections
and clears only those rows. Physical and sensor UIDs may overlap. There is no noise,
quantization, odometry or six-axis force measurement in this example.

## Writing another plugin

Define a `@configclass` subclass of `SensorCfg` with a unique default
`sensor_type`, a factory accepting `factory(cfg, *, context)`, and a
`SensorRegistration` value. Point an entry point with that exact type name at
the registration. Keep the package initializer lightweight; entry-point names
are discovered without loading implementations, and only a selected plugin is
imported. Duplicate identifiers (including built-ins) are errors. API version 1
is currently supported. Load errors identify the entry point and retain their
original cause. Installations are discovered once per process; restart after
installing or removing plugins.

For explicit in-process registration, call `register_sensor(SENSOR)` instead
of installing an entry point; do not use both for the same identifier.
`legacy_factory=True` explicitly adapts old `factory(cfg, device)` factories.
Existing `SimulationManager.SUPPORTED_SENSOR_TYPES` overrides remain a legacy
compatibility path; new plugins use the registry so configuration and creation
share one resolution contract. Direct Camera, StereoCamera and ContactSensor
constructor signatures and manager ownership remain unchanged.
