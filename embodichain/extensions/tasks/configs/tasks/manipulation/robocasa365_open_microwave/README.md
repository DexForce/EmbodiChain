# RoboCasa365 OpenMicrowave workcell

A Franka FR3 with its parallel hand opens Microwave048 on a counter. The
fixture preserves the meshes, collision geometry, hinge and button joints of
RoboCasa365 OpenMicrowave episode 86 (layout 56, style 15). The Franka and
pedestal occupy the same pose in the Task Program and policy deployments.
The existing `sim/robocasa365/replay` example reproduces the source kitchen
and its recorded PandaOmron trajectory.

```bash
embodichain run-task --gym-config embodichain/extensions/tasks/configs/tasks/manipulation/robocasa365_open_microwave/task.franka.yaml --headless --disable-sensor --renderer no-render --device cpu --seed 0
```

The default catalog deployment runs one registered `simulation.open_door`
call. The native `OpenDoor` skill reads the live door-link pose and hinge
position, approaches the vertical handle, closes the fingers, follows the
hinge arc, opens the fingers and retreats. The handle is one box among the
door's physical collision shapes. The integration selects it by a point in
door coordinates and applies the Franka jaw calibration in its local frame.
The Cartesian Franka embodiment holds the seventh-joint IK branch fixed
through the swing. The robot moves the door through contact.

The final evaluation reads the native `microwave_right_group_1_microjoint`
position. This joint opens in the negative direction from 0 to −π/2 rad.
The benchmark requires at least 81° of opening (90% travel) within the
300-step episode. The timer and buttons have separate joints. Add
`--record-trajectory --trajectory-save-dir outputs/open_microwave` to save
the physical robot and fixture trajectory.

The catalog's `franka_policy` deployment (`task.policy.franka.yaml`, Gym
ID `RoboCasa365OpenMicrowave-Franka-v1`) accepts relative arm-joint actions
and a continuous parallel-jaw command. It exposes the fixture's joint
positions and computes per-arena reward and success from the measured
fraction of hinge travel.

LightWheel AI created the model, NVIDIA published the
[kitchen MJCF collection](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF),
and the [RoboCasa team](https://robocasa.ai/) supplied the episode configuration.
The asset archive includes the CC BY 4.0 license, attribution, source checksums,
selected fixture MJCF and conversion information. It contains only the appliance.
