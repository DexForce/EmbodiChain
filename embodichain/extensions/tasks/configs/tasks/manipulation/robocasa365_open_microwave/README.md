# RoboCasa365 OpenMicrowave workcell

A Franka FR3 with its parallel hand controls the door of Microwave048 on a
counter. The fixture uses the calibrated mesh scales, collision geometry, hinge
and button joints of RoboCasa365 OpenMicrowave episode 86 (layout 56, style 15).
The robot and workcell placement are specific to this deployment. The existing
`sim/robocasa365/replay` example separately reproduces the entire source
kitchen and its recorded PandaOmron trajectory.

```bash
embodichain run-task --gym-config embodichain/extensions/tasks/configs/tasks/manipulation/robocasa365_open_microwave/task.franka.yaml --preview
```

Use `gym.make` with the resolved deployment to supply relative arm-joint actions
and a continuous parallel-jaw command. Observations include the articulation's
native joint positions. The success metric reads only
`microwave_right_group_1_microjoint` and requires at least 90% of the measured
travel from the closed upper limit to the open lower limit. Operating the timer
and buttons does not open the door.

LightWheel AI created the model, NVIDIA published the
[kitchen MJCF collection](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF),
and the [RoboCasa team](https://robocasa.ai/) supplied the episode configuration.
The asset archive includes the CC BY 4.0 license, attribution, source checksums,
selected fixture MJCF and conversion information. It contains only the appliance.
