# PikaGripper on UR5

This example mounts the standalone DexEChain Pika gripper on the existing UR5
catalog robot. The original archive is hosted at
[DexForceAI/embodichain_data/eef_assets/PikaGripper.zip](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/main/eef_assets/PikaGripper.zip).

From the repository root:

```bash
embodichain data download --name PikaGripper
embodichain preview-robot \
  --robot-config examples/sim/robots/pika_gripper/ur5.yaml \
  --headless --no-gizmo --viser
```

Open the printed Viser URL. Use the `FINGER1_JOINT` slider to open and close both
jaws; the second finger follows through its URDF mimic relation. `arm` contains
six independent UR5 joints. `eef` contains both finger joints, with one independent
control joint. The stock UR5 IK solver continues to target the wrist `ee_link`;
the example does not define a calibrated fingertip TCP.

The URDF assembler connects `ee_link` to `GRIPPER_ROOT_LINK` with an identity
mount transform. Pika's root adapter rotation remains inside its source URDF.
Visual meshes use DAE; collision meshes use STL. Asset paths resolve with
`get_data_path("PikaGripper/PikaGripper.urdf")` and download on first use.

The finger command is displacement per jaw: 0 m is closed, 0.025 m is half open,
and 0.05 m is fully open. This is distinct from the distance between the finger
link origins. Their separation increases by 0.10 m over the full stroke.

Qualification used the Default physics backend and CPU simulation, with FastRT
rendering on an RTX 5090. All three commanded positions settled within 0.01 mm,
and the measured separation change was 99.992 mm. This verifies assembly,
rendering, actuation and mimic following; grasping/contact qualification is a
separate task.
