# Drawer storage Task Program

A Default-physics UR5 + DH PGI 140/80 deployment using the existing drawer
asset and shared Task Program/Atomic Skill execution. The program opens the
rail to 0.26 m, picks the 50 mm cube, places it in the drawer, releases and
retreats, then closes the rail. No simulator state writes or attachment welds
are used during execution.

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage/task.ur5.yaml
python -m embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage/task.ur5.yaml \
  --headless --device cpu --renderer no-render --disable-sensor --seed 7
```

The pedestal, pickup height, lifted transfer and opening distance are specific
to this workstation. The cube must clear the drawer front before crossing it;
a larger lift or rail opening is not interchangeable with a faster trajectory.
The top-down cube grasp is calibrated to the selected end effector. This is
one embodiment and one fixed scene, not a cross-robot or randomized-scene claim.

The execution policy is projected: command completion alone does not prove the
physical effects. The open rail and lifted cube have measured segment validators;
opening and post-release/post-close stability have measured windows. The
registered environment independently requires every cube corner inside the
actual drawer frame, a rail position in [-1 mm, 10 mm], an open measured hand
and both finger origins more than 80 mm from the cube. Segment failure cannot
be bypassed by the final geometric check. The interior uses the asset's inner
side/floor bounds with 1 mm contact tolerance, rather than an XY-only centre test.

Physical qualification remains a pilot until the separate fixed-source
qualification results establish repeatability. Retain failed attempts and report
all denominators; do not infer robustness from static checks or one successful
collection. The existing tableware drawer task and its success labels are unchanged.
