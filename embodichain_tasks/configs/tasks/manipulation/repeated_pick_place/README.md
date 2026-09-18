# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
physical environment is selected separately from the robot embodiment and program.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka (default) | franka_panda | default | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default | `task.ur5.yaml` |
| franka_newton | franka_panda | newton | `task.franka.newton.yaml` |
| ur5_newton | ur5_dh_pgi_140_80 | newton | `task.ur5.newton.yaml` |
| ur5_objective | ur5_dh_pgi_140_80 | default | `task.ur5.objective.yaml` |

Inspect the available deployments:

```bash
embodichain show-task embodichain_tasks:repeated_pick_place
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

Each deployment owns its physics backend through its environment component. Choose
a deployment for another backend; `--physics` does not switch an existing one.
Task Program execution is the supported expert demonstration route. These original
examples do not independently qualify measured physical placement. No preview or
physical qualification result is claimed by this catalog.

The task-local `catalog.yaml` names deployments; it does not register a Gym ID.
Generate a simulator-free local gallery from the source checkout with:

```bash
embodichain list-task --config-root embodichain_tasks=embodichain_tasks/configs/tasks --category manipulation --export-html task-gallery.html
```

## Independent physical evaluation

The `ur5_objective` deployment observes the cube in ordered A, B, A regions.
Each region requires 0.1 seconds with linear speed at most 0.05 m/s and angular
speed at most 0.2 rad/s. Bounds describe the settled cube center near ground
height 0.025 m, not the gripper's 0.10 m release pose. The initial B position
does not count before A. Historical progress is retained, but leaving the final
region revokes physical success; regaining it requires another stable hold.
This predicate does not prove gripper detachment.

Run one expert episode and dynamic action replay using the same objective:

```bash
python -m embodichain.lab.scripts.evaluate_task_objective \
  --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.objective.yaml \
  --output-dir objective-baseline --seed 0 --num_envs 1 --headless --device cuda

python -m embodichain.lab.scripts.evaluate_task_objective \
  --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.objective.yaml \
  --output-dir objective-perturbed --seed 0 --num_envs 1 --headless --device cuda \
  --initial-position-jitter 0.005
```

Choose a new output directory for each invocation. The second command samples
initial X/Y offsets within ±0.005 m through the existing reset event mechanism;
the report records the actual pose. Reproducibility assumes the same deployment,
runtime, seed and reset schedule; it is not a cross-backend determinism guarantee.

`report.json` retains authored component contents/hashes, resolved configuration,
code provenance and separate execution, physical, demo-acceptance and persistence
outcomes. The optional 20-step settling tail is reported explicitly and included
in `expert.pt`; this diagnostic trajectory is marked as controller actions for
dynamic replay. Dataset saving is disabled. Kinematic playback is not a physical
execution qualification.

The checked-in [startup record](validation/default-startup.json) documents a real
attempt blocked before rollout: the installed DexSim descriptor API did not accept
`com_quaternion`, required by the existing EmbodiChain spawn adapter. Its physical
outcome is unavailable. It must not be interpreted as a failed task measurement
or a successful qualification. Re-run the commands with a compatible DexSim build
to obtain expert/replay physical results.
