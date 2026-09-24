# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
physical environment is selected separately from the robot embodiment and program.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka (default) | franka_panda | default | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default | `task.ur5.yaml` |
| franka_newton | franka_panda | newton | `task.franka.newton.yaml` |
| ur5_newton | ur5_dh_pgi_140_80 | newton | `task.ur5.newton.yaml` |

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

## Configured trajectory-generation showcase

Run the same deployment with the separate source-neutral Generation Profile:

```bash
python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.demo.yaml \
  --output-dir /tmp/repeated-pick-place-generation \
  --seed 7 --device cuda --headless
```

The showcase executes candidate ordinals 0, 1, and 2 as separate Task Program
episodes and writes JSON provenance plus joint-trajectory plots. Add
`--save-video` for one MP4 per candidate. The profile varies only the Place
retract phase on the original control grid; Pick, release/contact samples, and
phase endpoints remain unchanged.

The explicit environment seed rewinds reset and grasp-sampling streams before
each candidate so all three ordinals start from the same reference episode.

This is a projected-assurance visualization. Completion means the configured
command sequence completed; it is not measured task success, receipt-confirmed
coverage, or a qualified dataset commit.

The task-local `catalog.yaml` names deployments; it does not register a Gym ID.
Generate a simulator-free local gallery from the source checkout with:

```bash
embodichain list-task --config-root embodichain_tasks=embodichain_tasks/configs/tasks --category manipulation --export-html task-gallery.html
```
