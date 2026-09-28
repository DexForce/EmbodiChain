# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
physical environment is selected separately from the robot embodiment and program.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka (default) | franka_panda | default | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default | `task.ur5.yaml` |

Inspect the available deployments:

```bash
embodichain show-task embodichain_tasks:repeated_pick_place
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

Each deployment offers `default` and `newton` environment variants. Select the
backend with `--physics`; requesting an undefined variant fails before simulator
construction.
Task Program execution is the supported expert demonstration route. These original
examples do not independently qualify measured physical placement. No preview or
physical qualification result is claimed by this catalog.

## Configured trajectory-generation showcase

The generation deployment binds a default profile directly in
`task.ur5.generation.yaml`, so the formal task entry point can run the complete
64-recipe batch schedule without an example-specific wrapper:

```bash
embodichain run-task \
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.generation.yaml \
  --num-envs 16 --arena-space 2.5 --device cpu --headless --seed 7 \
  --output-dir /tmp/repeated-pick-place-generation \
  --dataset-dir /tmp/repeated-pick-place-lerobot
```

`--generation-profile` and `--generation-candidate-indices` override the values
declared by the task deployment. The profile remains a separate resource so a
deployment selects an expansion policy without duplicating its full schema.
The dataset recorder is owned by `env.yaml`; pass `--filter-dataset-saving` to
run without saving episodes.

The Python showcase remains available for visual debugging, but it uses the same
combined profile and task deployment as the formal entry point. The formal
`run-task` command is the supported collection path.

The combined profile currently exposes a simulator-free deterministic preflight
for the episode scheduler. It validates the 16-row capacity, enumerates four
reference families and all 64 episode recipes, and writes `schedule.jsonl` plus
an explicit `configured_preflight` manifest:

```bash
python examples/sim/motion/repeated_pick_place_generation_combined.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.generation.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.combined.yaml \
  --output-dir /tmp/repeated-pick-place-generation
```

The physical host runner and measured artifact capture consume this schedule;
the preflight command does not claim simulator execution or task success.

For the simulator showcase, the task's `generation.runtime` overlay registers
a `generation_profile_reset` reset event. The runner supplies the selected
candidate's cube pose and visual assignments before each reset, and the event
applies them through the environment reset lifecycle.

The generation runtime overlay now enables the LeRobot recorder. A successful
combined showcase commits one LeRobot episode per physical row at
`/tmp/repeated-pick-place-lerobot` and stores the generation records in
`meta/embodichain_episodes.jsonl`.

It also enables the synchronous `record_camera_data` event recorder. With
`num_envs: 16`, each captured frame is composed as a 4×4 grid and the episode
video is written to `/tmp/repeated-pick-place-camera` as
`episode_<index>_repeated_pick_place_16grid.mp4`. The asynchronous recorder is
not used here because its current contract only collects four rows.

The full four-family qualification uses the conservative physical profile and
four batch starts. Together they cover all 64 recipes on sixteen rows and write
64 LeRobot episodes:

```bash
python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.generation.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.showcase.m4.ultrasafe.yaml \
  --candidate-indices 0 16 32 48 --num_envs 16 --seed 7 \
  --device cpu --headless --output-dir /tmp/repeated-pick-place-showcase-m4
```

The task-local `catalog.yaml` names deployments; it does not register a Gym ID.
Generate a simulator-free local gallery from the source checkout with:

```bash
embodichain list-task --config-root embodichain_tasks=embodichain_tasks/configs/tasks --category manipulation --export-html task-gallery.html
```

For a first physical showcase, use the reduced one-family profile with four
Affordance branches and four trajectory variants. The run below executes all
16 recipes sequentially on one physical row, records one MP4 per recipe, and
concatenates them into one review video:

```bash
python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.generation.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.showcase.m1.yaml \
  --candidate-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --num_envs 1 --device cpu --headless --save-video --concat-video \
  --output-dir /tmp/repeated-pick-place-showcase-m1
```
