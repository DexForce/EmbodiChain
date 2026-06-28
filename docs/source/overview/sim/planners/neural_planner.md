# NeuralPlanner

````{admonition} Experimental
:class: warning

`NeuralPlanner` is an **experimental** feature. The API, checkpoint format,
and default parameters may change without a deprecation cycle. It is currently
only validated on the **Franka Panda** robot.
````

`NeuralPlanner` is a learning-based EEF waypoint planner. It rolls out a
trained APG checkpoint through `MotionGenerator` to reach Cartesian targets.

## Configuration

Pre-trained checkpoints are hosted on HuggingFace and can be downloaded with
`download_neural_planner_checkpoint()` (requires `HF_TOKEN` environment variable).
If you already have `franka.pt`, pass it explicitly or place it at:

```text
~/.cache/embodichain_data/checkpoints/dexforce/neural_motion_generator/franka/franka.pt
```

You can also set `EMBODICHAIN_NEURAL_PLANNER_CHECKPOINT=/path/to/franka.pt`.

```python
from embodichain.data.assets.planner_assets import download_neural_planner_checkpoint
from embodichain.lab.sim.planners import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    NeuralPlannerCfg,
    PlanState,
)
from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions

checkpoint_path = download_neural_planner_checkpoint()

motion_generator = MotionGenerator(
    cfg=MotionGenCfg(
        planner_cfg=NeuralPlannerCfg(
            robot_uid=robot.uid,
            checkpoint_path=checkpoint_path,
            control_part="main_arm",
        )
    )
)

result = motion_generator.generate(
    target_states=[
        PlanState(move_type=MoveType.EEF_MOVE, xpos=waypoint)
        for waypoint in waypoints
    ],
    options=MotionGenOptions(
        plan_opts=NeuralPlanOptions(
            control_part="main_arm",
            start_qpos=start_qpos,
        ),
    ),
)
```

## Example

```bash
python examples/sim/planners/neural_planner.py --headless --device cuda
```

Use a local checkpoint without HuggingFace:

```bash
python examples/sim/planners/neural_planner.py --headless --device cuda --checkpoint-path /path/to/franka.pt
```

The example downloads the checkpoint automatically on first run only when no
local checkpoint is provided or found in the default cache path.
