# NeuralPlanner

````{admonition} Experimental
:class: warning

`NeuralPlanner` is an **experimental** feature. The API, ONNX model contract,
and default parameters may change without a deprecation cycle. It is currently
only validated on the **Franka Panda** robot.
````

`NeuralPlanner` is a learning-based EEF waypoint planner. It rolls out a
standalone NMG ONNX policy through `MotionGenerator` to reach Cartesian targets.
The ONNX graph must include raw-observation normalization.

## Configuration

Install the optional runtime and export the trained policy to ONNX before use:

```bash
pip install -e '.[nmg]'
```

```python
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import MoveType, NeuralPlannerCfg, PlanState
from embodichain.lab.sim.motion.planners.neural_planner import NeuralPlanOptions

onnx_model_path = "/path/to/best_mean.onnx"

motion_generator = MotionGenerator(
    cfg=MotionGenCfg(
        planner_cfg=NeuralPlannerCfg(
            robot_uid=robot.uid,
            onnx_model_path=onnx_model_path,
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
python examples/sim/motion/planners/neural_planner.py \
  --headless --device cuda \
  --onnx-model-path /path/to/best_mean.onnx
```

The NMG exporter produces a dynamic-batch ONNX policy, so one graph can serve
single-env and env-batched rollout. If the runtime robot base frame or TCP differs
from training, configure `policy_frame_from_world` and
`runtime_tcp_from_policy_tcp` as explicit homogeneous transforms. The conversion is

```text
policy_T_policy_tcp = policy_T_world
                    @ world_T_runtime_tcp
                    @ runtime_tcp_T_policy_tcp
```

Pose targets and FK observations inside `NeuralPlanner` use EmbodiChain's
`xyz + xyzw` convention. `quat_from_matrix()` already returns `xyzw`; no
additional quaternion reordering is required before the ONNX observation is
assembled. Conversions to `wxyz` belong only at an explicitly documented
external-library boundary.
