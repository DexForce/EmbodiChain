# W1 cloth fold

This Task Program runs a fixed-scene, one-way Newton fold. The program declares
four phases. The W1 integration binds them to ranges of one qualified joint
stream, including the grippers. Newton interpolates the stream inside each
physics step. The Gym bridge still owns phase execution and completion.

Set `EMBODICHAIN_DATA_ROOT` to the `inputs` directory extracted from the
[M02 input bundle](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/cafe3e14f650f93123d47fe338e49c003443f3a4/reports/dexuni-physics-2026-10-27/inputs/m02_cloth_single_fold/m02_cloth_single_fold.zip).
The deployment checks the cloth mesh and W1 URDF against the qualified inputs.
The wheel includes the placement-adjusted command stream. Its float32 position
array has SHA-256
`6a71627357c1a5729204fecdcc7be5e690ca5ad4221f8da40beb4210d4728a2e`.

Use the installed task package and its standard Gym Task Program executor:

```python
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import make
from embodichain.lab.sim import SimulationManager
from embodichain.utils.utility import load_config
from embodichain_tasks.configs import get_config_path

path = get_config_path("tasks/manipulation/cloth_fold/task.w1.yaml")
cfg = config_to_cfg(load_config(path), source_path=path)
env = make("TaskProgramClothFold-W1-v1", cfg=cfg)
try:
    env.reset(seed=0)
    result = execute_demo_episode(env)
    print(result.completed, result.all_success)
finally:
    env.close(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

The deployment uses one CUDA arena at 60 Hz, ten substeps, `surface-fast` at
16 iterations and CUDA Graph disabled. The final success query reads the
current particle positions and checks that both paired free-edge corner
centroids are within 15 mm in XY after the program completes.

The corner tolerance and particle pairs live in `task_program/evaluation.yaml`
as ordinary shared `near_target` final rules.
The physical `simulation.replay` declaration owns the command file, digest and
cadence. Gym reads it once, registers native controls before physics preparation,
and shares the bound command snapshot with the phase lowerer. This replay keeps
cloth physics enabled and requires position-only commands; additional velocity
or effort targets are rejected. Its cadence is checked after resolving
`target_control_frequency` or `sim_steps_per_control`. Episode metadata includes
the source digest, joint orders and rate.

Mesh-qualified world-centroid samples are captured once per query and sealed in
`final_acceptance_observations`; offline replay uses the same measured criteria.
Program completion additionally requires reaching the final replay frame. Invalid
XYZ measurements are unavailable evidence; the XY distance limit stays inclusive.
The earlier native qualification matrix predates the shared-final integration and
is not a new native qualification of this synchronization.
