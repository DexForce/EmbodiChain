# Robot Challenge Tasks

A collection of robot manipulation tasks for [Robot Challenge](https://robochallenge.ai/). This project follows the EmbodiChain framework structure and focuses on environment construction and demonstration tasks.

> [!NOTE]
> The current tasks are from [Table-30](https://robochallenge.ai/benchmark_detail). More details about the tasks and their configurations can be found in the provided link.

## Project Structure

```
robot_challenge_tasks/
├── README.md
├── pyproject.toml        # Project configuration and dependencies
├── configs/              # Task configuration files
│   └── demo/             # Use one folder for each task
│       └── dummy.json    # Put all gym config and action config into the folder. 
└── tasks/                # Task implementations
    ├── __init__.py
    └── basic/
        ├── __init__.py
        └── dummy_task.py
```

## Installation

Install EmbodiChain in development mode to use the latest features.

```python
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
```

Then install the `robot_challenge_tasks` package in development mode.

```bash
# Install in development mode
cd robot_challenge_tasks
pip install -e . 
```

## Quick Start

### Implement a Task

1. Create a new task environment class in `tasks/{task_name}.py` that inherits from `EmbodiedEnv`.
2. Create a configuration file in `configs/{task_name}/xxx.json` that defines the environment and robot setup.
3. Implement the `create_demo_action_list()` method in your task environment to generate demonstration actions based on the task requirements.

#### Important Notes

If you are implementing a digital twin of a real-world task (e.g., a task in Table-30), it is recommended to follow the steps below to ensure the accuracy of the simulation environment:

1. Use sim-ready assets to construct the simulation environment.

> [!NOTE] Currently, a sim-ready asset should have the following properties at least:
> - Accurate geometry and dimensions.
> - Correct coordinate system and origin point.
> - Reasonable number of vertices (not too high for real-time simulation, and not too low to lose important details).
> - Properly defined visual materials and textures (if necessary for the task).

> We may use USD format for sim-ready assets in the future, which can provide more standardized and comprehensive support for the above properties.

2. Replay real demonstration data in the simulation environment to check the feasibility and accuracy of the environment setup. You can use the utilities provided in this repository to facilitate the replay process.

### Running a Task

```python
# Launch the environment in data generation mode.
python scripts/run_env.py \
    --gym_config configs/demo/dummy.json \
    ...

# Launch the environment in preview mode.
python scripts/run_env.py \
    --gym_config configs/demo/dummy.json \
    --preview \
    ...
```

After finishing the data generation, you can find the saved dataset in `~/.cache/embodichain_datasets/` by default. You can also specify a custom path for saving the dataset in your config file.

```json
{
    env: {
        "dataset": {
            "lerobot": {
                "func": "LeRobotRecorder",
                "mode": "save",
                "params": {
                    "save_path": "/your/custom/path",
                    ...
                }
            }
        }
    }
}
```

The following command-line arguments are commonly used when running the environment:

- `--enable_rt`: Enable ray tracing rendering backend. (recommended used for most of case)
- `--headless`: Run the environment in headless mode. (must be used on servers without display)
- `--filter_dataset_saving`: Prevent saving dataset for episodes. This argument is used for debugging and testing purposes.

## Useful Utilities

### Configure camera pose with keyboard control

Run the env in `preview` mode, and execute the following code snippet in the Python console to control the camera pose with keyboard input. Once you are satisfied with the camera pose, press `p` to print the pose in the console, and you can copy the printed pose to your config file.

```python
from embodichain.lab.sim.utility.keyboard_utils import run_keyboard_control_for_camera

run_keyboard_control_for_camera(cam_uid, vis_pose=True)
```

### Configure light conditions with keyboard control

Run the env in `preview` mode, and execute the following code snippet in the Python console to control the light conditions with keyboard input. Once you are satisfied with the light conditions, press `p` to print the light configuration in the console, and you can copy the printed configuration to your config file.

```python
from embodichain.lab.sim.utility.keyboard_utils import run_keyboard_control_for_light

run_keyboard_control_for_light(light_uid, vis_config=True)
```

### Use gizmo to control robot for feasibility check

Run the env in `preview` mode, and execute the following code snippet in the Python console to control the robot with gizmo. This is useful for checking the feasibility of a task. Use `p` to print the current robot state (joint positions, end-effector pose, etc.) in the console, which can be used as a reference for generating demonstration actions.


```python
from embodichain.lab.sim.utility.gizmo_utils import run_gizmo_robot_control_loop
robot = env.get_wrapper_attr("robot")
run_gizmo_robot_control_loop(robot, control_part=part)
```

### Replay real data (robot states) in simulation environment

You can replay real data (e.g., robot states) in the simulation environment for visualization and analysis. 

1. Download a specific task dataset from [Robot Challenge](https://huggingface.co/RoboChallenge).

2. You may use `cat stack_bowls.tar-aa stack_bowls.tar-ab | tar xvf -` to extract the downloaded dataset if it is split into multiple parts.

3. Use the following code snippet to load the dataset and replay the robot states in the simulation environment.

```
python scripts/replay_real_data.py \
    --gym_config configs/{your_task}/xxx.json \
    --data_path /path/to/your/downloaded/dataset \
    --episode_id {your_episode_id} \
    --enable_rt \
```