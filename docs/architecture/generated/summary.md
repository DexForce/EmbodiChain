# Architecture overview

Source revision: `dc2e78c0568a83177b33e3918faa016cbfd9a4d9` · package 0.2.4

Curated static relationships; not a complete dependency graph or runtime trace.

| Module | Responsibility | Evidence and documentation |
| --- | --- | --- |
| CLI | Dispatches commands and lazily loads task discovery, environment, and training entry points. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/cli/main.py#L17-L17) |
| Official tasks | Registers official task entry points and organises reusable robot tasks. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain_tasks/embodichain_tasks/__init__.py#L17-L23) |
| Component composition | Resolves environment and embodiment components into runnable Gym configurations. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/utils/_component_composition.py#L17-L17) |
| EmbodiedEnv | Assembles scenes, managers, and demonstrations with optional Task Program integration. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/envs/embodied_env.py#L284-L284) · [EmbodiedEnv documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/overview/gym/env.md) |
| Managers | Organises observations, actions, rewards, events, and data recording within the environment lifecycle. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/envs/managers/__init__.py#L17-L20) |
| TaskProgramCompiler | Compiles task configuration into an analysable program using a static scene manifest. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/task_program/compiler/program.py#L1179-L1179) · [TaskProgramCompiler documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.task_program.compiler.rst) |
| SemanticCallCompiler | Analyses and lowers Semantic Calls against the bound scene and skill integration. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/task_program/compiler/lowering.py#L767-L767) · [SemanticCallCompiler documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.task_program.compiler.rst) |
| SemanticCallExecutor | Lowers and executes semantic calls, creating an execution session for each call. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/task_program/runtime/executor.py#L216-L216) · [SemanticCallExecutor documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.task_program.runtime.rst) |
| TaskProgramEnvironmentAdapter | Validates integration choices and assembles the semantic runtime and Gym bridge. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/task_program/integrations/environment.py#L295-L295) · [TaskProgramEnvironmentAdapter documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.task_program.integrations.rst) |
| TaskProgramDemoBridge | Converts program segments into lazy Gym demonstrations with a shared runtime, buffer, and clock. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/envs/task_program/bridge.py#L951-L951) · [TaskProgramDemoBridge documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.gym.envs.task_program.rst) |
| AtomicActionEngine | Manages planning resources and atomic skills to create executable action sessions. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/atomic_actions/engine.py#L45-L45) · [AtomicActionEngine documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst) |
| MotionGenerator | Composes planning backends behind a unified robot motion generation interface. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/motion/motion_generator.py#L191-L191) · [MotionGenerator documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/api_reference/embodichain/embodichain.lab.sim.motion.motion_generator.rst) |
| Planners | Provides motion planning backends that generate constrained robot trajectories. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/motion/planners/__init__.py#L17-L22) |
| Solvers | Provides inverse kinematics backends and stateful robot solving capabilities. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/motion/solvers/__init__.py#L17-L20) |
| SimulationManager | Manages the world, robots, sensors, physics, and visualisation lifecycle. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/sim_manager.py#L481-L481) · [SimulationManager documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/overview/sim/sim_manager.md) |
| Robot | Represents a simulated robot with control parts, joint state, and kinematics. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/objects/robot.py#L17-L17) |
| Sensors | Provides camera, stereo, and contact observations for the environment. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/sensors/__init__.py#L17-L20) |
| Physics backends | Constructs the configured physics backend and connects simulation state to physics solving. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/sim/physics/__init__.py#L16-L21) |
| Browser visualization | Publishes simulation snapshots and interactive controls through a browser runtime. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/visualization/runtime.py#L17-L17) |
| Compute | Shared numerical algorithms for kinematics, trajectories, geometry, and images. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/compute/__init__.py#L17-L21) |
| RL training | Organises policy learning, collection, evaluation, and model persistence. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/learning/rl/utils/trainer.py#L17-L17) |
| Data pipeline | Online data engine managing workers and their lifecycle. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/data_pipeline/engine/data.py#L17-L17) |
| Scene generation | Generates scene state from input images for scene asset workflows. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/gen_sim/scene_engine/pipeline/generate.py#L17-L17) |
| Devices | Base interfaces for real robot controllers. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/devices/device.py#L17-L17) |
| Environment launcher | Runs task discovery, configuration loading, and Gym environment creation. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/scripts/run_env.py#L994-L994) |
| Config loader | Loads deployment files, resolves components, and assembles the typed environment configuration. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/utils/gym_utils.py#L1210-L1210) |
| Task discovery | Imports installed task packages through entry points to populate the environment registry. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/utils/registration.py#L430-L430) |
| BaseEnv | Owns the Gym loop, simulation initialisation, and environment step lifecycle. | [Source](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/embodichain/lab/gym/envs/base_env.py#L103-L103) · [BaseEnv documentation](https://github.com/DexForce/EmbodiChain/blob/dc2e78c0568a83177b33e3918faa016cbfd9a4d9/docs/source/overview/gym/env.md) |

## Coverage and interpretation

Snapshot: 34 nodes, 36 source-reviewed relationships, and 29 statically extracted relationships across all views.

Import declarations do not establish runtime calls or execution order. Missing edges do not imply architectural independence.

Topics without represented nodes: Differentiable Environment (APG), Configclass Pattern, Domain Randomization, Data Assets, Robot Workspace. Topic representation does not imply complete coverage.

Overview nodes without mapped relationships: Compute, Data pipeline, Scene generation, Devices.

- A curated source snapshot, not a complete import graph, runtime trace, or execution order.
- The sequential Gym integration is emphasized. Parallel execution, recovery, validators, recording, and planner internals are not exhaustively modeled.
- References through protocols and injected factories are scoped to the described production assembly.
- Source-reviewed means checked against the pinned source revision; it is not a physical qualification result.
- Prototype overview imports cover selected source modules only. They do not establish runtime ordering or a complete dependency graph.
