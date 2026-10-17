(task-programs)=

# Task Program

Task Program lets you customize an expert demonstration by describing **what
objects to pick, where to place them, and in what order**. Reuse an existing
scene and robot, edit a YAML program, and run it through the normal environment
launcher. You do not need to write a motion planner or an integration to start.

**Start with {doc}`quickstart`: run a Franka example, change two moves, check the
cube's resting position, and save one trajectory.**

```{toctree}
:hidden:

quickstart
recipes
troubleshooting
validation
architecture
scene_registry
robot_profiles
```

## What can I make?

The official repeated Pick/Place program moves a cube between two targets.
Its `repeat` node chooses A → B → A; each repetition contains a Pick segment
and a Place segment. You can change the repetition count, targets, step order,
and language descriptions in **one local `program.yaml`**. An existing deployment
supplies the scene, robot, available calls and execution settings.

The {doc}`quickstart` uses `repeated_pick_place/task.franka.yaml`, a single
Franka Panda in the Default physics environment, without trajectory expansion.
The complete original file is available as a
{download}`download <../../../../extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task_program/program.yaml>`.

## What do I need?

- An installed SDK and compatible DexSim/GPU environment: {doc}`/quick_start/install`.
- The selected deployment's robot assets: {doc}`/resources/assets`.
- A text editor and a directory for your program and outputs.

The tutorial uses an installed package reference for the deployment. A source
checkout is only needed for the separate maintainer checks.

## Choose a route

| Your goal | Start here | Continue with |
|---|---|---|
| Customize a trajectory in an existing task | {doc}`quickstart` | {doc}`recipes`, then {doc}`troubleshooting` |
| Learn the language without starting simulation | {doc}`/tutorial/task_program_python` | {doc}`architecture` |
| Bind a new scene, robot or skill | {doc}`/tutorial/task_program` | {doc}`scene_registry`, {doc}`robot_profiles` |
| Understand the compiler and runtime | {doc}`architecture` | {doc}`/api_reference/embodichain/embodichain.lab.task_program` |

## How do I know it worked?

**Command execution completed** means the requested motion finished. **Measured
checks passed** means the observed state satisfied the checks you wrote. They
are different results: the example's `projected` policy updates expected task
state after motion, without measuring every grasp or release.

In the Quickstart you add a wait for stability and a position check after each
Place. Inspect those results as well as episode completion, and confirm that
the saved episode can be read. Passing these checks establishes position and
stability for that run; it does not establish every possible physical property.
See [assurance and acceptance](architecture.md#effect-assurance-is-mandatory).

## What stays with the integration?

Programs can only name objects and calls supported by their selected integration.
Changing an `instruction` changes the description and language annotation;
the structured calls determine the actions. A new object, robot, unregistered
call or different motion policy needs integration/deployment work. Follow
{doc}`/tutorial/task_program` for those changes.

For later workflows, see {doc}`/guides/offline_expert_datasets` for recording and
consumption, {doc}`/guides/run_env` for expansion and launcher options, and
{doc}`/features/generative_sim/index` for scene generation. These are optional follow-ups.

## Previous architecture sections

The implementation reference has moved to {doc}`architecture`. Existing section
links remain available below.

<a id="ownership-boundaries"></a>

[Ownership boundaries](architecture.md#ownership-boundaries)

<a id="semantic-layer"></a>

[Semantic layer](architecture.md#semantic-layer)

<a id="program-schema"></a>

[Program schema](architecture.md#program-schema)

<a id="load-and-compile"></a>

[Load and compile](architecture.md#load-and-compile)

<a id="configure-a-simulation-environment"></a>

[Configure a simulation environment](architecture.md#configure-a-simulation-environment)

<a id="runtime-lifecycle"></a>

[Runtime lifecycle](architecture.md#runtime-lifecycle)

<a id="effect-assurance-is-mandatory"></a>

[Effect assurance is mandatory](architecture.md#effect-assurance-is-mandatory)

<a id="parallel-programs"></a>

[Parallel programs](architecture.md#parallel-programs)

<a id="registered-calls"></a>

[Registered calls](architecture.md#registered-calls)

<a id="reference-tasks"></a>

[Reference tasks](architecture.md#reference-tasks)

<a id="recommended-change-sites"></a>

[Recommended change sites](architecture.md#recommended-change-sites)

<a id="focused-validation"></a>

[Focused validation](architecture.md#focused-validation)

<a id="further-reading"></a>

[Further reading](architecture.md#further-reading)

<a id="embodied-task-program"></a>

[Embodied Task Program architecture](architecture.md)
