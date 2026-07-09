# Action Agent Pipeline

The action-agent pipeline is the supported agent workflow for generated tabletop
manipulation tasks. It converts an image or an existing generated gym project
into a task-specific simulation config, asks the task model for a JSON task
graph, compiles that graph into atomic-action specs, and executes it through the
`AtomicActionsAgent-v3` environment.

The legacy Python-code generation agent stack has been removed. New demos and
task generation should use the modules under
`embodichain.gen_sim.action_agent_pipeline`.

## End-to-end Pipeline

Run image-to-scene, config generation, and agent execution in one command:

```bash
python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent_pipeline \
    --use-image2scene \
    --server "http://127.0.0.1:4523" \
    --image-name "demo1" \
    --task_description "Pick up the target object and place it in the basket." \
    --config-output-dir "gym_project/action_agent_pipeline/configs/demo1_text" \
    --task_name "Demo1_Text" \
    --regenerate
```

## Generate Config Only

Use an existing gym project to generate the task config and agent config:

```bash
python -m embodichain.gen_sim.action_agent_pipeline.cli.generate_action_agent_config \
    --gym_project "gym_project/environment/image2tabletop/downloads/example_gym_project" \
    --output_dir "gym_project/action_agent_pipeline/configs/demo_text" \
    --task_name "Demo_Text" \
    --task_description "Pick up the target object and place it in the basket." \
    --overwrite
```

## Run Generated Config

Run a previously generated config with the action-agent environment:

```bash
python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent \
    --task_name "Demo_Text" \
    --gym_config "gym_project/action_agent_pipeline/configs/demo_text/fast_gym_config.json" \
    --agent_config "gym_project/action_agent_pipeline/configs/demo_text/agent_config.json" \
    --regenerate
```

## Runtime Shape

- `TaskAgent` produces a deterministic JSON graph.
- `CompileAgent` caches and validates the graph artifact.
- `AgenticGenSimEnv` registers `AtomicActionsAgent-v3` and exposes
  `create_demo_action_list()`.
- Runtime graph execution calls atomic actions from
  `embodichain.gen_sim.action_agent_pipeline.runtime`.

## See Also

- [SimReady Asset Pipeline](simready_pipeline.md) — Generating simulation-ready assets
- [Atomic Actions Tutorial](../../tutorial/atomic_actions.rst) — Atomic action primitives
- [Supported Tasks](../../resources/task/index.rst) — Available task environments
