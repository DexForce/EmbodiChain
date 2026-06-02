# EmbodiAgent

EmbodiAgent is a hierarchical multi-agent system that enables robots to perform complex manipulation tasks through closed-loop planning, graph compilation, and validation. The system combines vision-language models (VLMs) and large language models (LLMs) to translate high-level goals into executable robot actions.

## Quick Start

### Prerequisites
Ensure you have access to OpenAI or an OpenAI-compatible LLM endpoint. Do not
commit credentials to the repository.

```bash
# Create a local credential file. .env is ignored by git.
cp .env.example .env
```

Fill in the local `.env` file:

```bash
OPENAI_API_KEY="<your-api-key>"
OPENAI_BASE_URL="<optional-openai-compatible-base-url>"
OPENAI_MODEL="gpt-4o"
EMBODICHAIN_LLM_PROXY=""
```

`OPENAI_BASE_URL` is optional for the default OpenAI endpoint. Set it when using
an OpenAI-compatible proxy or vendor endpoint.

### Run the System

Run the agent system with the following command:

```bash
python embodichain/lab/scripts/run_agent.py \
    --task_name YourTask \
    --gym_config configs/gym/your_task/gym_config.yaml \
    --agent_config configs/gym/agent/your_agent/agent_config.json \
    --regenerate False
```

**Parameters:**
- `--task_name`: Name identifier for the task
- `--gym_config`: Path to the gym environment configuration file (``.json``, ``.yaml``, or ``.yml``)
- `--agent_config`: Path to the agent configuration file (defines prompts and agent behavior)
- `--regenerate`: If `True`, forces regeneration of graph artifacts even if cached

## System Architecture

The system operates on a closed-loop control cycle:

- **Observe**: The `TaskAgent` perceives the environment via multi-view camera inputs.
- **Plan**: It decomposes the goal into a nominal atomic-action graph.
- **Recover**: The `RecoveryAgent` writes lightweight monitor-to-recovery bindings.
- **Compile**: The `CompileAgent` expands bindings into an executable recovery graph artifact.
- **Execute**: The graph runtime executes atomic actions and switches to recovery branches when monitors trigger.
- **Validate**: The `ValidationAgent` analyzes the result images, selects the best camera angle, and judges success.
- **Refine**: If validation fails, feedback is sent back to the agents to regenerate the graph artifacts.

---

## Core Components

### TaskAgent
*Located in:* `embodichain/gen_sim/action_agent_pipeline/agents/task_agent.py`

Responsible for high-level graph planning. It parses visual observations and
outputs a nominal atomic-action graph.

* Generates `agent_task_graph.json`
* Uses atomic actions exactly as described in the prompt context
* Leaves recovery bindings to `RecoveryAgent`

### RecoveryAgent
*Located in:* `embodichain/gen_sim/action_agent_pipeline/agents/recovery_agent.py`

Generates compact monitor-to-recovery bindings for the nominal graph.

* Generates `agent_recovery_spec.json`
* Uses possible external failures only as prompt context
* Outputs `recovery_bindings` only; it does not write recovery nodes, recovery edges, or recovery branches

### CompileAgent
*Located in:* `embodichain/gen_sim/action_agent_pipeline/agents/compile_agent.py`

Compiles generated atomic-action graph specs into an executable graph artifact.

* Expands `agent_recovery_spec.json` into explicit recovery nodes, edges, and branches
* Canonicalizes aliases and simple omissions before expansion; uses at most one LLM call only when a recovery spec contains unresolved semantic intent
* Packages the nominal task graph, recovery spec, and compiled recovery graph into `agent_compiled_graph.json`
* Executes compiled graph JSON through the graph runtime
* Does not call an LLM to generate Python control code


### ValidationAgent
*Status:* Planned validation-loop component; no implementation is currently shipped in the repository.

Closes the loop by verifying if the robot actually achieved what it planned.

* Uses a specialized LLM call (`select_best_view_dir`) to analyze images from all cameras and pick the single best angle that proves the action's outcome, ignoring irrelevant views.
* If an error occurs (runtime or logic), it generates a detailed explanation which is fed back to the `TaskAgent` or `CompileAgent` for the next attempt.

---

## Configuration Guide

The `Agent` configuration block controls the context provided to the LLMs. Prompt files are resolved in the following order:

1. **Config directory**: Task-specific prompt files in the same directory as the agent configuration file (e.g., `configs/gym/agent/pour_water_agent/`)
2. **Default prompts directory**: Reusable prompt templates in `embodichain/gen_sim/action_agent_pipeline/prompts/`

| Parameter | Description | Typical Use |
| :--- | :--- | :--- |
| `task_prompt` | Task-specific goal description | "Pour water from the red cup to the blue cup." |
| `basic_background` | Physical rules & constraints | World coordinate system definitions, safety rules. |
| `atom_actions` | API Documentation | List of available functions (e.g., `drive(action='pick', ...)`). |
| `error_functions` | Failure context only | Possible external failure descriptions for recovery spec design. |
| `monitor_functions` | Monitor API documentation | Available runtime monitor functions. |
| `recovery_rules` | Recovery spec rules | Constraints for lightweight monitor and recovery bindings. |

---

## File Structure

```text
embodichain/agents/
├── hierarchy/
│   ├── agent_base.py          # Abstract base handling prompts & images
│   ├── task_agent.py          # Nominal graph generation logic
│   ├── recovery_agent.py      # Recovery graph generation logic
│   ├── compile_agent.py       # Graph compilation & execution interface
│   ├── validation_agent.py    # Visual analysis & view selection
│   └── llm.py                 # LLM configuration and instances
├── mllm/
│   └── prompt/                # Prompt templates (LangChain)
└── prompts/                   # Agent prompt templates
```

---

## See Also

- [Online Data Streaming](../online_data.md) — Streaming live simulation data for training
- [RL Architecture](../../overview/rl/index.rst) — RL training pipeline and algorithms
- [Atomic Actions Tutorial](../../tutorial/atomic_actions.rst) — Action primitives used by the CodeAgent
- [Supported Tasks](../../resources/task/index.rst) — Available task environments
