# EmbodiAgent

EmbodiAgent is a hierarchical multi-agent system that enables robots to perform complex manipulation tasks through closed-loop planning, graph compilation, and validation. The system combines vision-language models (VLMs) and large language models (LLMs) to translate high-level goals into executable robot actions.

## Quick Start

### Prerequisites
Ensure you have access to Azure OpenAI or a compatible LLM endpoint.

```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### Using Different LLM/VLM APIs

The system uses LangChain's `AzureChatOpenAI` by default. To use different LLM/VLM providers, you can modify the `create_llm` function in `embodichain/agents/hierarchy/llm.py`.

#### Azure OpenAI
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export OPENAI_API_VERSION="2024-10-21"  # Optional, defaults to "2024-10-21"
```

#### OpenAI
To use OpenAI directly instead of Azure, modify `llm.py`:
```python
from langchain_openai import ChatOpenAI

def create_llm(*, temperature=0.0, model="gpt-4o"):
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
```

Then set:
```bash
export OPENAI_API_KEY="your-api-key"
```

#### Other Providers
You can use other LangChain-compatible providers by modifying the `create_llm` function, for example:

**Anthropic Claude:**
```python
from langchain_anthropic import ChatAnthropic

def create_llm(*, temperature=0.0, model="claude-3-opus-20240229"):
    return ChatAnthropic(
        temperature=temperature,
        model=model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
```

**Google Gemini:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

def create_llm(*, temperature=0.0, model="gemini-pro"):
    return ChatGoogleGenerativeAI(
        temperature=temperature,
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
```

### Run the System

Run the agent system with the following command:

```bash
python embodichain/lab/scripts/run_agent.py \
    --task_name YourTask \
    --gym_config configs/gym/your_task/gym_config.json \
    --agent_config configs/gym/agent/your_agent/agent_config.json \
    --regenerate False
```

**Parameters:**
- `--task_name`: Name identifier for the task
- `--gym_config`: Path to the gym environment configuration file
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
*Located in:* `embodichain/agents/hierarchy/task_agent.py`

Responsible for high-level graph planning. It parses visual observations and
outputs a nominal atomic-action graph.

* Generates `agent_task_graph.json`
* Uses atomic actions exactly as described in the prompt context
* Leaves recovery bindings to `RecoveryAgent`

### RecoveryAgent
*Located in:* `embodichain/agents/hierarchy/recovery_agent.py`

Generates compact monitor-to-recovery bindings for the nominal graph.

* Generates `agent_recovery_spec.json`
* Uses possible external failures only as prompt context
* Outputs `recovery_bindings` only; it does not write recovery nodes, recovery edges, or recovery branches

### CompileAgent
*Located in:* `embodichain/agents/hierarchy/compile_agent.py`

Compiles generated atomic-action graph specs into an executable graph artifact.

* Expands `agent_recovery_spec.json` into explicit recovery nodes, edges, and branches
* Canonicalizes aliases and simple omissions before expansion; uses at most one LLM call only when a recovery spec contains unresolved semantic intent
* Packages the nominal task graph, recovery spec, and compiled recovery graph into `agent_compiled_graph.json`
* Executes compiled graph JSON through the graph runtime
* Does not call an LLM to generate Python control code


### ValidationAgent
*Located in:* `embodichain/agents/hierarchy/validation_agent.py`

Closes the loop by verifying if the robot actually achieved what it planned.

* Uses a specialized LLM call (`select_best_view_dir`) to analyze images from all cameras and pick the single best angle that proves the action's outcome, ignoring irrelevant views.
* If an error occurs (runtime or logic), it generates a detailed explanation which is fed back to the `TaskAgent` or `CompileAgent` for the next attempt.

---

## Configuration Guide

The `Agent` configuration block controls the context provided to the LLMs. Prompt files are resolved in the following order:

1. **Config directory**: Task-specific prompt files in the same directory as the agent configuration file (e.g., `configs/gym/agent/pour_water_agent/`)
2. **Default prompts directory**: Reusable prompt templates in `embodichain/agents/prompts/`

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
