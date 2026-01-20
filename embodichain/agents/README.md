# EmbodiAgent System


## Quick Start

### 1. Prerequisites
Ensure you have access to Azure OpenAI or a compatible LLM endpoint.

```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT="[https://your-endpoint.openai.azure.com/](https://your-endpoint.openai.azure.com/)"
export AZURE_OPENAI_API_KEY="your-api-key"
```


### 2. Run the System

```bash
python embodichain/lab/scripts/run_agent.py \
    --task_name YourTask \
    --gym_config configs/gym/your_task/gym_config.json \
    --agent_config configs/gym/agent/your_agent/agent_config.json \
    --regenerate False
```



## System Architecture

The system operates on a closed-loop control cycle:

1.  **Observe**: The `TaskAgent` perceives the environment via multi-view camera inputs.
2.  **Plan**: It decomposes the goal into natural language steps.
3.  **Code**: The `CodeAgent` translates steps into executable Python code using atomic actions.
4.  **Execute**: The code runs in the environment; runtime errors are caught immediately.
5.  **Validate**: The `ValidationAgent` analyzes the result images, selects the best camera angle, and judges success.
6.  **Refine**: If validation fails, feedback is sent back to the agents to regenerate the plan or code.


---

## Core Components

### 1. TaskAgent ("The Planner")
*Located in:* `embodichain/agents/hierarchy/task_agent.py`

Responsible for high-level reasoning. It parses visual observations and outputs a structured plan.

* For every step, it generates a specific condition (e.g., "The cup must be held by the gripper") which is used later by the ValidationAgent.
* Prompt Strategies:
    * `one_stage_prompt`: Direct VLM-to-Plan generation.
    * `two_stage_prompt`: Separates visual analysis from planning logic.

### 2. CodeAgent ("The Coder")
*Located in:* `embodichain/agents/hierarchy/code_agent.py`

Translates natural language plans into executable Python code.


### 3. ValidationAgent ("The Judger")
*Located in:* `embodichain/agents/hierarchy/validation_agent.py`

Closes the loop by verifying if the robot actually achieved what it planned.

* Uses a specialized LLM call (`select_best_view_dir`) to analyze images from all cameras and pick the single best angle that proves the action's outcome, ignoring irrelevant views.
* If an error occurs (runtime or logic), it generates a detailed explanation which is fed back to the `TaskAgent` or `CodeAgent` for the next attempt.

---

## Configuration Guide

The `Agent` configuration block controls the context provided to the LLMs. All files are resolved relative to `embodichain/database/agent_prompt/`.

| Parameter | Description | Typical Use |
| :--- | :--- | :--- |
| `task_prompt` | Task-specific goal description | "Pour water from the red cup to the blue cup." |
| `basic_background` | Physical rules & constraints | World coordinate system definitions, safety rules. |
| `atom_actions` | API Documentation | List of available functions (e.g., `drive(action='pick', ...)`). |
| `code_prompt` | Coding guidelines | "Use provided APIs only. Do not use loops." |
| `code_example` | Few-shot examples | Previous successful code snippets to guide style. |

---

## File Structure

```text
embodichain/agents/
├── hierarchy/
│   ├── agent_base.py          # Abstract base handling prompts & images
│   ├── task_agent.py          # Plan generation logic
│   ├── code_agent.py          # Code generation & AST execution engine
│   ├── validation_agent.py    # Visual analysis & view selection
│   └── llm.py                 # LLM configuration and instances
├── mllm/
│   └── prompt/                # Prompt templates (LangChain)
└── README.md                  # This file
```
