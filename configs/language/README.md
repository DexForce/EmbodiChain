# Language Support for VLA Training

This directory contains configuration and examples for the hierarchical language support feature in EmbodiChain, enabling Vision-Language-Action (VLA) model training with Online Data Streaming (ODS).

## Overview

The language support feature adds hierarchical language descriptions to the rollout buffer, organized at three abstraction levels:

1. **Task Level**: High-level goal or overall task description
2. **Subtask Level**: Intermediate step descriptions
3. **Primitive Level**: Low-level action descriptions

This hierarchical structure enables VLA models to learn from multi-scale language representations, similar to human task understanding.

## Features

- **Multiple Language Sources**: Support for file-based, environment-based, template-based, and LLM-generated language
- **Hierarchical Structure**: Organize instructions at multiple abstraction levels
- **Flexible Storage**: Support for tokens, embeddings, or hybrid storage modes
- **Dynamic Chunk Sizes**: Works with variable-length trajectory chunks
- **Curriculum Learning**: Gradually increase language complexity during training
- **Token Agnostic**: Works with various tokenizers (GPT, BERT, etc.)

## Quick Start

### 1. Prepare Language Configuration

Create a YAML file with task descriptions:

```yaml
# tasks.yaml
pick_and_place:
  task:
    - "Pick up the red block and place it in the blue basket."

  subtask:
    - "Move the gripper to the red block."
    - "Grasp the red block."
    - "Lift the block and move to the blue basket."
    - "Release the block into the basket."

  primitive:
    - "Close gripper."
    - "Move up."
    - "Move right."
    - "Open gripper."
```

### 2. Configure ODS Engine

```python
from embodichain.agents.engine.data import OnlineDataEngine, OnlineDataEngineCfg

language_cfg = {
    "mode": "tokens",
    "hierarchy_levels": ["task", "subtask", "primitive"],
    "max_tokens": 512,
    "tokenizer": "gpt2",
    "language_source": "file",
    "language_config_path": "configs/language/tasks.yaml",
    "max_instructions_per_level": 5,
}

engine_cfg = OnlineDataEngineCfg(
    buffer_size=16,
    max_episode_steps=300,
    state_dim=14,
    gym_config={...},
    language_cfg=language_cfg,
)

engine = OnlineDataEngine(engine_cfg)
engine.start()
```

### 3. Use Language Data in Training

```python
from embodichain.agents.datasets.online_data import OnlineDataset
from torch.utils.data import DataLoader

dataset = OnlineDataset(engine, chunk_size=64, batch_size=8)
loader = DataLoader(dataset, batch_size=None)

for batch in loader:
    obs = batch["obs"]
    actions = batch["actions"]
    language = batch["language"]

    # Access language at different hierarchy levels
    task_tokens = language["task_level_tokens"]
    subtask_tokens = language["subtask_level_tokens"]
    primitive_tokens = language["primitive_level_tokens"]

    # Train your VLA model
    # loss = vla_model(obs, language, actions)
```

## Configuration Options

### Language Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "tokens" | Storage mode: 'tokens', 'embeddings', or 'hybrid' |
| `hierarchy_levels` | list | ["task", "subtask", "primitive"] | Hierarchy levels to store |
| `max_tokens` | int | 512 | Maximum sequence length per instruction |
| `tokenizer` | str | "gpt2" | Tokenizer identifier |
| `pad_token_id` | int | 0 | Token ID used for padding |
| `max_instructions_per_level` | int | 3 | Maximum number of instructions per level |
| `embedding_dim` | int | 768 | Dimension for embeddings (if mode='embeddings') |
| `language_source` | str | "env" | Source of language: 'env', 'file', 'llm', 'template' |
| `language_config_path` | str | None | Path to language config file (if source='file') |

### Language Sources

#### File-Based (`language_source: "file"`)
Load language descriptions from YAML or JSON files. Best for static task descriptions.

```python
language_cfg = {
    "language_source": "file",
    "language_config_path": "configs/language/tasks.yaml",
}
```

#### Environment-Based (`language_source: "env"`)
Generate language descriptions from the environment. The environment should implement:
- `get_task_language(task_id, context) -> HierarchicalLanguageData`
- Or have a `task_description` attribute

```python
language_cfg = {
    "language_source": "env",
}
```

#### Template-Based (`language_source: "template"`)
Use templates with variable substitution for structured tasks.

```python
language_cfg = {
    "language_source": "template",
    "templates": {
        "pick_and_place": {
            "task": "Pick up the {color} {object} and place it {location}.",
            "subtasks": [...],
        }
    },
    "variables": {"color": "red", "object": "block", "location": "in basket"},
}
```

#### LLM-Based (`language_source: "llm"`)
Generate descriptions using an LLM (e.g., GPT-4, Claude).

```python
language_cfg = {
    "language_source": "llm",
    "model": "gpt-4",
    "api_key": "your-api-key",
}
```

## Buffer Structure

When language support is enabled, the rollout buffer includes the following fields:

### Per-Hierarchy-Level Fields

For each level in `hierarchy_levels` (e.g., "task", "subtask", "primitive"):

- `{level}_tokens`: `[batch_size, max_episode_steps, max_instructions, max_tokens]`
- `{level}_attention_mask`: `[batch_size, max_episode_steps, max_instructions, max_tokens]`
- `{level}_count`: `[batch_size, max_episode_steps]`

### Global Fields

- `instruction_counts`: `[batch_size, max_episode_steps, 3]` - Counts per hierarchy level
- `change_points`: `[batch_size, max_episode_steps, max_instructions]` - Timesteps where language changes
- `hierarchy_depth`: `[batch_size, max_episode_steps]` - Current depth of hierarchy (1-3)
- `instruction_types`: `[batch_size, max_episode_steps, max_instructions]` - Instruction type IDs

## Advanced Usage

### Custom Language Provider

```python
from embodichain.lab.gym.envs.managers import LanguageProvider, HierarchicalLanguageData

class MyLanguageProvider(LanguageProvider):
    def get_language(self, task_id, context=None):
        # Generate custom language data
        return HierarchicalLanguageData(
            task_level=[...],
            subtask_level=[...],
            primitive_level=[...],
        )

    def get_available_tasks(self):
        return ["task1", "task2"]
```

### Language Augmentation

```python
from embodichain.lab.gym.envs.managers import LanguageAugmentationCfg

augmentation_cfg = LanguageAugmentationCfg(
    synonym_replacement=0.1,
    template_variation=True,
    augmentation_prob=0.5,
)
```

### Curriculum Learning

```python
from embodichain.lab.gym.envs.managers import LanguageCurriculumCfg

curriculum_cfg = LanguageCurriculumCfg(
    enabled=True,
    stage_duration=1000,
    stages=[
        # Simple language first
        LanguageCurriculumCfg.CurriculumStage(
            max_words=10,
            max_sentences=1,
            max_hierarchy_depth=1,
        ),
        # Then more complex
        LanguageCurriculumCfg.CurriculumStage(
            max_words=50,
            max_sentences=3,
            max_hierarchy_depth=3,
        ),
    ],
)
```

## Examples

See `usage_example.py` for complete examples of:
- File-based language loading
- Environment-based language generation
- Template-based language
- Dynamic chunk sizes with language
- Custom environments with language

## Files

- `tasks_example.yaml` - Example task descriptions in YAML format
- `usage_example.py` - Complete usage examples
- `README.md` - This file

## API Reference

### Core Classes

- `LanguageCfg` - Configuration for language data
- `LanguageManager` - Manages tokenization and language data
- `LanguageData` - Single-level language data
- `HierarchicalLanguageData` - Multi-level hierarchical language data
- `LanguageProvider` - Abstract base for language sources
- `FileBasedLanguageProvider` - Load from YAML/JSON files
- `LLMBasedLanguageProvider` - Generate with LLM
- `EnvBasedLanguageProvider` - Generate from environment
- `TemplateBasedLanguageProvider` - Template-based generation

## Notes

- Language data is broadcast across all timesteps in an episode
- Tokenization happens in the simulation subprocess for efficiency
- Shared memory ensures zero-copy data transfer to training process
- Compatible with all existing ODS features (dynamic chunks, etc.)
