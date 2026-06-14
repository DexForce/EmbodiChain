# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

import yaml
import json

from embodichain.utils.logger import log_info, log_warning, log_error
from .language import (
    LanguageCfg,
    HierarchicalLanguageData,
    LanguageData,
)

__all__ = [
    "LanguageProvider",
    "FileBasedLanguageProvider",
    "LLMBasedLanguageProvider",
    "EnvBasedLanguageProvider",
    "TemplateBasedLanguageProvider",
]


class LanguageProvider(ABC):
    """Abstract base class for language data sources.

    Language providers are responsible for generating or retrieving
    hierarchical language descriptions for tasks. Different providers
    can be used depending on the data source (files, LLMs, environment, etc.).

    Args:
        cfg: Language configuration.
    """

    def __init__(self, cfg: LanguageCfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def get_language(
        self, task_id: str, context: Optional[Dict[str, Any]] = None
    ) -> HierarchicalLanguageData:
        """Get hierarchical language data for a specific task.

        Args:
            task_id: Unique identifier for the task.
            context: Optional context dictionary with environment state.

        Returns:
            HierarchicalLanguageData with task descriptions at multiple levels.
        """
        ...

    @abstractmethod
    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs.

        Returns:
            List of task identifiers.
        """
        ...

    def validate_hierarchy_data(self, data: HierarchicalLanguageData) -> bool:
        """Validate that hierarchical language data meets configuration constraints.

        Args:
            data: HierarchicalLanguageData to validate.

        Returns:
            True if data is valid, False otherwise.
        """
        # Check each level doesn't exceed max instructions
        if len(data.task_level) > self.cfg.max_instructions_per_level:
            log_warning(
                f"Task level has {len(data.task_level)} instructions, "
                f"exceeding max {self.cfg.max_instructions_per_level}"
            )
            return False

        if len(data.subtask_level) > self.cfg.max_instructions_per_level:
            log_warning(
                f"Subtask level has {len(data.subtask_level)} instructions, "
                f"exceeding max {self.cfg.max_instructions_per_level}"
            )
            return False

        if len(data.primitive_level) > self.cfg.max_instructions_per_level:
            log_warning(
                f"Primitive level has {len(data.primitive_level)} instructions, "
                f"exceeding max {self.cfg.max_instructions_per_level}"
            )
            return False

        return True


class FileBasedLanguageProvider(LanguageProvider):
    """Language provider that loads task descriptions from files.

    Supports YAML and JSON file formats. The file structure should contain
    task IDs mapped to their hierarchical descriptions.

    Example YAML structure:
        ```yaml
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

    Args:
        cfg: Language configuration.
        config_path: Path to the configuration file (YAML or JSON).
        reload_on_access: Whether to reload the file on each access (for dynamic updates).
    """

    def __init__(
        self,
        cfg: LanguageCfg,
        config_path: str,
        reload_on_access: bool = False,
    ) -> None:
        super().__init__(cfg)
        self.config_path = Path(config_path)
        self.reload_on_access = reload_on_access
        self._data: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load language data from the configuration file."""
        if not self.config_path.exists():
            log_error(
                f"Language config file not found: {self.config_path}",
                error_type=FileNotFoundError,
            )

        suffix = self.config_path.suffix.lower()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                if suffix in [".yaml", ".yml"]:
                    self._data = yaml.safe_load(f)
                elif suffix == ".json":
                    self._data = json.load(f)
                else:
                    log_error(
                        f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json",
                        error_type=ValueError,
                    )

            log_info(
                f"[FileBasedLanguageProvider] Loaded {len(self._data)} task descriptions "
                f"from {self.config_path}"
            )
        except Exception as e:
            log_error(
                f"Failed to load language config from {self.config_path}: {e}",
                error_type=RuntimeError,
            )

    def get_language(
        self, task_id: str, context: Optional[Dict[str, Any]] = None
    ) -> HierarchicalLanguageData:
        """Get language data from file for a specific task.

        Args:
            task_id: Unique identifier for the task.
            context: Optional context (not used in file-based provider).

        Returns:
            HierarchicalLanguageData loaded from file.
        """
        if self.reload_on_access:
            self._load_data()

        if task_id not in self._data:
            log_error(
                f"Task ID '{task_id}' not found in language config. "
                f"Available tasks: {list(self._data.keys())}",
                error_type=KeyError,
            )

        task_data = self._data[task_id]

        # Extract hierarchical descriptions
        task_texts = task_data.get("task", [])
        subtask_texts = task_data.get("subtask", [])
        primitive_texts = task_data.get("primitive", [])
        change_points = task_data.get("change_points", None)

        # Import LanguageManager to create data (we need tokenizer access)
        from .language import LanguageManager

        # Create a temporary manager for tokenization
        # In practice, the environment should provide the manager
        class _TempManager:
            def __init__(self, cfg):
                self.cfg = cfg
                self._tokenizer = None
                self._load_tokenizer()

            def _load_tokenizer(self):
                if self.cfg.tokenizer_backend == "huggingface":
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.cfg.tokenizer,
                        trust_remote_code=self.cfg.trust_remote_code,
                    )
                    if (
                        self.cfg.pad_token_id == 0
                        and self._tokenizer.pad_token_id is not None
                    ):
                        self.cfg.pad_token_id = self._tokenizer.pad_token_id
                else:
                    import tiktoken

                    self._tokenizer = tiktoken.encoding_for_model(self.cfg.tokenizer)

            def tokenize(self, text):
                if self.cfg.tokenizer_backend == "huggingface":
                    result = self._tokenizer(
                        text,
                        max_length=self.cfg.max_tokens,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    return result["input_ids"].squeeze(0).to(torch.int64), result[
                        "attention_mask"
                    ].squeeze(0).to(torch.int64)
                else:
                    import torch

                    tokens = self._tokenizer.encode(
                        text, max_length=self.cfg.max_tokens, truncation=True
                    )
                    if len(tokens) < self.cfg.max_tokens:
                        tokens = tokens + [self.cfg.pad_token_id] * (
                            self.cfg.max_tokens - len(tokens)
                        )
                    else:
                        tokens = tokens[: self.cfg.max_tokens]
                    input_ids = torch.tensor(tokens, dtype=torch.int64)
                    attention_mask = (input_ids != self.cfg.pad_token_id).to(
                        torch.int64
                    )
                    return input_ids, attention_mask

            def create_language_data(self, text):
                tokens, mask = self.tokenize(text)
                return LanguageData(tokens=tokens, attention_mask=mask, raw_text=text)

        temp_mgr = _TempManager(self.cfg)

        # Build hierarchical language data
        task_level = [
            temp_mgr.create_language_data(t) if isinstance(t, str) else t
            for t in (task_texts if isinstance(task_texts, list) else [task_texts])
        ]
        subtask_level = (
            [
                temp_mgr.create_language_data(t) if isinstance(t, str) else t
                for t in (subtask_texts if isinstance(subtask_texts, list) else [])
            ]
            if subtask_texts
            else []
        )
        primitive_level = (
            [
                temp_mgr.create_language_data(t) if isinstance(t, str) else t
                for t in (primitive_texts if isinstance(primitive_texts, list) else [])
            ]
            if primitive_texts
            else []
        )

        return HierarchicalLanguageData(
            task_level=task_level,
            subtask_level=subtask_level,
            primitive_level=primitive_level,
            change_points=change_points,
        )

    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs from the file.

        Returns:
            List of task identifiers.
        """
        return list(self._data.keys())


class LLMBasedLanguageProvider(LanguageProvider):
    """Language provider that generates descriptions using an LLM.

    This provider uses a language model to generate task descriptions
    on-the-fly based on task context and templates.

    Args:
        cfg: Language configuration.
        model: Model identifier (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for the LLM service.
        templates: Optional dictionary of templates for different task types.
    """

    def __init__(
        self,
        cfg: LanguageCfg,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        templates: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg)
        self.model = model
        self.api_key = api_key
        self.templates = templates or self._default_templates()
        self._client = None
        self._init_client()

    def _default_templates(self) -> Dict[str, str]:
        """Default prompt templates for language generation."""
        return {
            "task": "Generate a clear, concise task description for: {task_name}.",
            "subtask": "Break down the task '{task_name}' into {num_steps} step-by-step instructions.",
            "primitive": "For each subtask, provide low-level action descriptions in: {task_name}.",
        }

    def _init_client(self) -> None:
        """Initialize the LLM client based on model type."""
        if self.model.startswith("gpt"):
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                log_warning(
                    "openai library not available. LLM provider will use fallback."
                )
        elif self.model.startswith("claude"):
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                log_warning(
                    "anthropic library not available. LLM provider will use fallback."
                )
        else:
            log_warning(
                f"Unknown model type: {self.model}. LLM provider will use fallback."
            )

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            Generated text string.
        """
        if self._client is None:
            # Fallback: return a generic response
            log_warning("LLM client not available, using fallback response.")
            return "Complete the task as described in the environment."

        try:
            if self.model.startswith("gpt"):
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            elif self.model.startswith("claude"):
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
        except Exception as e:
            log_warning(f"LLM generation failed: {e}. Using fallback.")
            return "Complete the task as described in the environment."

    def get_language(
        self, task_id: str, context: Optional[Dict[str, Any]] = None
    ) -> HierarchicalLanguageData:
        """Generate language data using LLM for a specific task.

        Args:
            task_id: Unique identifier for the task.
            context: Optional context with task details.

        Returns:
            HierarchicalLanguageData generated by LLM.
        """
        task_name = context.get("task_name", task_id) if context else task_id

        # Generate task-level description
        task_prompt = self.templates["task"].format(task_name=task_name)
        task_text = self._generate_with_llm(task_prompt)

        # Generate subtask-level descriptions
        num_subtasks = context.get("num_subtasks", 3) if context else 3
        subtask_prompt = self.templates["subtask"].format(
            task_name=task_name, num_steps=num_subtasks
        )
        subtask_text = self._generate_with_llm(subtask_prompt)
        subtask_texts = [
            line.strip() for line in subtask_text.split("\n") if line.strip()
        ]

        # Generate primitive-level descriptions (optional)
        primitive_texts = []
        if context and context.get("include_primitive", False):
            primitive_prompt = self.templates["primitive"].format(task_name=task_name)
            primitive_text = self._generate_with_llm(primitive_prompt)
            primitive_texts = [
                line.strip() for line in primitive_text.split("\n") if line.strip()
            ]

        # Create LanguageData objects (would need LanguageManager in practice)
        # This is a simplified version - in production, use LanguageManager
        return HierarchicalLanguageData(
            task_level=[],  # Would be populated with LanguageData objects
            subtask_level=[],
            primitive_level=[],
        )

    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs.

        For LLM provider, this returns an empty list as tasks are
        generated on-the-fly.

        Returns:
            Empty list (tasks are generated dynamically).
        """
        return []


class EnvBasedLanguageProvider(LanguageProvider):
    """Language provider that extracts descriptions from the environment.

    This provider delegates language generation to the environment itself,
    allowing task-specific implementations to provide custom logic.

    Args:
        cfg: Language configuration.
        env: The environment instance.
    """

    def __init__(self, cfg: LanguageCfg, env) -> None:
        super().__init__(cfg)
        self.env = env

    def get_language(
        self, task_id: str, context: Optional[Dict[str, Any]] = None
    ) -> HierarchicalLanguageData:
        """Get language data from the environment.

        The environment should implement one of:
        - get_task_language(task_id, context) -> HierarchicalLanguageData
        - task_description attribute (simple string)
        - generate_task_description() method

        Args:
            task_id: Unique identifier for the task.
            context: Optional context dictionary.

        Returns:
            HierarchicalLanguageData from the environment.
        """
        # Check for dedicated method
        if hasattr(self.env, "get_task_language"):
            return self.env.get_task_language(task_id, context)

        # Check for attribute
        if hasattr(self.env, "task_description"):
            task_desc = self.env.task_description
            # Would need LanguageManager to tokenize
            return HierarchicalLanguageData(
                task_level=[],  # Would be populated
                subtask_level=[],
                primitive_level=[],
            )

        # Check for method
        if hasattr(self.env, "generate_task_description"):
            task_desc = self.env.generate_task_description(context)
            return HierarchicalLanguageData(
                task_level=[],
                subtask_level=[],
                primitive_level=[],
            )

        log_error(
            "Environment does not provide language data. "
            "Implement get_task_language, set task_description attribute, or generate_task_description method.",
            error_type=NotImplementedError,
        )

    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs from the environment.

        The environment can optionally provide:
        - available_tasks attribute
        - get_available_tasks() method

        Returns:
            List of task identifiers or empty list.
        """
        if hasattr(self.env, "available_tasks"):
            return self.env.available_tasks

        if hasattr(self.env, "get_available_tasks"):
            return self.env.get_available_tasks()

        return []


class TemplateBasedLanguageProvider(LanguageProvider):
    """Language provider that uses templates with variable substitution.

    This provider fills in templates with task-specific variables to generate
    hierarchical descriptions. Useful for structured tasks with predictable patterns.

    Example templates:
        ```python
        templates = {
            "pick_and_place": {
                "task": "Pick up the {color} {object} and place it {location}.",
                "subtasks": [
                    "Move to the {color} {object}.",
                    "Grasp the {color} {object}.",
                    "Move {location}.",
                    "Release the {object}.",
                ],
            }
        }
        ```

    Args:
        cfg: Language configuration.
        templates: Dictionary of templates keyed by task ID.
        variables: Optional default variable values.
    """

    def __init__(
        self,
        cfg: LanguageCfg,
        templates: Dict[str, Dict[str, Any]],
        variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg)
        self.templates = templates
        self.variables = variables or {}

    def get_language(
        self, task_id: str, context: Optional[Dict[str, Any]] = None
    ) -> HierarchicalLanguageData:
        """Generate language data from templates for a specific task.

        Args:
            task_id: Unique identifier for the task.
            context: Optional context with variable values.

        Returns:
            HierarchicalLanguageData generated from templates.
        """
        if task_id not in self.templates:
            log_error(
                f"Task ID '{task_id}' not found in templates. "
                f"Available tasks: {list(self.templates.keys())}",
                error_type=KeyError,
            )

        template = self.templates[task_id]

        # Merge default variables with context
        vars_to_use = {**self.variables, **(context or {})}

        # Fill in task-level template
        task_template = template.get("task", "Complete the task.")
        task_text = task_template.format(**vars_to_use)

        # Fill in subtask templates
        subtask_templates = template.get("subtasks", [])
        subtask_texts = [
            st.format(**vars_to_use) for st in subtask_templates if isinstance(st, str)
        ]

        # Fill in primitive templates
        primitive_templates = template.get("primitives", [])
        primitive_texts = [
            pt.format(**vars_to_use)
            for pt in primitive_templates
            if isinstance(pt, str)
        ]

        # Get change points if specified
        change_points = template.get("change_points", None)

        # Would need LanguageManager to tokenize - return placeholder
        return HierarchicalLanguageData(
            task_level=[],  # Would be populated with LanguageData objects
            subtask_level=[],
            primitive_level=[],
            change_points=change_points,
        )

    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs from templates.

        Returns:
            List of task identifiers.
        """
        return list(self.templates.keys())
