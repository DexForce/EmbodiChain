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
from typing import Any, Dict, List, Optional, Literal, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np

from embodichain.utils import configclass
from embodichain.utils.logger import log_info, log_warning, log_error

__all__ = [
    "LanguageCfg",
    "LanguageCurriculumCfg",
    "LanguageAugmentationCfg",
    "LanguageManager",
    "LanguageData",
    "HierarchicalLanguageData",
]


@configclass
class LanguageCfg:
    """Configuration for language data in rollout buffers.

    Supports three storage modes:
    - 'tokens': Store token IDs (default, most flexible)
    - 'embeddings': Store pre-computed embeddings
    - 'hybrid': Store both tokens and embeddings

    Supports hierarchical language structure for VLA training:
    - task_level: Overall goal/description
    - subtask_level: Intermediate step descriptions
    - primitive_level: Low-level action descriptions

    Args:
        mode: Storage mode ('tokens', 'embeddings', or 'hybrid').
        hierarchy_levels: List of hierarchy levels to store. If None, uses
            all levels. Valid levels: 'task', 'subtask', 'primitive'.
        max_tokens: Maximum sequence length for tokenized text.
        tokenizer: Tokenizer/model identifier (huggingface or OpenAI).
        pad_token_id: Token ID used for padding.
        max_instructions_per_level: Maximum number of instructions per hierarchy level.
        embedding_dim: Dimension of text embeddings (when mode='embeddings' or 'hybrid').
        embedding_type: How to compute embeddings from tokens.
        tokenizer_backend: 'huggingface' or 'openai'.
        trust_remote_code: Whether to trust remote code for huggingface tokenizers.
    """

    mode: Literal["tokens", "embeddings", "hybrid"] = "tokens"
    """Storage mode for language data."""

    hierarchy_levels: Optional[List[Literal["task", "subtask", "primitive"]]] = None
    """Hierarchy levels to store. If None, uses all levels."""

    max_tokens: int = 512
    """Maximum sequence length for tokenized text per instruction."""

    tokenizer: str = "gpt2"
    """Tokenizer/model identifier."""

    pad_token_id: int = 0
    """Token ID used for padding."""

    max_instructions_per_level: int = 3
    """Maximum number of instructions per hierarchy level."""

    embedding_dim: int = 768
    """Dimension of text embeddings."""

    embedding_type: Literal["mean_pool", "cls", "last"] = "mean_pool"
    """How to compute embeddings from tokens."""

    tokenizer_backend: Literal["huggingface", "openai"] = "huggingface"
    """Tokenizer backend to use."""

    trust_remote_code: bool = False
    """Whether to trust remote code for huggingface tokenizers."""

    def __post_init__(self) -> None:
        if self.hierarchy_levels is None:
            self.hierarchy_levels = ["task", "subtask", "primitive"]

        # Validate hierarchy levels
        valid_levels = {"task", "subtask", "primitive"}
        for level in self.hierarchy_levels:
            if level not in valid_levels:
                log_error(
                    f"Invalid hierarchy level: {level}. Must be one of {valid_levels}.",
                    error_type=ValueError,
                )


@configclass
class LanguageCurriculumCfg:
    """Language complexity curriculum for progressive training.

    Defines stages of increasing language complexity, allowing the model
    to learn from simple descriptions before tackling complex ones.

    Args:
        stages: List of curriculum stages, each defining complexity constraints.
        stage_duration: Number of training steps per curriculum stage.
        enabled: Whether curriculum learning is enabled.
    """

    @dataclass
    class CurriculumStage:
        """Configuration for a single curriculum stage."""

        max_words: int = 50
        """Maximum number of words per instruction."""

        max_sentences: int = 2
        """Maximum number of sentences per instruction."""

        max_hierarchy_depth: int = 1
        """Maximum hierarchy depth (1=task only, 2=task+subtask, 3=all)."""

        vocabulary_complexity: Literal["simple", "moderate", "complex"] = "simple"
        """Vocabulary complexity level."""

        instruction_types: List[str] = field(default_factory=lambda: ["imperative"])
        """Allowed instruction types: 'imperative', 'declarative', 'conditional'."""

    stages: List[CurriculumStage] = field(
        default_factory=lambda: [
            LanguageCurriculumCfg.CurriculumStage(
                max_words=10,
                max_sentences=1,
                max_hierarchy_depth=1,
                vocabulary_complexity="simple",
                instruction_types=["imperative"],
            ),
            LanguageCurriculumCfg.CurriculumStage(
                max_words=25,
                max_sentences=2,
                max_hierarchy_depth=2,
                vocabulary_complexity="moderate",
                instruction_types=["imperative", "declarative"],
            ),
            LanguageCurriculumCfg.CurriculumStage(
                max_words=50,
                max_sentences=3,
                max_hierarchy_depth=3,
                vocabulary_complexity="complex",
                instruction_types=["imperative", "declarative", "conditional"],
            ),
        ]
    )

    stage_duration: int = 1000
    """Number of training steps per curriculum stage."""

    enabled: bool = False
    """Whether curriculum learning is enabled."""


@configclass
class LanguageAugmentationCfg:
    """Configuration for language data augmentation.

    Augmentations are applied during sampling to increase data diversity
    and improve model generalization.

    Args:
        back_translation: Use back-translation for paraphrasing.
        synonym_replacement: Probability of replacing words with synonyms.
        template_variation: Apply template-based rephrasing.
        drop_word: Probability of randomly dropping a word.
        swap_word: Probability of swapping two adjacent words.
        insert_word: Probability of inserting a filler word.
    """

    back_translation: bool = False
    """Use back-translation for paraphrasing."""

    synonym_replacement: float = 0.0
    """Probability of replacing words with synonyms [0.0, 1.0]."""

    template_variation: bool = False
    """Apply template-based rephrasing."""

    drop_word: float = 0.0
    """Probability of randomly dropping a word [0.0, 1.0]."""

    swap_word: float = 0.0
    """Probability of swapping two adjacent words [0.0, 1.0]."""

    insert_word: float = 0.0
    """Probability of inserting a filler word [0.0, 1.0]."""

    augmentation_prob: float = 0.5
    """Overall probability of applying any augmentation [0.0, 1.0]."""


@dataclass
class LanguageData:
    """Single-level language data structure.

    Contains tokenized text and metadata for a single instruction.

    Args:
        tokens: Token IDs tensor of shape [seq_len].
        attention_mask: Attention mask tensor of shape [seq_len].
        raw_text: Original raw text string (for debugging).
        instruction_type: Type of instruction (imperative, declarative, etc.).
        metadata: Additional metadata dictionary.
    """

    tokens: torch.Tensor
    attention_mask: torch.Tensor
    raw_text: str
    instruction_type: str = "imperative"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tokens": self.tokens,
            "attention_mask": self.attention_mask,
            "raw_text": self.raw_text,
            "instruction_type": self.instruction_type,
            "metadata": self.metadata,
        }


@dataclass
class HierarchicalLanguageData:
    """Hierarchical language data structure for VLA training.

    Organizes language instructions at multiple abstraction levels:
    - task_level: High-level goal/description
    - subtask_level: Intermediate step descriptions
    - primitive_level: Low-level action descriptions

    This structure enables VLA models to learn from multi-scale language
    representations, similar to human task understanding.

    Args:
        task_level: List of task-level instructions.
        subtask_level: List of subtask-level instructions.
        primitive_level: List of primitive-level instructions.
        hierarchy_depth: Current depth of the hierarchy (1-3).
        change_points: Timesteps where language changes within the trajectory.
    """

    task_level: List[LanguageData] = field(default_factory=list)
    subtask_level: List[LanguageData] = field(default_factory=list)
    primitive_level: List[LanguageData] = field(default_factory=list)
    hierarchy_depth: int = 3
    change_points: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.change_points is None:
            self.change_points = [0]

    def get_level(self, level: str) -> List[LanguageData]:
        """Get language data for a specific hierarchy level.

        Args:
            level: Hierarchy level ('task', 'subtask', 'primitive').

        Returns:
            List of LanguageData for the requested level.
        """
        level_map = {
            "task": self.task_level,
            "subtask": self.subtask_level,
            "primitive": self.primitive_level,
        }
        if level not in level_map:
            log_error(f"Invalid hierarchy level: {level}", error_type=ValueError)
        return level_map[level]

    def set_level(self, level: str, data: List[LanguageData]) -> None:
        """Set language data for a specific hierarchy level.

        Args:
            level: Hierarchy level ('task', 'subtask', 'primitive').
            data: List of LanguageData to set.
        """
        level_map = {
            "task": "task_level",
            "subtask": "subtask_level",
            "primitive": "primitive_level",
        }
        if level not in level_map:
            log_error(f"Invalid hierarchy level: {level}", error_type=ValueError)
        setattr(self, level_map[level], data)

    def flatten(self) -> Dict[str, List[LanguageData]]:
        """Flatten hierarchical structure into a dictionary.

        Returns:
            Dictionary mapping level names to their language data.
        """
        return {
            "task": self.task_level,
            "subtask": self.subtask_level,
            "primitive": self.primitive_level,
        }

    def to_buffer_format(self, cfg: LanguageCfg) -> Dict[str, torch.Tensor]:
        """Convert hierarchical language data to buffer tensor format.

        Args:
            cfg: Language configuration for buffer layout.

        Returns:
            Dictionary with tensor fields ready for rollout buffer.
        """
        result = {}

        # Process each hierarchy level
        for level in cfg.hierarchy_levels:
            level_data = self.get_level(level)
            level_key = f"{level}_level"

            # Pad to max_instructions_per_level
            padded_tokens = []
            padded_masks = []

            for i in range(cfg.max_instructions_per_level):
                if i < len(level_data):
                    # Pad sequence to max_tokens
                    tokens = level_data[i].tokens
                    mask = level_data[i].attention_mask

                    seq_len = tokens.shape[0]
                    if seq_len < cfg.max_tokens:
                        pad_len = cfg.max_tokens - seq_len
                        tokens = torch.cat(
                            [
                                tokens,
                                torch.full(
                                    (pad_len,),
                                    cfg.pad_token_id,
                                    dtype=tokens.dtype,
                                    device=tokens.device,
                                ),
                            ]
                        )
                        mask = torch.cat(
                            [
                                mask,
                                torch.zeros(
                                    (pad_len,), dtype=mask.dtype, device=mask.device
                                ),
                            ]
                        )
                    elif seq_len > cfg.max_tokens:
                        tokens = tokens[: cfg.max_tokens]
                        mask = mask[: cfg.max_tokens]
                else:
                    # Empty instruction
                    tokens = torch.full(
                        (cfg.max_tokens,),
                        cfg.pad_token_id,
                        dtype=torch.int64,
                        device="cpu",
                    )
                    mask = torch.zeros(
                        (cfg.max_tokens,),
                        dtype=torch.int64,
                        device="cpu",
                    )

                padded_tokens.append(tokens)
                padded_masks.append(mask)

            # Stack instructions
            result[f"{level_key}_tokens"] = torch.stack(padded_tokens)
            result[f"{level_key}_attention_mask"] = torch.stack(padded_masks)

        # Add instruction counts
        result["instruction_counts"] = torch.tensor(
            [
                len(self.task_level),
                len(self.subtask_level),
                len(self.primitive_level),
            ],
            dtype=torch.int64,
        )

        # Add change points (padded to max_instructions_per_level)
        change_points = torch.full(
            (cfg.max_instructions_per_level,),
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        for i, cp in enumerate(self.change_points[: cfg.max_instructions_per_level]):
            change_points[i] = cp
        result["change_points"] = change_points

        return result


class LanguageManager:
    """Manages language data generation, tokenization, and storage.

    The LanguageManager handles:
    - Loading and configuring tokenizers
    - Generating or retrieving hierarchical language descriptions
    - Tokenizing text into model-ready format
    - Managing language curriculum and augmentation

    Args:
        cfg: Language configuration.
        env: Reference to the environment for context.
    """

    def __init__(self, cfg: LanguageCfg, env) -> None:
        self.cfg = cfg
        self.env = env
        self._tokenizer = None
        self._load_tokenizer()

        # Curriculum state
        self._curriculum_step = 0
        self._current_stage = 0

        # Cache for tokenized language
        self._language_cache: Dict[str, HierarchicalLanguageData] = {}

        log_info(
            f"[LanguageManager] Initialized with mode={cfg.mode}, "
            f"hierarchy={cfg.hierarchy_levels}, tokenizer={cfg.tokenizer}"
        )

    def _load_tokenizer(self) -> None:
        """Load the tokenizer based on configuration."""
        if self.cfg.tokenizer_backend == "huggingface":
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.tokenizer,
                    trust_remote_code=self.cfg.trust_remote_code,
                )

                # Update pad_token_id from tokenizer if not specified
                if (
                    self.cfg.pad_token_id == 0
                    and self._tokenizer.pad_token_id is not None
                ):
                    self.cfg.pad_token_id = self._tokenizer.pad_token_id

                log_info(
                    f"[LanguageManager] Loaded huggingface tokenizer: {self.cfg.tokenizer}"
                )
            except ImportError:
                log_error(
                    "transformers library not installed. "
                    "Install with: pip install transformers",
                    error_type=ImportError,
                )
            except Exception as e:
                log_error(
                    f"Failed to load huggingface tokenizer: {e}",
                    error_type=RuntimeError,
                )
        elif self.cfg.tokenizer_backend == "openai":
            try:
                import tiktoken

                self._tokenizer = tiktoken.encoding_for_model(self.cfg.tokenizer)
                log_info(
                    f"[LanguageManager] Loaded OpenAI tokenizer: {self.cfg.tokenizer}"
                )
            except ImportError:
                log_error(
                    "tiktoken library not installed. "
                    "Install with: pip install tiktoken",
                    error_type=ImportError,
                )
        else:
            log_error(
                f"Unknown tokenizer backend: {self.cfg.tokenizer_backend}",
                error_type=ValueError,
            )

    def tokenize(
        self, text: str, return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single text string.

        Args:
            text: Text to tokenize.
            return_tensors: Return tensor format ('pt' for PyTorch).

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        if self._tokenizer is None:
            log_error("Tokenizer not initialized", error_type=RuntimeError)

        if self.cfg.tokenizer_backend == "huggingface":
            result = self._tokenizer(
                text,
                max_length=self.cfg.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors=return_tensors,
            )
            # Ensure dtype is int64
            result["input_ids"] = result["input_ids"].to(torch.int64)
            result["attention_mask"] = result["attention_mask"].to(torch.int64)
            return result
        else:  # openai/tiktoken
            tokens = self._tokenizer.encode(
                text,
                max_length=self.cfg.max_tokens,
                truncation=True,
            )
            # Pad to max_tokens
            if len(tokens) < self.cfg.max_tokens:
                tokens = tokens + [self.cfg.pad_token_id] * (
                    self.cfg.max_tokens - len(tokens)
                )
            else:
                tokens = tokens[: self.cfg.max_tokens]

            input_ids = torch.tensor(tokens, dtype=torch.int64)
            attention_mask = (input_ids != self.cfg.pad_token_id).to(torch.int64)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def tokenize_batch(
        self, texts: List[str], return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of text strings.

        Args:
            texts: List of texts to tokenize.
            return_tensors: Return tensor format ('pt' for PyTorch).

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        if self._tokenizer is None:
            log_error("Tokenizer not initialized", error_type=RuntimeError)

        if self.cfg.tokenizer_backend == "huggingface":
            result = self._tokenizer(
                texts,
                max_length=self.cfg.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors=return_tensors,
            )
            result["input_ids"] = result["input_ids"].to(torch.int64)
            result["attention_mask"] = result["attention_mask"].to(torch.int64)
            return result
        else:  # openai/tiktoken
            batch_tokens = []
            for text in texts:
                tokens = self._tokenizer.encode(
                    text,
                    max_length=self.cfg.max_tokens,
                    truncation=True,
                )
                if len(tokens) < self.cfg.max_tokens:
                    tokens = tokens + [self.cfg.pad_token_id] * (
                        self.cfg.max_tokens - len(tokens)
                    )
                else:
                    tokens = tokens[: self.cfg.max_tokens]
                batch_tokens.append(tokens)

            input_ids = torch.tensor(batch_tokens, dtype=torch.int64)
            attention_mask = (input_ids != self.cfg.pad_token_id).to(torch.int64)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode.

        Returns:
            Decoded text string.
        """
        if self._tokenizer is None:
            log_error("Tokenizer not initialized", error_type=RuntimeError)

        # Remove padding
        mask = token_ids != self.cfg.pad_token_id
        token_ids = token_ids[mask]

        if self.cfg.tokenizer_backend == "huggingface":
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        else:  # openai/tiktoken
            return self._tokenizer.decode(token_ids)

    def create_language_data(
        self, text: str, instruction_type: str = "imperative", **metadata
    ) -> LanguageData:
        """Create a LanguageData object from raw text.

        Args:
            text: Raw text string.
            instruction_type: Type of instruction.
            **metadata: Additional metadata.

        Returns:
            LanguageData object with tokenized text.
        """
        tokenized = self.tokenize(text)
        return LanguageData(
            tokens=tokenized["input_ids"].squeeze(0),
            attention_mask=tokenized["attention_mask"].squeeze(0),
            raw_text=text,
            instruction_type=instruction_type,
            metadata=metadata,
        )

    def create_hierarchical_language_data(
        self,
        task_texts: List[str] | str,
        subtask_texts: Optional[List[str] | str] = None,
        primitive_texts: Optional[List[str] | str] = None,
        change_points: Optional[List[int]] = None,
    ) -> HierarchicalLanguageData:
        """Create hierarchical language data from text at multiple levels.

        Args:
            task_texts: Task-level descriptions (string or list).
            subtask_texts: Subtask-level descriptions (optional).
            primitive_texts: Primitive-level descriptions (optional).
            change_points: Timesteps where language changes (optional).

        Returns:
            HierarchicalLanguageData object.
        """
        # Normalize to lists
        if isinstance(task_texts, str):
            task_texts = [task_texts]
        if subtask_texts is not None and isinstance(subtask_texts, str):
            subtask_texts = [subtask_texts]
        if primitive_texts is not None and isinstance(primitive_texts, str):
            primitive_texts = [primitive_texts]

        # Create language data for each level
        task_level = [self.create_language_data(text) for text in task_texts]
        subtask_level = (
            [self.create_language_data(text) for text in subtask_texts]
            if subtask_texts is not None
            else []
        )
        primitive_level = (
            [self.create_language_data(text) for text in primitive_texts]
            if primitive_texts is not None
            else []
        )

        return HierarchicalLanguageData(
            task_level=task_level,
            subtask_level=subtask_level,
            primitive_level=primitive_level,
            change_points=change_points,
        )

    def get_task_language(
        self, task_id: Optional[str] = None
    ) -> HierarchicalLanguageData:
        """Generate or retrieve language description for the current task.

        This method should be overridden in subclasses or configured via
        language providers to implement custom language generation logic.

        Args:
            task_id: Optional task identifier for cache lookup.

        Returns:
            HierarchicalLanguageData for the current task.
        """
        cache_key = task_id or "default"

        if cache_key in self._language_cache:
            return self._language_cache[cache_key]

        # Default implementation: generate generic task description
        task_name = getattr(self.env, "task_name", "unknown_task")
        task_description = getattr(
            self.env,
            "task_description",
            f"Complete the {task_name} task.",
        )

        language_data = self.create_hierarchical_language_data(
            task_texts=task_description,
            subtask_texts=None,  # Can be generated by subclasses
            primitive_texts=None,  # Can be generated by subclasses
        )

        self._language_cache[cache_key] = language_data
        return language_data

    def set_curriculum_step(
        self, step: int, curriculum_cfg: Optional[LanguageCurriculumCfg] = None
    ) -> None:
        """Update curriculum learning step.

        Args:
            step: Current curriculum step.
            curriculum_cfg: Optional curriculum configuration.
        """
        self._curriculum_step = step

        if curriculum_cfg and curriculum_cfg.enabled:
            self._current_stage = min(
                step // curriculum_cfg.stage_duration,
                len(curriculum_cfg.stages) - 1,
            )
            log_info(
                f"[LanguageManager] Curriculum: stage {self._current_stage}/{len(curriculum_cfg.stages)-1} "
                f"(step {step})"
            )

    def get_current_stage_constraints(
        self, curriculum_cfg: Optional[LanguageCurriculumCfg] = None
    ) -> Optional[Dict[str, Any]]:
        """Get constraints for the current curriculum stage.

        Args:
            curriculum_cfg: Optional curriculum configuration.

        Returns:
            Dictionary of constraints or None if curriculum is disabled.
        """
        if not curriculum_cfg or not curriculum_cfg.enabled:
            return None

        stage = curriculum_cfg.stages[self._current_stage]
        return {
            "max_words": stage.max_words,
            "max_sentences": stage.max_sentences,
            "max_hierarchy_depth": stage.max_hierarchy_depth,
            "vocabulary_complexity": stage.vocabulary_complexity,
            "instruction_types": stage.instruction_types,
        }

    def clear_cache(self) -> None:
        """Clear the language cache."""
        self._language_cache.clear()
        log_info("[LanguageManager] Language cache cleared")
