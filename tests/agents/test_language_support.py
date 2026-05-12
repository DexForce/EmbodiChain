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

"""Tests for language support in ODS and VLA training."""

import pytest
import torch
import tempfile
from pathlib import Path

from embodichain.lab.gym.envs.managers import (
    LanguageCfg,
    LanguageManager,
    LanguageData,
    HierarchicalLanguageData,
    FileBasedLanguageProvider,
    TemplateBasedLanguageProvider,
)
from embodichain.lab.gym.utils.gym_utils import _init_language_buffer


class MockEnv:
    """Mock environment for testing."""

    task_name = "test_task"
    task_description = "Complete the test task."


class TestLanguageData:
    """Tests for LanguageData and HierarchicalLanguageData."""

    def test_language_data_creation(self):
        """Test creating LanguageData objects."""
        tokens = torch.tensor([1, 2, 3, 0, 0], dtype=torch.int64)
        mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.int64)

        data = LanguageData(
            tokens=tokens,
            attention_mask=mask,
            raw_text="Test instruction",
            instruction_type="imperative",
        )

        assert data.tokens.shape == (5,)
        assert data.attention_mask.shape == (5,)
        assert data.raw_text == "Test instruction"
        assert data.instruction_type == "imperative"

    def test_hierarchical_language_data_creation(self):
        """Test creating HierarchicalLanguageData."""
        task_tokens = torch.tensor([1, 2, 3, 0], dtype=torch.int64)
        task_mask = torch.tensor([1, 1, 1, 0], dtype=torch.int64)

        task_data = LanguageData(
            tokens=task_tokens,
            attention_mask=task_mask,
            raw_text="Task description",
        )

        subtask_tokens = torch.tensor([4, 5, 0, 0], dtype=torch.int64)
        subtask_mask = torch.tensor([1, 1, 0, 0], dtype=torch.int64)

        subtask_data = LanguageData(
            tokens=subtask_tokens,
            attention_mask=subtask_mask,
            raw_text="Subtask description",
        )

        hierarchical = HierarchicalLanguageData(
            task_level=[task_data],
            subtask_level=[subtask_data],
            primitive_level=[],
        )

        assert len(hierarchical.task_level) == 1
        assert len(hierarchical.subtask_level) == 1
        assert len(hierarchical.primitive_level) == 0

    def test_hierarchical_language_data_flatten(self):
        """Test flattening hierarchical language data."""
        task_data = LanguageData(
            tokens=torch.tensor([1, 2, 0], dtype=torch.int64),
            attention_mask=torch.tensor([1, 1, 0], dtype=torch.int64),
            raw_text="Task",
        )

        hierarchical = HierarchicalLanguageData(
            task_level=[task_data],
            subtask_level=[],
            primitive_level=[],
        )

        flattened = hierarchical.flatten()
        assert "task" in flattened
        assert "subtask" in flattened
        assert "primitive" in flattened


class TestLanguageBuffer:
    """Tests for language buffer initialization."""

    def test_init_language_buffer(self):
        """Test initializing language buffer tensors."""
        language_cfg = {
            "hierarchy_levels": ["task", "subtask"],
            "max_tokens": 256,
            "max_instructions_per_level": 3,
            "pad_token_id": 0,
            "mode": "tokens",
        }

        buffer = _init_language_buffer(
            language_cfg, batch_size=4, max_episode_steps=100, device="cpu"
        )

        # Check that expected keys are present
        assert "task_level_tokens" in buffer
        assert "task_level_attention_mask" in buffer
        assert "subtask_level_tokens" in buffer
        assert "subtask_level_attention_mask" in buffer

        # Check tensor shapes
        assert buffer["task_level_tokens"].shape == (4, 100, 3, 256)
        assert buffer["task_level_attention_mask"].shape == (4, 100, 3, 256)
        assert buffer["task_level_count"].shape == (4, 100)

        # Check global fields
        assert "instruction_counts" in buffer
        assert buffer["instruction_counts"].shape == (4, 100, 3)
        assert "change_points" in buffer
        assert buffer["change_points"].shape == (4, 100, 3)
        assert "hierarchy_depth" in buffer
        assert buffer["hierarchy_depth"].shape == (4, 100)


class TestLanguageManager:
    """Tests for LanguageManager."""

    def test_language_manager_initialization(self):
        """Test initializing LanguageManager."""
        cfg = LanguageCfg(
            mode="tokens",
            hierarchy_levels=["task", "subtask"],
            max_tokens=256,
            tokenizer="gpt2",
        )

        env = MockEnv()

        # Test with a simple tokenizer that doesn't require external dependencies
        try:
            manager = LanguageManager(cfg, env)
            assert manager.cfg == cfg
            assert manager.env == env
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Tokenizer not available: {e}")

    def test_create_language_data(self):
        """Test creating LanguageData from raw text."""
        cfg = LanguageCfg(
            mode="tokens",
            max_tokens=256,
            tokenizer="gpt2",
        )

        env = MockEnv()

        try:
            manager = LanguageManager(cfg, env)
            data = manager.create_language_data("Test instruction")
            assert isinstance(data, LanguageData)
            assert data.raw_text == "Test instruction"
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Tokenizer not available: {e}")

    def test_create_hierarchical_language_data(self):
        """Test creating hierarchical language data."""
        cfg = LanguageCfg(
            mode="tokens",
            max_tokens=256,
            tokenizer="gpt2",
        )

        env = MockEnv()

        try:
            manager = LanguageManager(cfg, env)
            data = manager.create_hierarchical_language_data(
                task_texts="Pick up the block.",
                subtask_texts=["Move to block.", "Grasp block."],
                primitive_texts=["Close gripper."],
            )

            assert isinstance(data, HierarchicalLanguageData)
            assert len(data.task_level) == 1
            assert len(data.subtask_level) == 2
            assert len(data.primitive_level) == 1
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Tokenizer not available: {e}")

    def test_to_buffer_format(self):
        """Test converting hierarchical data to buffer format."""
        cfg = LanguageCfg(
            mode="tokens",
            hierarchy_levels=["task", "subtask"],
            max_tokens=256,
            max_instructions_per_level=3,
            tokenizer="gpt2",
        )

        env = MockEnv()

        try:
            manager = LanguageManager(cfg, env)
            data = manager.create_hierarchical_language_data(
                task_texts="Task description.",
                subtask_texts=["Step 1.", "Step 2."],
            )

            buffer_format = data.to_buffer_format(cfg)

            assert "task_level_tokens" in buffer_format
            assert "subtask_level_tokens" in buffer_format
            assert "instruction_counts" in buffer_format

            # Check shapes
            assert buffer_format["task_level_tokens"].shape == (3, 256)
            assert buffer_format["subtask_level_tokens"].shape == (3, 256)
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Tokenizer not available: {e}")


class TestFileBasedLanguageProvider:
    """Tests for FileBasedLanguageProvider."""

    def test_file_provider_initialization(self):
        """Test initializing file-based provider."""
        cfg = LanguageCfg(
            mode="tokens",
            max_tokens=256,
            tokenizer="gpt2",
        )

        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
test_task:
  task:
    - "Test task description."
  subtask:
    - "Step 1."
    - "Step 2."
""")
            temp_path = f.name

        try:
            provider = FileBasedLanguageProvider(cfg, temp_path)
            assert provider.config_path == Path(temp_path)
            assert "test_task" in provider.get_available_tasks()
        finally:
            Path(temp_path).unlink()


class TestTemplateBasedLanguageProvider:
    """Tests for TemplateBasedLanguageProvider."""

    def test_template_provider_initialization(self):
        """Test initializing template-based provider."""
        cfg = LanguageCfg(
            mode="tokens",
            max_tokens=256,
            tokenizer="gpt2",
        )

        templates = {
            "test_task": {
                "task": "Complete the {object} task.",
                "subtasks": ["Move to {object}.", "Grasp {object}."],
            }
        }

        provider = TemplateBasedLanguageProvider(cfg, templates)
        assert "test_task" in provider.get_available_tasks()

    def test_template_provider_get_language(self):
        """Test getting language from templates."""
        cfg = LanguageCfg(
            mode="tokens",
            max_tokens=256,
            tokenizer="gpt2",
        )

        templates = {
            "test_task": {
                "task": "Pick up the {color} {object}.",
                "subtasks": [
                    "Move to {color} {object}.",
                    "Grasp {color} {object}.",
                ],
            }
        }

        provider = TemplateBasedLanguageProvider(cfg, templates)

        context = {"color": "red", "object": "block"}
        language_data = provider.get_language("test_task", context)

        assert isinstance(language_data, HierarchicalLanguageData)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
