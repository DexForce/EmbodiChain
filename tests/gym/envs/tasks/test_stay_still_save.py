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

"""Tests for the StayStillSave benchmark task environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)

# Trigger task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.special.stay_still_save import (  # noqa: E402
    StayStillSaveEnv,
)


class TestStayStillSaveEnv:
    """Registration and structure tests for StayStillSaveEnv."""

    def test_module_all(self):
        """``__all__`` exports the env class."""
        from embodichain_tasks.special import stay_still_save

        assert "StayStillSaveEnv" in stay_still_save.__all__

    def test_registered_with_gym_id(self):
        """The env is registered under the StayStillSave-v1 gym id."""
        assert "StayStillSave-v1" in REGISTERED_ENVS
        spec = REGISTERED_ENVS["StayStillSave-v1"]
        assert spec.cls.__name__ == "StayStillSaveEnv"
        assert spec.max_episode_steps == 100

    def test_uses_future_annotations(self):
        """Module source starts with ``from __future__ import annotations``."""
        from embodichain_tasks.special import stay_still_save

        src = Path(stay_still_save.__file__).read_text()
        assert "from __future__ import annotations" in src

    def test_is_embodied_env_subclass(self):
        """StayStillSaveEnv subclasses EmbodiedEnv."""
        from embodichain.lab.gym.envs import EmbodiedEnv

        assert issubclass(StayStillSaveEnv, EmbodiedEnv)

    @pytest.mark.parametrize("gym_id", ["StayStillSave-v1"])
    def test_gym_spec_exists(self, gym_id):
        """The gym spec is registered with gymnasium."""
        import gymnasium

        # gymnasium.registry raises if not found.
        gymnasium.spec(gym_id)
