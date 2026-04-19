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


class NameNormalizer:
    """Handles name normalization for different entity types."""

    VALID_KEYS = {"joint", "link"}
    VALID_MODES = {"upper", "lower", "none"}

    def __init__(self, default_case: dict[str, str] | None = None):
        """Initialize the NameNormalizer with default cases.

        Args:
            default_case (dict[str, str] | None): Default normalization modes for "joint" and "link".
        """
        self._name_case = {
            "joint": "upper",
            "link": "lower",
        }
        if default_case:
            for key, mode in default_case.items():
                if key in self.VALID_KEYS and mode in self.VALID_MODES:
                    self._name_case[key] = mode
                else:
                    raise ValueError(
                        f"Invalid default_case entry {key}={mode}. "
                        f"Allowed keys: {self.VALID_KEYS}, allowed modes: {self.VALID_MODES}."
                    )

    def set_case(self, key: str, mode: str):
        """Set the normalization mode for a specific key.

        Args:
            key (str): The entity type ("joint" or "link").
            mode (str): The normalization mode ("upper", "lower", "none").
        """
        if key in self.VALID_KEYS and mode in self.VALID_MODES:
            self._name_case[key] = mode
        else:
            raise ValueError(
                f"Invalid key or mode: {key}={mode}. "
                f"Allowed keys: {self.VALID_KEYS}, allowed modes: {self.VALID_MODES}."
            )

    def normalize(self, kind: str, name: str | None) -> str | None:
        """Normalize a name according to the configured case policy.

        Args:
            kind (str): One of "joint" or "link".
            name (str | None): The original name.

        Returns:
            str | None: The normalized name, or the original value if kind is unknown or mode is "none".
        """
        if name is None:
            return None

        mode = self._name_case.get(kind, "none")
        if mode == "lower":
            return name.lower()
        if mode == "upper":
            return name.upper()
        return name
