# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import os
import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)

__all__ = ["URDFAssemblySignatureManager"]


class URDFAssemblySignatureManager:
    r"""Simple MD5-based signature manager for URDF assemblies without persistent cache."""

    def __init__(self):
        self.logger = URDFAssemblyLogger.get_logger("signature_manager")

    def _calculate_file_md5(self, file_path: str) -> str:
        r"""Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating MD5 for {file_path}: {e}")
            return ""

    def _calculate_string_md5(self, content: str) -> str:
        r"""Calculate MD5 hash of a string."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def calculate_assembly_signature(self, urdf_dict: dict, output_path: str) -> str:
        r"""Calculate a unique signature for the assembly configuration.

        Args:
            urdf_dict (dict): Dictionary of components and their configurations
            output_path (str): Target output path for the assembly

        Returns:
            str: MD5 hash representing the assembly configuration
        """
        signature_data = {
            "output_filename": os.path.basename(output_path),
            "components": {},
        }

        def to_serializable(obj):
            r"""Recursively convert objects to types that are JSON serializable.

            Args:
                obj: The object to convert (could be Path, dict, list, or other types).

            Returns:
                The converted object, ready for JSON serialization.
                - Path objects are converted to strings.
                - dict and list are recursively processed.
                - Other types are returned as-is.
            """
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            else:
                return obj

        # Process each component
        for comp_type, comp_obj in urdf_dict.items():
            if comp_obj is None:
                continue

            # Calculate file MD5
            file_md5 = self._calculate_file_md5(comp_obj.urdf_path)
            if not file_md5:
                self.logger.warning(f"Could not calculate MD5 for {comp_obj.urdf_path}")
                continue

            # Include component configuration
            comp_data = {
                "urdf_path": str(comp_obj.urdf_path),
                "file_md5": file_md5,
                "params": to_serializable(comp_obj.params or {}),
                "transform": (
                    comp_obj.transform.tolist()
                    if comp_obj.transform is not None
                    else None
                ),
            }

            signature_data["components"][comp_type] = comp_data

        # Convert to JSON string for consistent hashing
        signature_json = json.dumps(signature_data, sort_keys=True, ensure_ascii=False)
        assembly_md5 = self._calculate_string_md5(signature_json)

        self.logger.info(f"Assembly signature calculated: [{assembly_md5}]")
        self.logger.debug(f"Signature data: {signature_json}")

        return assembly_md5

    def extract_signature_from_urdf(self, urdf_file_path: str) -> str:
        r"""Extract signature from existing URDF file's header comment.

        Args:
            urdf_file_path (str): Path to existing URDF file

        Returns:
            str: Extracted signature or empty string if not found
        """
        try:
            with open(urdf_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for signature in comment
            import re

            # 1. <!-- ASSEMBLY_SIGNATURE: [hash] -->
            # 2. <!--配置签名 ASSEMBLY_SIGNATURE: [hash]-->
            patterns = [
                r"<!--\s*ASSEMBLY_SIGNATURE:\s*([a-f0-9]{32})\s*-->",
                r"<!--.*ASSEMBLY_SIGNATURE:\s*([a-f0-9]{32}).*-->",
            ]

            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    signature = match.group(1)
                    self.logger.info(
                        f"Found existing signature in ({urdf_file_path}): [{signature}]"
                    )
                    return signature

            self.logger.debug(f"No signature found in {urdf_file_path}")
            return ""

        except Exception as e:
            self.logger.warning(
                f"Failed to extract signature from {urdf_file_path}: {e}", exc_info=True
            )
            return ""

    def is_assembly_up_to_date(self, current_signature: str, output_path: str) -> bool:
        r"""Check if the assembly at output_path has the same signature as current configuration.

        Args:
            current_signature (str): MD5 signature of current assembly configuration
            output_path (str): Path to existing URDF file

        Returns:
            bool: True if signatures match and file exists
        """
        if not os.path.exists(output_path):
            self.logger.info(f"Output file does not exist: {output_path}")
            return False

        # Verify file is not empty and is valid URDF
        try:
            if os.path.getsize(output_path) == 0:
                self.logger.warning(f"Output file is empty: {output_path}")
                return False

            # Try to parse as XML to ensure it's valid
            ET.parse(output_path)
        except Exception as e:
            self.logger.warning(
                f"Output file is invalid: {output_path}, error: {e}", exc_info=True
            )
            return False

        # Extract signature from existing file
        existing_signature = self.extract_signature_from_urdf(output_path)

        if existing_signature == current_signature:
            self.logger.info(
                f"✅ Assembly is up-to-date. Signature: {current_signature}"
            )
            return True
        else:
            if existing_signature:
                self.logger.info(f"Assembly signatures differ:")
                self.logger.info(f"  Current:  {current_signature}")
                self.logger.info(f"  Existing: {existing_signature}")
            else:
                self.logger.info(f"No signature found in existing file: {output_path}")
            return False
