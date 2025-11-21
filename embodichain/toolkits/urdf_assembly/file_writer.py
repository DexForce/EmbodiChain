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

from datetime import datetime

import xml.etree.ElementTree as ET

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)

__all__ = ["URDFFileWriter"]


class URDFFileWriter:
    r"""Responsible for formatting XML and writing URDF files with proper headers."""

    def __init__(self, module_names: list = None):
        r"""Initialize the URDFFileWriter.

        Args:
            module_names (list): List of module names to include in the header.
        """
        self.module_names = module_names or []
        self.logger = URDFAssemblyLogger.get_logger("file_writer")

    def create_section_comment(
        self, content: str, comment_type: str = "section"
    ) -> ET.Comment:
        r"""Create standardized section comments for URDF organization.

        Args:
            content (str): The content of the comment.
            comment_type (str): Type of comment - "section", "start", "end", "empty".

        Returns:
            ET.Comment: XML comment element.
        """
        if comment_type == "empty":
            return ET.Comment("")
        elif comment_type == "start":
            return ET.Comment(f" Start of ({content.lower()}) ")
        elif comment_type == "end":
            return ET.Comment(f" End of ({content.lower()}) ")
        else:
            return ET.Comment(f" {content} ")

    def add_section_comments(
        self, elements_list: list, part_name: str, section_type: str
    ):
        r"""Add standardized section comments to elements list.

        Args:
            elements_list (list): List to add comments to (links or joints).
            part_name (str): Name of the component part.
            section_type (str): Type of section ("Links" or "Joints").
        """
        elements_list.append(self.create_section_comment("", "empty"))
        elements_list.append(
            self.create_section_comment(
                f"{section_type} for part: {part_name}", "start"
            )
        )

    def add_section_end_comments(
        self, elements_list: list, part_name: str, section_type: str
    ):
        r"""Add standardized section end comments to elements list.

        Args:
            elements_list (list): List to add comments to (links or joints).
            part_name (str): Name of the component part.
            section_type (str): Type of section ("Links" or "Joints").
        """
        elements_list.append(
            self.create_section_comment(f"{section_type} for part: {part_name}", "end")
        )
        elements_list.append(self.create_section_comment("", "empty"))

    def make_comment_line(self, content: str, width: int = 80) -> str:
        r"""Create a properly formatted comment line with centered content.

        Args:
            content (str): The content to be centered in the comment.
            width (int): Total width of the comment line (default is 80).

        Returns:
            str: A formatted XML comment line.
        """
        content = content.strip()
        pad_total = width - 7 - len(content)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total < 0:
            pad_left = 0
            pad_right = 0
        return f"<!--{' ' * pad_left}{content}{' ' * pad_right}-->"

    def generate_header(
        self, module_names: list = None, assembly_signature: str = None
    ) -> str:
        r"""Generate a standard header for URDF files with assembly signature.

        Args:
            module_names (list): List of module names to include in the header.
            assembly_signature (str): MD5 signature of the assembly configuration.

        Returns:
            str: Formatted header string.
        """
        if module_names is None:
            module_names = self.module_names

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate proper spacing for centered content
        header_width = 80
        separator_line = "<!--" + "=" * (header_width - 8) + "-->"

        def center_comment(text: str) -> str:
            """Center text within comment brackets with proper padding."""
            content_width = header_width - 8  # Account for <!-- and -->
            text_len = len(text)
            if text_len >= content_width:
                return f"<!--{text[:content_width]}-->"

            padding = content_width - text_len
            left_pad = padding // 2
            right_pad = padding - left_pad
            return f"<!--{' ' * left_pad}{text}{' ' * right_pad}-->"

        header_lines = [
            '<?xml version="1.0"?>',
            separator_line,
            center_comment("Robot URDF Model Generation Report"),
            center_comment(f"Generation Time: {now}"),
            center_comment("Tool Version: DexForce URDF Composer V1.0"),
            center_comment(f"Included Modules: {' + '.join(module_names)}"),
        ]

        # Add assembly signature if provided
        if assembly_signature:
            header_lines.append(
                center_comment(f"配置签名 ASSEMBLY_SIGNATURE: {assembly_signature}")
            )

        header_lines.append(separator_line)

        return "\n".join(header_lines) + "\n"

    def prettify(self, elem: ET.Element, level: int = 0) -> ET.Element:
        r"""Format an XML element by adding newlines and indentation.

        Args:
            elem (ET.Element): The XML element to format.
            level (int): The current indentation level (default is 0).

        Returns:
            ET.Element: The formatted XML element.
        """
        indent = "\n" + "  " * level  # Create indentation string based on level
        if len(elem):  # If the element has children
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "  # Add indentation if no text
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent  # Add indentation after the element
            for child in elem:
                self.prettify(child, level + 1)  # Recursive call for children
            if not child.tail or not child.tail.strip():
                child.tail = indent  # Ensure the last child has proper tail indentation
        else:  # If the element has no children
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent  # Add indentation for elements at a non-zero level
        return elem

    def write_urdf(
        self,
        merged_urdf: ET.Element,
        output_path: str,
        module_names: list = None,
        assembly_signature: str = None,
    ):
        r"""Write the merged URDF to file with proper formatting and header including signature.

        Args:
            merged_urdf (ET.Element): The merged URDF XML element.
            output_path (str): Path where the URDF file will be written.
            module_names (list): Optional list of module names for the header.
            assembly_signature (str): Optional assembly signature to include in header.
        """
        header = self.generate_header(module_names, assembly_signature)
        xml_str = ET.tostring(self.prettify(merged_urdf), encoding="unicode")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(xml_str)

        self.logger.info(f"URDF file written to: {output_path}")
