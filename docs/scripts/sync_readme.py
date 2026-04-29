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
"""Synchronize README.md to docs/source/introduction.rst.

Uses pypandoc for Markdown-to-RST conversion, then post-processes the output
to fix Sphinx-specific formatting issues.

Usage:
    python docs/scripts/sync_readme.py           # Overwrite introduction.rst
    python docs/scripts/sync_readme.py --check    # Exit 1 if stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

__all__ = ["convert_readme_to_rst", "postprocess_rst"]

# Resolve paths relative to this script
REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
RST_PATH = REPO_ROOT / "docs" / "source" / "introduction.rst"

# Prefix to make repo-root-relative paths work from docs/source/
_DOCS_PATH_PREFIX = "../../"


def _fix_image_path(path: str) -> str:
    """Prefix a repo-root-relative image path for use from docs/source/.

    Args:
        path: Image path from pandoc output (repo-root-relative).

    Returns:
        Path adjusted for the RST file location in docs/source/.
    """
    if path.startswith(("http://", "https://")):
        return path
    return _DOCS_PATH_PREFIX + path


def convert_readme_to_rst(readme_content: str) -> str:
    """Convert Markdown content to RST via pypandoc.

    Args:
        readme_content: Raw Markdown text from README.md.

    Returns:
        Raw RST string from pandoc (before post-processing).
    """
    import pypandoc

    return pypandoc.convert_text(readme_content, "rst", format="md")


def postprocess_rst(rst: str, readme_content: str) -> str:
    """Fix pandoc RST output for Sphinx compatibility.

    Applies these transformations:
    1. Strip badge substitution references and definitions.
    2. Convert ``[!NOTE]`` blockquote to ``.. NOTE::`` directive.
    3. Convert ``.. raw:: html`` centered-image blocks to ``.. image::``.
    4. Replace ``.. code:: bibtex`` with ``.. code-block:: bibtex``.
    5. Convert ``.. figure::`` (with caption) to ``.. image::``.

    Args:
        rst: Raw RST from pandoc.
        readme_content: Original Markdown (used to extract image paths).

    Returns:
        Cleaned RST suitable for Sphinx.
    """
    # Extract image paths from README <img> tags for centered HTML blocks
    readme_images = re.findall(r'<img\s+[^>]*src="([^"]+)"[^>]*>', readme_content)

    lines = rst.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # --- 1. Strip badge substitution reference lines ---
        if re.match(r"^\|.*\|", line):
            i += 1
            continue

        # --- 1b. Strip badge substitution definitions at the bottom ---
        if re.match(r"^\.\. \|\w[\w ]*\w\| image::", line):
            i += 1
            while i < len(lines) and lines[i].startswith("   "):
                i += 1
            continue

        # --- 2. Convert [!NOTE] blockquote to .. NOTE:: ---
        if re.match(r"^\s+\[!NOTE\]", line):
            note_match = re.match(r"^\s+\[!NOTE\]\s*(.*)", line)
            note_text = note_match.group(1) if note_match else ""
            note_text = note_text.replace("\\*", "*")
            note_lines: list[str] = []
            if note_text:
                note_lines.append(note_text)
            i += 1
            while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                cleaned = lines[i].strip().replace("\\*", "*")
                note_lines.append(cleaned)
                i += 1
            result_lines.append(".. NOTE::")
            for nl in note_lines:
                result_lines.append(f"   {nl}")
            continue

        # --- 3. Convert .. raw:: html centered blocks to .. image:: ---
        if line.strip() == ".. raw:: html":
            # Look ahead (skipping blank lines) for <p align="center">
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and "<p align" in lines[j]:
                # Skip from i through the matching </p> raw block
                i = j + 1  # skip past <p> line
                while i < len(lines):
                    if "</p>" in lines[i]:
                        i += 1
                        # Skip any trailing .. raw:: html for </p>
                        while i < len(lines) and (
                            lines[i].strip() == ""
                            or lines[i].strip() == ".. raw:: html"
                            or "</p>" in lines[i]
                        ):
                            i += 1
                        break
                    i += 1
                # Insert images from README source
                for img_src in readme_images:
                    result_lines.append(f".. image:: {_fix_image_path(img_src)}")
                    result_lines.append("   :align: center")
                result_lines.append("")  # blank line after directive
                continue
            elif j < len(lines) and "</p>" in lines[j]:
                i = j + 1
                continue

        # --- 4. Replace .. code:: bibtex with .. code-block:: bibtex ---
        if re.match(r"^\.\. code:: bibtex\s*$", line):
            result_lines.append(".. code-block:: bibtex")
            i += 1
            continue

        # --- 5. Convert .. figure:: with caption to .. image:: ---
        if re.match(r"^\.\. figure::", line):
            path_match = re.match(r"^\.\. figure:: (.+)", line)
            if path_match:
                img_path = path_match.group(1).strip()
                result_lines.append(f".. image:: {_fix_image_path(img_path)}")
                i += 1
                # Skip :alt:, blank line, and caption lines
                while i < len(lines):
                    if lines[i].startswith("   :"):
                        i += 1
                        continue
                    if lines[i].strip() == "":
                        i += 1
                        continue
                    if lines[i].startswith("   "):
                        i += 1
                        continue
                    break
                continue

        result_lines.append(line)
        i += 1

    # Clean up excessive blank lines
    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    """CLI entry point for syncing README.md to introduction.rst."""
    parser = argparse.ArgumentParser(
        description="Sync README.md to docs/source/introduction.rst"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if introduction.rst is up-to-date (exit 1 if stale)",
    )
    args = parser.parse_args()

    if not README_PATH.exists():
        print(f"Error: {README_PATH} not found", file=sys.stderr)
        sys.exit(1)

    readme_content = README_PATH.read_text(encoding="utf-8")
    raw_rst = convert_readme_to_rst(readme_content)
    final_rst = postprocess_rst(raw_rst, readme_content)

    if args.check:
        if not RST_PATH.exists():
            print(
                f"Error: {RST_PATH} does not exist. Run without --check to generate.",
                file=sys.stderr,
            )
            sys.exit(1)
        current = RST_PATH.read_text(encoding="utf-8")
        if current != final_rst:
            print(
                f"Error: {RST_PATH} is out of sync with README.md. "
                "Run 'python docs/scripts/sync_readme.py' to update.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK: {RST_PATH} is up-to-date.")
    else:
        RST_PATH.parent.mkdir(parents=True, exist_ok=True)
        RST_PATH.write_text(final_rst, encoding="utf-8")
        print(f"Synced: {README_PATH} -> {RST_PATH}")


if __name__ == "__main__":
    main()
