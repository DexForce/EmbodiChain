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

"""Build a small real Sphinx site for architecture browser integration checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/build/architecture-fixture"


def main() -> None:
    """Build the real extension with lightweight documentation targets."""
    with tempfile.TemporaryDirectory(prefix="architecture-sphinx-") as folder:
        source = Path(folder) / "source"
        source.mkdir()
        (source / "_static").mkdir()
        (source / "conf.py").write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'docs/scripts')!r})\n"
            "extensions = ['myst_parser', 'architecture_sphinx']\n"
            "project = 'EmbodiChain'\n"
            "html_theme = 'sphinx_book_theme'\n"
            "html_static_path = ['_static']\n"
        )
        entry = source / "overview/architecture/index.md"
        entry.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "docs/source/overview/architecture/index.md", entry)
        data = json.loads((ROOT / "docs/architecture/curated.json").read_text())
        docnames = sorted(
            {doc["docname"] for node in data["nodes"] for doc in node["documentation"]}
        )
        for docname in docnames:
            path = source / (docname + ".rst")
            path.parent.mkdir(parents=True, exist_ok=True)
            title = docname.rsplit("/", 1)[-1]
            path.write_text(
                title
                + "\n"
                + "=" * len(title)
                + "\n\nDocumentation target for architecture link verification.\n"
            )
        (source / "index.rst").write_text(
            "EmbodiChain\n===========\n\n.. toctree::\n\n   overview/architecture/index\n"
            + "".join("   " + name + "\n" for name in docnames)
        )
        built = Path(folder) / "html"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "-b",
                "html",
                str(source),
                str(built),
            ],
            check=True,
        )
        # Text builders must provide searchable content without the Node build.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "-b",
                "text",
                str(source),
                str(Path(folder) / "text"),
            ],
            check=True,
        )
        text = (Path(folder) / "text/overview/architecture/index.txt").read_text()
        assert "AtomicActionEngine" in text and "<iframe" not in text
        # Historical checkouts without this extension retain their plain Sphinx build.
        legacy = Path(folder) / "legacy"
        legacy.mkdir()
        (legacy / "conf.py").write_text("project = 'Historical documentation'\n")
        (legacy / "index.rst").write_text(
            "Historical documentation\n========================\n"
        )
        legacy_output = Path(folder) / "legacy-html"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "-b",
                "html",
                str(legacy),
                str(legacy_output),
            ],
            check=True,
        )
        assert (legacy_output / "index.html").is_file()
        assert not (legacy_output / "_static/architecture").exists()
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        shutil.copytree(built, OUTPUT)
        for prefix in ("main", "v0.2.4", "EmbodiChain/main"):
            shutil.copytree(built, OUTPUT / prefix)
    print(f"Architecture Sphinx fixture ready: {OUTPUT}")


if __name__ == "__main__":
    main()
