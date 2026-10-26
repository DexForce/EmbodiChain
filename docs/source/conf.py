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

"""Configure the Sphinx documentation builder."""

from __future__ import annotations

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "AZURE_OPENAI_ENDPOINT", "https://mock-endpoint.openai.azure.com/"
)
os.environ.setdefault("AZURE_OPENAI_API_KEY", "mock-api-key-for-docs-build")

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "extensions" / "embodichain_models")
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


project = "EmbodiChain"
copyright = "2021-2026, The EmbodiChain Project Developers"
author = "The EmbodiChain Project Developers"

# Read version from VERSION file if it exists
with open(os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "architecture_sphinx",
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",  # optional, shows type hints
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_parser",  # if you prefer Markdown pages
    "sphinx_copybutton",
]
# Napoleon settings if using Google/NumPy docstring style:
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_typehints = "signature"
autodoc_class_signature = "separated"
# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = False
# default autodoc settings
autodoc_default_options = {
    "autosummary": True,
}

# If using MyST and writing .md API stubs:
myst_enable_extensions = ["colon_fence", "deflist", "html_admonition"]
myst_heading_anchors = 4
myst_fence_as_directive = ["mermaid"]


templates_path = ["_templates"]
exclude_patterns = []


# -- Version selector sidebar ---------------------------------------------------
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "versioning.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# Don't include version-redirect.js automatically - we add it manually to root
html_js_files = []
# html_logo = "_static/logo_e.png"

# Configure HTML base URL for better local previewing
# Use empty string to use relative paths from the build directory
html_baseurl = ""

# Keep published model URLs working after the vendor/model reorganization.
robot_doc_redirects = {
    "resources/robot/anymal_c": "resources/robot/anybotics/anymal_c",
    "resources/robot/microduck": "resources/robot/pollen_robotics/microduck",
    "resources/robot/unitree_g1": "resources/robot/unitree/g1",
    "resources/robot/unitree_go1": "resources/robot/unitree/go1",
    "resources/robot/unitree_go2": "resources/robot/unitree/go2",
    "resources/robot/unitree_h1_2": "resources/robot/unitree/h1_2",
    "resources/robot/cobotmagic": "resources/robot/agilex/cobotmagic",
    "resources/robot/arx5": "resources/robot/arx/arx5",
    "resources/robot/dexforce_w1": "resources/robot/dexforce/w1",
    "resources/robot/flexiv_rizon4s": "resources/robot/flexiv/rizon4s",
    "resources/robot/franka_panda": "resources/robot/franka/panda",
    "resources/robot/so101": "resources/robot/the_robot_studio/so101",
    "resources/robot/tianji_marvin": "resources/robot/tianji/marvin",
    "resources/robot/ur_robot": "resources/robot/universal_robots/ur_robot",
    "resources/robot/aloha_mini": "resources/robot/alohamini/aloha_mini",
}
html_additional_pages = {
    old_docname: "robot_redirect.html" for old_docname in robot_doc_redirects
}

# HTML context for better path handling
html_context = {
    "github_user": "dexforce",
    "github_repo": "EmbodiChain",
    "github_version": "main",
    "doc_path": "docs/source",
    "robot_doc_redirects": robot_doc_redirects,
}

html_theme_options = {
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "max_navbar_depth": 4,
    "collapse_navbar": False,
}
