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

import logging
import os
import shutil
import sys
from pathlib import Path

from setuptools import Command, find_namespace_packages, setup

__all__ = ["get_package_dir", "get_packages", "get_version"]

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

THIS_DIR = Path(__file__).resolve().parent
CORE_PACKAGE_PATTERNS = ["embodichain", "embodichain.*"]
TASKS_PROJECT_DIR = THIS_DIR / "embodichain_tasks"
TASKS_PACKAGE_PATTERNS = ["embodichain_tasks", "embodichain_tasks.*"]

# Defer importing torch until it's actually needed (when building extensions).
# This prevents `setup.py` from failing at import time in environments where
# torch isn't available or isn't on the same interpreter.
BuildExtension = None
CppExtension = None
CUDAExtension = None


class CleanCommand(Command):
    description = "Delete build, dist, *.egg-info and all __pycache__ directories."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for d in [
            "build",
            "dist",
            "embodichain.egg-info",
            "embodichain_tasks/embodichain_tasks.egg-info",
        ]:
            rm_path = THIS_DIR / d
            if not rm_path.exists():
                continue
            try:
                shutil.rmtree(rm_path, ignore_errors=True)
                logger.info(f"removed '{rm_path}'")
            except:
                pass

        for pdir, sdirs, filenames in os.walk(THIS_DIR):
            for sdir in sdirs:
                if sdir == "__pycache__":
                    rm_path = Path(pdir) / sdir
                    shutil.rmtree(str(rm_path), ignore_errors=True)
                    logger.info(f"removed '{rm_path}'")
            for filename in filenames:
                if filename.endswith(".so"):
                    rm_path = Path(pdir) / filename
                    rm_path.unlink()
                    logger.info(f"removed '{rm_path}'")


def get_packages() -> list[str]:
    """Return the core and official-task packages shipped in the main wheel."""
    core_packages = find_namespace_packages(
        where=str(THIS_DIR), include=CORE_PACKAGE_PATTERNS
    )
    task_packages = find_namespace_packages(
        where=str(TASKS_PROJECT_DIR), include=TASKS_PACKAGE_PATTERNS
    )
    return sorted(set([*core_packages, *task_packages, "embodichain_tasks.configs"]))


def get_package_dir() -> dict[str, str]:
    """Map the task package and configs without moving their source paths."""
    return {
        "embodichain_tasks": "embodichain_tasks/embodichain_tasks",
        "embodichain_tasks.configs": "embodichain_tasks/configs",
    }


def get_version() -> str:
    """Read the normalized package version from the repository version file."""
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        full_version = f.read().strip()
        version = ".".join(full_version.split(".")[:3])
    return version


def main():
    # Extract version
    version = get_version()

    # Make the version available as an explicit runtime package resource.
    src_version = THIS_DIR / "VERSION"
    dst_version = THIS_DIR / "embodichain" / "VERSION"
    if src_version.exists():
        shutil.copyfile(src_version, dst_version)
        logger.info(f"Copied VERSION to {dst_version}")

    cmdclass = {"clean": CleanCommand}
    if BuildExtension is not None:
        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

    setup(
        name="embodichain",
        version=version,
        url="https://github.com/DexForce/EmbodiChain",
        author="EmbodiChain Developers",
        description="An end-to-end, GPU-accelerated, and modular platform for building generalized Embodied Intelligence.",
        packages=get_packages(),
        package_dir=get_package_dir(),
        package_data={
            "embodichain": ["VERSION"],
            "embodichain.gen_sim.simready_pipeline.configs": ["*.json"],
            "embodichain_tasks.configs": ["**/*.json", "**/*.yaml", "**/*.yml"],
        },
        cmdclass=cmdclass,
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
