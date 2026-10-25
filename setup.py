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

__all__ = ["get_example_data_files", "get_package_dir", "get_packages", "get_version"]

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

THIS_DIR = Path(__file__).resolve().parent
CORE_PACKAGE_PATTERNS = ["embodichain", "embodichain.*"]
EXTENSIONS_DIR = THIS_DIR / "extensions"
TASKS_PROJECT_DIR = EXTENSIONS_DIR / "embodichain_tasks"
TASKS_PACKAGE_PATTERNS = ["embodichain_tasks", "embodichain_tasks.*"]
ROBOTS_PACKAGE_PATTERNS = ["embodichain_robots", "embodichain_robots.*"]
BENCHMARKS_PACKAGE_PATTERNS = ["embodichain_benchmarks", "embodichain_benchmarks.*"]
EXAMPLES_DIR = THIS_DIR / "examples"

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
            "extensions/embodichain_tasks/embodichain_tasks.egg-info",
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
    """Return core and bundled extension packages shipped in the main wheel."""
    core_packages = find_namespace_packages(
        where=str(THIS_DIR), include=CORE_PACKAGE_PATTERNS
    )
    task_packages = find_namespace_packages(
        where=str(TASKS_PROJECT_DIR), include=TASKS_PACKAGE_PATTERNS
    )
    robot_packages = find_namespace_packages(
        where=str(EXTENSIONS_DIR), include=ROBOTS_PACKAGE_PATTERNS
    )
    benchmark_packages = find_namespace_packages(
        where=str(EXTENSIONS_DIR), include=BENCHMARKS_PACKAGE_PATTERNS
    )
    return sorted(
        set(
            [
                *core_packages,
                *task_packages,
                *robot_packages,
                *benchmark_packages,
                "embodichain_tasks.configs",
            ]
        )
    )


def get_package_dir() -> dict[str, str]:
    """Map bundled extension sources to their stable Python package names."""
    return {
        "embodichain_tasks": "extensions/embodichain_tasks/embodichain_tasks",
        "embodichain_tasks.configs": "extensions/embodichain_tasks/configs",
        "embodichain_robots": "extensions/embodichain_robots",
        "embodichain_benchmarks": "extensions/embodichain_benchmarks",
    }


def get_example_data_files() -> list[tuple[str, list[str]]]:
    """Preserve the ordinary examples tree as installed distribution data.

    Returns:
        Destination directories and source files, without Python package discovery.
    """
    directories: dict[str, list[str]] = {}
    for source in sorted(EXAMPLES_DIR.rglob("*")):
        relative = source.relative_to(EXAMPLES_DIR)
        if (
            not source.is_file()
            or any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            )
            or source.suffix in {".pyc", ".pyo"}
        ):
            continue
        destination = Path("share/embodichain/examples") / relative.parent
        directories.setdefault(destination.as_posix(), []).append(
            source.relative_to(THIS_DIR).as_posix()
        )
    return sorted(directories.items())


def get_version() -> str:
    """Read the complete package version from the repository version file."""
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        return f.read().strip()


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
            "embodichain.data": ["default_materials.json"],
            "embodichain_robots": ["_locomotion_profiles.json"],
            "embodichain_benchmarks.motion_generation": ["suites/*.yaml"],
            "embodichain.gen_sim.simready_pipeline.configs": ["*.json"],
            "embodichain_tasks.configs": [
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "**/README.md",
            ],
        },
        data_files=get_example_data_files(),
        cmdclass=cmdclass,
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
