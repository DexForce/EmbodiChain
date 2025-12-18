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

import glob
import logging
import os
import shutil
import sys
import argparse
from os import path as osp
from pathlib import Path

from setuptools import Command, find_packages, setup

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()

THIS_DIR = Path(__file__).resolve().parent

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
        for d in ["build", "dist", "embodichain.egg-info"]:
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


def get_data_files_of_a_directory(source_dir, target_dir=None, ignore_py=False):
    if target_dir is None:
        target_dir = source_dir

    base_dir = os.sep + "embodichain" + os.sep

    filelist = []
    for parent_dir, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if ignore_py and filename.endswith(".py"):
                continue
            filelist.append(
                (
                    os.path.join(
                        base_dir, parent_dir.replace(source_dir, target_dir, 1)
                    ),
                    [os.path.join(parent_dir, filename)],
                )
            )

    return filelist


def get_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
        full_version = f.read().strip()
        version = ".".join(full_version.split(".")[:3])
    return version


def main():
    # Extract version
    version = get_version()

    data_files = []
    data_files += get_data_files_of_a_directory("embodichain", ignore_py=False)

    cmdclass = {"clean": CleanCommand}
    if BuildExtension is not None:
        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

    setup(
        name="embodichain",
        version=version,
        url="https://github.com/DexForce/EmbodiChain",
        author="EmbodiChain Developers",
        description="An end-to-end, GPU-accelerated, and modular platform for building generalized Embodied Intelligence.",
        packages=find_packages(exclude=["docs"]),
        data_files=data_files,
        entry_points={},
        cmdclass=cmdclass,
        include_package_data=True,
    )

    # Copy VERSION file into the package directory for wheel/sdist
    src_version = os.path.join(THIS_DIR, "VERSION")
    dst_version = os.path.join(THIS_DIR, "embodichain", "VERSION")
    if os.path.exists(src_version):
        shutil.copyfile(src_version, dst_version)
        logger.info(f"Copied VERSION to {dst_version}")


if __name__ == "__main__":
    main()
