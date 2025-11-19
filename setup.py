# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import glob
import logging
import os
import shutil
import sys
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


def get_torchsdf_extensions():
    try:
        import torch
        from torch.utils.cpp_extension import (
            BuildExtension as _BuildExtension,
            CppExtension as _CppExtension,
            CUDAExtension as _CUDAExtension,
        )
    except Exception:
        logger.warning(
            "torch or torch.utils.cpp_extension not available; skipping torchsdf extensions"
        )
        return []

    global BuildExtension, CppExtension, CUDAExtension
    BuildExtension = _BuildExtension
    CppExtension = _CppExtension
    CUDAExtension = _CUDAExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []
    include_dirs = []
    sources = glob.glob(
        "embodichain/toolkits/graspkit/dex_grasp/utils/torchsdf/csrc/**/*.cpp",
        recursive=True,
    )
    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        with_cuda = True
        define_macros += [
            ("WITH_CUDA", None),
            ("THRUST_IGNORE_CUB_VERSION_CHECK", None),
        ]
        sources += glob.glob(
            "embodichain/toolkits/graspkit/dex_grasp/utils/torchsdf/csrc/**/*.cu",
            recursive=True,
        )
        extension = CUDAExtension
        extra_compile_args.update(
            {"nvcc": ["-O3", "-DWITH_CUDA", "-DTHRUST_IGNORE_CUB_VERSION_CHECK"]}
        )
        include_dirs = []
    else:
        extension = CppExtension
        with_cuda = False
    extensions = []
    extensions.append(
        extension(
            name="embodichain.toolkits.graspkit.dex_grasp.utils.torchsdf._C",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        )
    )
    for extension in extensions:
        extension.libraries = [
            "cudart_static" if x == "cudart" else x for x in extension.libraries
        ]
    return extensions


# Extract version
here = osp.abspath(osp.dirname(__file__))
version = None
with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    full_version = f.read().strip()
    version = ".".join(full_version.split(".")[:3])

ignore_py = sys.argv[1] == "bdist_nuitka" if len(sys.argv) > 1 else False
data_files = []
data_files += get_data_files_of_a_directory("embodichain", ignore_py=ignore_py)

ext_modules = get_torchsdf_extensions()
cmdclass = {"clean": CleanCommand}
if BuildExtension is not None:
    cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

setup(
    name="embodichain",
    version=version,
    url="http://69.235.177.182:8081/Engine/embodichain",
    author="Dexforce",
    description="A modular platform for building generalized embodied intelligence.",
    packages=find_packages(exclude=["docs"]),
    data_files=data_files,
    entry_points={},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
)
