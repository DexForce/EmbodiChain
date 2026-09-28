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

import os
import importlib
import time
from contextlib import contextmanager
from typing import Iterator

from embodichain.utils import logger


@contextmanager
def _dataset_download_lock(path: str, prefix: str) -> Iterator[None]:
    """Serialize download and extraction for one dataset cache entry.

    Pytest-xdist workers and separate user processes can request the same
    dataset simultaneously. The lock lives outside the per-dataset download
    directory because failed-download cleanup removes that directory.

    Args:
        path: Root of the EmbodiChain data cache.
        prefix: Dataset cache directory name.
    """
    lock_dir = os.path.join(path, "download", ".locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{prefix}.lock")

    with open(lock_path, "a+") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write("0")
                lock_file.flush()
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def __getattr__(name: str):
    if name == "EmbodiChainDataset":
        from ._dataset_download import EmbodiChainDataset

        globals()[name] = EmbodiChainDataset
        return EmbodiChainDataset
    raise AttributeError(name)


DEFAULT_DATA_MODULES = [
    "embodichain.data",
    "embodichain.data.assets",
]


def get_data_class(dataset_name: str, extra_modules: list[str] | None = None):
    """Retrieve the dataset class from the available modules.

    Args:
        dataset_name (str): The name of the dataset class.
        extra_modules (list[str] | None): Optional list of additional module names to search for the dataset class.

    Returns:
        type: The dataset class.

    Raises:
        AttributeError: If the dataset class is not found in any module.
    """
    module_names = DEFAULT_DATA_MODULES + (
        extra_modules if extra_modules is not None else []
    )

    for module_name in module_names:
        try:
            return getattr(importlib.import_module(module_name), dataset_name)
        except AttributeError:
            continue

    raise AttributeError(f"Dataset class '{dataset_name}' not found in any module.")


def get_data_path(data_path_in_config: str) -> str:
    """Get the absolute path of the data file.

    Resolution order:
        1. If ``data_path_in_config`` is an absolute path, return it directly.
        2. If a matching file/directory exists under ``EMBODICHAIN_DEFAULT_DATA_ROOT``
           (which can be overridden via the ``EMBODICHAIN_DATA_ROOT`` environment
           variable), return that path.
        3. Otherwise, resolve via the registered data-class download mechanism.

    Args:
        data_path_in_config (str): The dataset name, optionally followed by a
            subpath in the format ``"dataset_name/subpath"``.

    Returns:
        str: The absolute path of the data file.
    """
    if os.path.isabs(data_path_in_config):
        return data_path_in_config

    # Try resolving under the user-configurable data root first
    from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

    local_path = os.path.join(EMBODICHAIN_DEFAULT_DATA_ROOT, data_path_in_config)
    if os.path.exists(local_path):
        return local_path

    extracted_path = os.path.join(
        EMBODICHAIN_DEFAULT_DATA_ROOT, "extract", data_path_in_config
    )
    if os.path.exists(extracted_path):
        return extracted_path

    # Fall back to the data-class download mechanism
    dataset_name, *sub_path_parts = data_path_in_config.split("/")

    data_class = get_data_class(dataset_name)
    data_obj = data_class()
    data_dir = data_obj.extract_dir
    return os.path.join(data_dir, *sub_path_parts)
