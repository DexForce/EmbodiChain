# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from pathlib import Path
import logging


def ensure_directory_exists(path: str, logger: logging.Logger = None):
    """Ensure the directory exists, create if not."""
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if logger:
            logger.error(f"Failed to create directory {path}: {e}")
        else:
            print(f"Failed to create directory {path}: {e}")
