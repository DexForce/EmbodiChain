# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os

embodichain_dir = os.path.dirname(__file__)

# Read version from VERSION file
def _get_version():
    version_file = os.path.join(os.path.dirname(embodichain_dir), "VERSION")
    try:
        with open(version_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("VERSION file not found.")
        return "unknown"


__version__ = _get_version()
