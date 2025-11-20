# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os

database_dir = os.path.dirname(os.path.abspath(__file__)).replace("data", "database")
video_dir = os.path.join(database_dir, "video")
weights_dir = os.path.join(database_dir, "weights")
database_2d_dir = os.path.join(database_dir, "2dasset")
database_lang_dir = os.path.join(database_dir, "lang")
database_demo_dir = os.path.join(database_dir, "demostration")
database_tmp_dir = os.path.join(database_dir, "tmp")
database_train_dir = os.path.join(database_dir, "train")


if not os.path.exists(database_tmp_dir):
    os.makedirs(database_tmp_dir, exist_ok=True)
if not os.path.exists(database_train_dir):
    os.makedirs(database_train_dir, exist_ok=True)

from . import assets
from .dataset import *
