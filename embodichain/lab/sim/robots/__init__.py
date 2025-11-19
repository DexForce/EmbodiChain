# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

try:
    from .robot import *
    from .manipulator import *
    from .dexterous_hands import *
    from .humanoids import *
    from .dexforce_w1 import *
    from .cobotmagic import CobotMagicCfg

    del robot
    del manipulator
    del dexterous_hands
    del humanoids
    del dexforce_w1

except ImportError as e:
    from dexsim.utility import log_warning

    log_warning(e)
