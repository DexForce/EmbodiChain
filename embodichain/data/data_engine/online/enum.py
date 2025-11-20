# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from enum import Enum


class ConsumerTeleEnum(Enum):
    SHAKEHAND = "Data is ready?"
    CONSUME = "Fetch data!"
    NOCONSUME = "Data_pool is full."
    GOTDATA = "Feched data!"
    NOGOTDATA = "Not fetching data."


class ProducerTeleEnum(Enum):
    READY = "Yes"
    NOREADY = "No ready"
    FULL = "Data_pool is full"
    FAIL = "Failed"
    SEND = "Send!"
    EMPTYSTR = "Empty String."
