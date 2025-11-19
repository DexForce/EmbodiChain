# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
import torch

from typing import Sequence, Union, Dict, Literal


Array = Union[torch.Tensor, np.ndarray, Sequence]
Device = Union[str, torch.device]

EnvObs = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]

EnvAction = Union[torch.Tensor, Dict[str, torch.Tensor]]
