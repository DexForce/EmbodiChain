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
import numpy as np

from copy import deepcopy
from typing import List, Union, Optional

from embodichain.utils import logger
from embodichain.lab.gym.utils.misc import validation_with_process_from_name


"""Node Generation Utils"""


def get_init_affordance(scope: str, tag: str = "init") -> str:
    return "{}_{}_qpos".format(scope, tag)


def generate_affordance_from_src(
    env,
    src_key: str,
    dst_key: str,
    valid_funcs_name_kwargs_proc: Optional[List] = None,
    to_array: bool = True,
) -> bool:
    """Generate a new affordance entry in env.affordance_datas by applying a validation and processing
       pipeline to an existing source affordance.

    Args:
        env: The environment object containing affordance data.
        src_key (str): The key of the source affordance in env.affordance_datas.
        dst_key (str): The key to store the generated affordance in env.affordance_datas.
        valid_funcs_name_kwargs_proc (Optional[List]): A list of validation or processing functions (with kwargs)
            to apply to the source affordance. Defaults to an empty list.
        to_array (bool): Whether to convert the result to a numpy array before storing. Defaults to True.

    Returns:
        bool: True if the affordance was successfully generated and stored, False otherwise.
    """
    if valid_funcs_name_kwargs_proc is None:
        valid_funcs_name_kwargs_proc = []
    try:
        result = validation_with_process_from_name(
            env,
            deepcopy(env.affordance_datas[src_key]),
            valid_funcs_name_kwargs_proc,
        )
        if result is None:
            logger.log_warning(f"Failed to generate {dst_key} from {src_key}")
            return False

        env.affordance_datas[dst_key] = np.asarray(result) if to_array else result
        return True
    except Exception as e:
        logger.log_error(f"Affordance generation error: {e}")
        return False
