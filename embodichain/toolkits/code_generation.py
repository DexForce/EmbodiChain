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

from typing import Dict, Tuple
from langchain_core.output_parsers import BaseOutputParser

import numpy as np


def merge_dicts(dicts: Dict):
    return {k: v for d in dicts for k, v in d.items()}


def get_executable_code_str(input_string, language="python"):
    start_marker = f"```{language}"
    end_marker = f"```"
    if input_string.find(start_marker) >= 0:

        start_index = input_string.find(start_marker) + len(start_marker)
        end_index = input_string.rfind(end_marker)

        code_string = input_string[start_index:end_index].strip()
    else:
        code_string = input_string

    return code_string


class OutputFormatting:
    @staticmethod
    def flatten_dict(output: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        ret = {}
        for _, val in output.items():
            ret.update(val)
        return ret


class ExecutableOutputParser(BaseOutputParser):
    # https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/custom/

    _fixed_vars = {"np": np}
    variable_vars = {}

    def update_vars(self, variable_vars: Dict):
        self.variable_vars = variable_vars

    def parse(self, text: str) -> Tuple[str, Dict, Dict]:
        code_str = get_executable_code_str(text)
        # if self._cfg["include_context"] and context != "":
        #     to_exec = f"{context}\n{code_str}"
        #     to_log = f"{context}\n{use_query}\n{code_str}"
        # else:
        #     to_exec = code_str
        #     to_log = f"{use_query}\n{to_exec}"

        # to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        # print(
        #     f"\033[34m====================================================\nLMP {self._name} exec:\033[0m\n\n{to_log_pretty}\n\n\033[34m====================================================\n\033[0m"
        # )

        # generate new functions
        # new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        # self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self.variable_vars])
        lvars = None

        if gvars is None:
            gvars = {}
        if lvars is None:
            lvars = {}
        empty_fn = lambda *args, **kwargs: None
        custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])

        return code_str, custom_gvars, lvars
