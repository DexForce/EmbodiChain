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

from embodichain.agents.hierarchy.agent_base import AgentBase
from langchain_core.prompts import ChatPromptTemplate
import os
import numpy as np
from typing import Dict, Tuple
from embodichain.agents.mllm.prompt import CodePrompt
from embodichain.data import database_agent_prompt_dir
from pathlib import Path
import re
import importlib.util
from datetime import datetime
import ast

class NormalizePartialMonitors(ast.NodeTransformer):
    """Rewrite partial(monitor_fn(...)) into partial(monitor_fn, ...)."""

    def visit_Call(self, node):
        self.generic_visit(node)

        is_partial_call = (
                isinstance(node.func, ast.Name) and node.func.id == "partial"
        )
        if not is_partial_call or not node.args:
            return node

        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Call):
            return node

        normalized_args = [first_arg.func, *first_arg.args, *node.args[1:]]
        normalized_keywords = [*first_arg.keywords, *node.keywords]

        return ast.copy_location(
            ast.Call(
                func=node.func,
                args=normalized_args,
                keywords=normalized_keywords,
            ),
            node,
        )

class CodeAgent(AgentBase):
    query_prefix = "# "
    query_suffix = "."
    prompt: ChatPromptTemplate
    prompt_kwargs: Dict[str, Dict]

    def __init__(self, llm, **kwargs) -> None:
        super().__init__(**kwargs)
        if llm is None:
            raise ValueError(
                "LLM is None. Please set the following environment variables:\n"
                "  - AZURE_OPENAI_ENDPOINT\n"
                "  - AZURE_OPENAI_API_KEY\n"
                "Example:\n"
                "  export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'\n"
                "  export AZURE_OPENAI_API_KEY='your-api-key'"
            )
        self.llm = llm

    def generate(self, **kwargs):
        log_dir = kwargs.get(
            "log_dir", Path(database_agent_prompt_dir) / self.task_name
        )
        file_path = log_dir / "agent_generated_code.py"

        # Check if the file already exists
        if not kwargs.get("regenerate", False):
            if file_path.exists():
                print(f"Code file already exists at {file_path}, skipping writing.")
                return file_path, kwargs, None

        # Generate code via LLM
        prompt = getattr(CodePrompt, self.prompt_name)(
            **kwargs,
        )
        llm_code = self.llm.invoke(prompt)

        # Normalize content
        llm_code = getattr(llm_code, "content", str(llm_code))
        print(f"\033[94m\nCode agent output:\n{llm_code}\n\033[0m")

        # Write the code to the file if it does not exist
        match = re.search(r"```python\n(.*?)\n```", llm_code, re.DOTALL)
        if match:
            code_to_save = match.group(1).strip()
        else:
            code_to_save = llm_code.strip()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(code_to_save)
        print(f"Generated function code saved to {file_path}")

        return file_path, kwargs, code_to_save

    def act(self, code_file_path, **kwargs):
        """Execute generated code with proper execution environment.

        Supports two modes:
        1. If code defines 'create_agent_action_list' function, call it
        2. If code contains module-level drive() calls, execute them directly
        """

        # Read the generated code file
        with open(code_file_path, "r") as f:
            code_content = f.read()

        # Keep only runtime kwargs for execution. Prompt/template contents should not be
        # forwarded into generated function calls, otherwise prompt keys can collide
        # with explicit drive call arguments such as `monitor_sequences`.
        prompt_only_keys = set(getattr(self, "prompt_kwargs", {}).keys())
        prompt_only_keys.update(
            {
                "task_plan",
                "anticipated_failures",
                "observations",
            }
        )
        runtime_kwargs = {
            key: value for key, value in kwargs.items() if key not in prompt_only_keys
        }

        # Build execution namespace with necessary imports
        ns = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(code_file_path),
            "kwargs": runtime_kwargs,  # Make runtime kwargs available for injection
        }

        # Import atom action functions into namespace
        try:
            exec(
                "from embodichain.lab.sim.agent.atom_actions import *",
                ns,
                ns,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to import embodichain.lab.sim.agent.atom_actions"
            ) from e

        # Parse code to check if it defines a function or contains module-level calls
        tree = ast.parse(code_content)
        tree = NormalizePartialMonitors().visit(tree)
        ast.fix_missing_locations(tree)

        # Check if code defines create_agent_action_list function
        has_function = any(
            isinstance(node, ast.FunctionDef)
            and node.name == "create_agent_action_list"
            for node in tree.body
        )

        if has_function:
            # Execute code (function will be defined in namespace)
            compiled_code = compile(tree, filename=str(code_file_path), mode="exec")
            exec(compiled_code, ns, ns)

            # Call the function if it exists
            if "create_agent_action_list" in ns:
                result = ns["create_agent_action_list"](**kwargs)
                print("Function executed successfully.")
                return result
            else:
                raise AttributeError(
                    "The function 'create_agent_action_list' was not found after execution."
                )
        else:
            # Code contains module-level drive() calls
            # AST transformer to inject **kwargs into function calls
            class InjectKwargs(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    # Inject **kwargs if not present
                    has_kwargs = any(
                        kw.arg is None
                        and isinstance(kw.value, ast.Name)
                        and kw.value.id == "kwargs"
                        for kw in node.keywords
                    )
                    if not has_kwargs:
                        node.keywords.append(
                            ast.keyword(
                                arg=None, value=ast.Name(id="kwargs", ctx=ast.Load())
                            )
                        )
                    return node

            # Transform AST to inject kwargs
            tree = InjectKwargs().visit(tree)
            ast.fix_missing_locations(tree)

            # Compile and execute transformed code
            compiled_code = compile(tree, filename=str(code_file_path), mode="exec")
            exec(compiled_code, ns, ns)

            # Collect actions from drive() calls if they were executed
            # drive() function stores actions in env._episode_action_list
            if "env" in runtime_kwargs:
                env = runtime_kwargs["env"]
                if hasattr(env, "_episode_action_list") and env._episode_action_list:
                    print(
                        f"Collected {len(env._episode_action_list)} actions from module-level drive() calls."
                    )
                    return env._episode_action_list

            print("Code executed successfully, but no actions were collected.")
            return []
