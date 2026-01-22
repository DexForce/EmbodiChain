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

from embodichain.agents.hierarchy.agent_base import AgentBase
from langchain_core.prompts import ChatPromptTemplate
import os
import numpy as np
import functools
from typing import Dict, Tuple, Any
from embodichain.toolkits.code_generation import (
    ExecutableOutputParser,
    OutputFormatting,
)
from embodichain.toolkits.toolkits import ToolkitsBase
from embodichain.agents.mllm.prompt import CodePrompt
from embodichain.data import database_agent_prompt_dir
from pathlib import Path
import re
import importlib.util
from langchain_core.messages import HumanMessage
from datetime import datetime
from embodichain.utils.utility import encode_image
import base64


def format_execution_history(execution_history):
    if not execution_history or len(execution_history) == 0:
        return "None."

    return "\n\n".join(f"{i}. {entry}" for i, entry in enumerate(execution_history, 1))


def extract_python_code_and_text(llm_response: str) -> Tuple[str, str]:
    """
    Extract exactly ONE python code block from the LLM response,
    and return:
      - code: the content inside the python block
      - text: all remaining explanation text (outside the code block)

    Raises ValueError if zero or multiple python blocks are found.
    """

    pattern = r"```python\s*(.*?)\s*```"
    matches = list(re.finditer(pattern, llm_response, re.DOTALL))

    if len(matches) == 0:
        raise ValueError("No python code block found in LLM response.")
    if len(matches) > 1:
        raise ValueError("Multiple python code blocks found in LLM response.")

    match = matches[0]
    code = match.group(1).strip()

    # Optional sanity check
    if not code.startswith("#") and not code.startswith("drive("):
        raise ValueError(
            f"Invalid code block content. Expected `drive(...)` or `# TASK_COMPLETE`, got:\n{code}"
        )

    # Extract remaining text (before + after the code block)
    text_before = llm_response[: match.start()].strip()
    text_after = llm_response[match.end() :].strip()

    explanation_text = "\n\n".join(part for part in [text_before, text_after] if part)

    return code, explanation_text


def format_llm_response_md(
    llm_analysis: str,  # plain-text explanation (NO code)
    extracted_code: str,  # validated executable code
    step_id: int = None,
    execution_history: str = None,
    obs_image_path: Path = None,
    md_file_path: Path = None,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"## Step: {step_id if step_id is not None else '-'} | {ts}\n\n"

    history_block = ""
    if execution_history:
        history_block = (
            "### Execution History (Input to LLM)\n\n"
            "```\n"
            f"{execution_history}\n"
            "```\n\n"
        )

    image_block = ""
    if obs_image_path is not None and md_file_path is not None:
        try:
            rel_path = obs_image_path.relative_to(md_file_path.parent)
        except ValueError:
            # Fallback: just use filename
            rel_path = obs_image_path.name

        image_block = (
            "### Observation Image\n\n" f"![]({Path(rel_path).as_posix()})\n\n"
        )

    body = (
        image_block + history_block + "### LLM Analysis\n\n"
        f"{llm_analysis.strip()}\n\n"
        "### Executed Code\n\n"
        "```python\n"
        f"{extracted_code.strip()}\n"
        "```\n\n"
        "---\n\n"
    )

    return header + body


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

        # insert feedback if exists
        if len(kwargs.get("error_messages", [])) != 0:
            # just use the last one
            last_code = kwargs["generated_codes"][-1]
            last_error = kwargs["error_messages"][-1]
            last_observation = (
                kwargs.get("observation_feedbacks")[-1]
                if kwargs.get("observation_feedbacks")
                else None
            )

            # Add extra human message with feedback
            feedback_msg = self.build_feedback_message(
                last_code, last_error, last_observation
            )
            prompt.messages.append(feedback_msg)

        llm_code = self.llm.invoke(prompt)

        # Normalize content
        llm_code = getattr(llm_code, "content", str(llm_code))

        print(f"\033[92m\nCode agent output:\n{llm_code}\n\033[0m")

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
        import ast

        # Read the generated code file
        with open(code_file_path, "r") as f:
            code_content = f.read()

        # Build execution namespace with necessary imports
        ns = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__file__": str(code_file_path),
            "kwargs": kwargs,  # Make kwargs available for injection
        }

        # Import toolkit functions into namespace
        try:
            exec(
                "from embodichain.toolkits.interfaces import *",
                ns,
                ns,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to import embodichain.toolkits.interfaces"
            ) from e

        # Parse code to check if it defines a function or contains module-level calls
        tree = ast.parse(code_content)

        # Check if code defines create_agent_action_list function
        has_function = any(
            isinstance(node, ast.FunctionDef)
            and node.name == "create_agent_action_list"
            for node in tree.body
        )

        if has_function:
            # Execute code (function will be defined in namespace)
            exec(code_content, ns, ns)

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
            if "env" in kwargs:
                env = kwargs["env"]
                if hasattr(env, "_episode_action_list") and env._episode_action_list:
                    print(
                        f"Collected {len(env._episode_action_list)} actions from module-level drive() calls."
                    )
                    return env._episode_action_list

            print("Code executed successfully, but no actions were collected.")
            return []
