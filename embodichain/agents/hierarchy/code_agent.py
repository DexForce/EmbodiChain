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
        # Dynamically import the generated function from the .py file
        spec = importlib.util.spec_from_file_location(
            "generated_function", code_file_path
        )
        generated_function_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_function_module)

        # Ensure that the function exists and call it with kwargs
        if hasattr(generated_function_module, "create_agent_action_list"):
            result = generated_function_module.create_agent_action_list(
                **kwargs
            )  # Call the function with kwargs
            print("Function executed successfully.")
            return result
        else:
            raise AttributeError(
                "The function 'create_agent_action_list' was not found in the generated code."
            )

    def build_feedback_message(
        self, last_code: str, last_error: str, last_observation: str = None
    ) -> HumanMessage:

        useful_info = (
            "The error may be caused by:\n"
            "1. You did not follow the basic background information, especially the world coordinate system with its xyz directions.\n"
            "2. You did not take into account the NOTE given in the atomic actions or in the example functions.\n"
            "3. You did not follow the steps of the task descriptions.\n"
        )

        # Optional observation section
        observation_text = ""
        if last_observation is not None:
            observation_text = (
                "\nThe visual observation feedback of the execution process was:\n"
                "```\n" + str(last_observation) + "\n```\n"
            )

        return HumanMessage(
            content=(
                "Your previously generated code was:\n"
                "```\n" + last_code + "\n```\n\n"
                "When this code was executed in the test environment, it failed with the following error:\n"
                "```\n"
                + last_error
                + "```\n"
                + observation_text
                + "\n"
                + useful_info
                + "\nAnalyze the cause of the failure and produce a corrected version of the code. "
                "Modify only what is necessary to fix the issue. The corrected code must:\n"
                " - strictly use only the allowed atomic API functions,\n"
                " - be executable and unambiguous,\n"
                " - directly resolve the error shown above.\n\n"
                "Your entire response must be EXACTLY one Python code block:\n"
                "```python\n"
                "# corrected solution code\n"
                "```\n"
            )
        )

    def generate_according_to_task_plan(self, task_plan, **kwargs):
        # Generate code via LLM
        prompt = getattr(CodePrompt, self.prompt_name)(task_plan=task_plan, **kwargs)

        llm_code = self.llm.invoke(prompt)
        llm_code = getattr(llm_code, "content", str(llm_code))

        match = re.search(r"```python\n(.*?)\n```", llm_code, re.DOTALL)
        if match:
            llm_code = match.group(1).strip()
        else:
            llm_code = llm_code.strip()

        print(f"\033[92m\nCode agent output:\n{llm_code}\n\033[0m")

        return kwargs, llm_code

    def act_single_action(self, code: str, **kwargs):
        import ast

        # ---- 0. Build execution namespace ----
        ns = {
            "__builtins__": __builtins__,
            "kwargs": kwargs,  # visible for **kwargs injection
        }

        # ---- 1. Executor-controlled import ----
        try:
            exec(
                "from embodichain.toolkits.interfaces import *",
                ns,
                ns,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to import embodichain.toolkits.interfaces in act_single_action"
            ) from e

        # ---- 2. Parse generated code ----
        tree = ast.parse(code)
        body = tree.body

        # ---------- AST transformer: inject **kwargs everywhere ----------
        class InjectKwargs(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)

                # Check if **kwargs already exists
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

        transformer = InjectKwargs()

        # ---- 3. Execute actions step by step ----
        for step_id, node in enumerate(body, start=1):
            try:
                node = transformer.visit(node)
                ast.fix_missing_locations(node)

                step_mod = ast.Module([node], type_ignores=[])
                compiled = compile(
                    step_mod, filename=f"<generated_step_{step_id}>", mode="exec"
                )

                print(
                    f"\033[95m\nExecuting the current action {code} with **kwargs\033[0m"
                )
                exec(compiled, ns, ns)

            except Exception as e:
                raise RuntimeError(
                    f"Execution failed at step {step_id} with action {code}:\n{e}"
                )

        print("\033[95m\nThe current action step executed successfully.\033[0m")
