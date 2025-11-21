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

import importlib
from typing import List, Union, Optional, Callable, Any


def find_function_from_modules(
    function_name: str, modules: List[Union[str, Any]], raise_if_not_found: bool = True
) -> Optional[Callable]:
    """
    Find a function from multiple Python modules.

    Args:
        function_name (str): Name of the function to find
        modules (List[Union[str, Any]]): List of module names (strings) or module objects
        raise_if_not_found (bool): Whether to raise an exception if function is not found

    Returns:
        Optional[Callable]: The function if found, None otherwise

    Raises:
        AttributeError: If function is not found and raise_if_not_found is True
        ImportError: If a module cannot be imported
    """
    for module in modules:
        try:
            # Handle both module names (strings) and module objects
            if isinstance(module, str):
                mod = importlib.import_module(module)
            else:
                mod = module

            # Check if the function exists in this module
            if hasattr(mod, function_name):
                return getattr(mod, function_name)

        except ImportError as e:
            print(f"Warning: Could not import module {module}: {e}")
            continue

    if raise_if_not_found:
        raise AttributeError(
            f"Function '{function_name}' not found in any of the provided modules: {modules}"
        )

    return None


def find_class_from_modules(
    class_name: str, modules: List[Union[str, Any]], raise_if_not_found: bool = True
) -> Optional[type]:
    """
    Find a class from multiple Python modules.

    Args:
        class_name (str): Name of the class to find
        modules (List[Union[str, Any]]): List of module names (strings) or module objects
        raise_if_not_found (bool): Whether to raise an exception if class is not found

    Returns:
        Optional[type]: The class if found, None otherwise

    Raises:
        AttributeError: If class is not found and raise_if_not_found is True
        ImportError: If a module cannot be imported
    """
    for module in modules:
        try:
            # Handle both module names (strings) and module objects
            if isinstance(module, str):
                mod = importlib.import_module(module)
            else:
                mod = module

            # Check if the class exists in this module
            if hasattr(mod, class_name):
                return getattr(mod, class_name)

        except ImportError as e:
            print(f"Warning: Could not import module {module}: {e}")
            continue

    if raise_if_not_found:
        raise AttributeError(
            f"Class '{class_name}' not found in any of the provided modules: {modules}"
        )

    return None


def get_all_functions_from_module(module: Union[str, Any]) -> dict:
    """
    Get all functions from a module.

    Args:
        module (Union[str, Any]): Module name (string) or module object

    Returns:
        dict: Dictionary mapping function names to function objects
    """
    import inspect

    if isinstance(module, str):
        mod = importlib.import_module(module)
    else:
        mod = module

    functions = {}
    for name, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj):
            functions[name] = obj

    return functions


def find_function_by_pattern(
    pattern: str, modules: List[Union[str, Any]], case_sensitive: bool = True
) -> dict:
    """
    Find functions matching a pattern from multiple modules.

    Args:
        pattern (str): Pattern to match (supports wildcards * and ?)
        modules (List[Union[str, Any]]): List of module names or module objects
        case_sensitive (bool): Whether the search should be case sensitive

    Returns:
        dict: Dictionary mapping module names to dictionaries of matching functions
    """
    import fnmatch

    results = {}

    for module in modules:
        try:
            if isinstance(module, str):
                mod = importlib.import_module(module)
                module_name = module
            else:
                mod = module
                module_name = mod.__name__

            module_functions = get_all_functions_from_module(mod)
            matching_functions = {}

            for func_name, func_obj in module_functions.items():
                if case_sensitive:
                    if fnmatch.fnmatch(func_name, pattern):
                        matching_functions[func_name] = func_obj
                else:
                    if fnmatch.fnmatch(func_name.lower(), pattern.lower()):
                        matching_functions[func_name] = func_obj

            if matching_functions:
                results[module_name] = matching_functions

        except ImportError as e:
            print(f"Warning: Could not import module {module}: {e}")
            continue

    return results


def get_all_exported_items_from_module(module: Union[str, Any]) -> List[str]:
    """
    Get all exported items from a module by checking its __all__ attribute.

    Args:
        module (Union[str, Any]): Module name (string) or module object

    Returns:
        List[str]: List of exported item names
    """
    if isinstance(module, str):
        mod = importlib.import_module(module)
    else:
        mod = module

    if hasattr(mod, "__all__"):
        return list(mod.__all__)
    else:
        # If __all__ is not defined, return all public attributes (not starting with _)
        return [name for name in dir(mod) if not name.startswith("_")]


# Example usage and test functions
if __name__ == "__main__":
    # Example 1: Find a specific function from multiple modules
    modules_to_search = ["math", "os", "sys"]

    # Find 'sqrt' function
    sqrt_func = find_function_from_modules(
        "sqrt", modules_to_search, raise_if_not_found=False
    )
    if sqrt_func:
        print(f"Found sqrt function: {sqrt_func}")
        print(f"sqrt(16) = {sqrt_func(16)}")

    # Example 2: Find functions by pattern
    pattern_results = find_function_by_pattern(
        "*path*", ["os", "os.path"], case_sensitive=False
    )
    print(f"Functions matching '*path*': {pattern_results}")
