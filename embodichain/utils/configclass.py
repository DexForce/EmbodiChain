# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
# All rights reserved.
#
# This file incorporates code from the Isaac Lab Project
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ----------------------------------------------------------------------------


import torch
import inspect
import types
from collections.abc import Callable, Mapping, Iterable, Sized
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, replace
from typing import Any, ClassVar, Optional
from .string import callable_to_string, string_to_callable


_CONFIGCLASS_METHODS = ["to_dict", "replace", "copy", "validate"]
"""List of class methods added at runtime to dataclass."""

"""
Wrapper around dataclass.
"""


def __dataclass_transform__():
    """Add annotations decorator for PyLance."""
    return lambda a: a


def is_configclass(cls: Any) -> bool:
    """Check if a class is a configclass.

    Args:
        cls: The class to check.

    Returns:
        True if the class is a configclass, False otherwise.
    """
    return hasattr(cls, "validate")


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """Wrapper around `dataclass` functionality to add extra checks and utilities.

    As of Python 3.7, the standard dataclasses have two main issues which makes them non-generic for
    configuration use-cases. These include:

    1. Requiring a type annotation for all its members.
    2. Requiring explicit usage of :meth:`field(default_factory=...)` to reinitialize mutable variables.

    This function provides a decorator that wraps around Python's `dataclass`_ utility to deal with
    the above two issues. It also provides additional helper functions for dictionary <-> class
    conversion and easily copying class instances.

    Usage:

    .. code-block:: python

        from dataclasses import MISSING

        from isaaclab.utils.configclass import configclass


        @configclass
        class ViewerCfg:
            eye: list = [7.5, 7.5, 7.5]  # field missing on purpose
            lookat: list = field(default_factory=[0.0, 0.0, 0.0])


        @configclass
        class EnvCfg:
            num_envs: int = MISSING
            episode_length: int = 2000
            viewer: ViewerCfg = ViewerCfg()

        # create configuration instance
        env_cfg = EnvCfg(num_envs=24)

        # print information as a dictionary
        print(env_cfg.to_dict())

        # create a copy of the configuration
        env_cfg_copy = env_cfg.copy()

        # replace arbitrary fields using keyword arguments
        env_cfg_copy = env_cfg_copy.replace(num_envs=32)

    Args:
        cls: The class to wrap around.
        **kwargs: Additional arguments to pass to :func:`dataclass`.

    Returns:
        The wrapped class.

    .. _dataclass: https://docs.python.org/3/library/dataclasses.html

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/configclass.py
    """
    # add type annotations
    _add_annotation_types(cls)
    # add field factory
    _process_mutable_types(cls)
    # copy mutable members
    # note: we check if user defined __post_init__ function exists and augment it with our own
    if hasattr(cls, "__post_init__"):
        setattr(
            cls, "__post_init__", combined_function(cls.__post_init__, custom_post_init)
        )
    else:
        setattr(cls, "__post_init__", custom_post_init)
    # add helper functions for dictionary conversion
    setattr(cls, "to_dict", class_to_dict)
    # setattr(cls, "from_dict", update_class_from_dict)
    setattr(cls, "replace", _replace_class_with_kwargs)
    setattr(cls, "copy", _replace_class_with_kwargs)
    setattr(cls, "validate", _validate)
    # wrap around dataclass
    cls = dataclass(cls, **kwargs)
    # return wrapped class
    return cls


def combined_function(f1: Callable, f2: Callable) -> Callable:
    """Combine two functions into one.

    Args:
        f1: The first function.
        f2: The second function.

    Returns:
        The combined function.
    """

    def _combined(*args, **kwargs):
        # call both functions
        f1(*args, **kwargs)
        f2(*args, **kwargs)

    return _combined


def custom_post_init(obj):
    """Deepcopy all elements to avoid shared memory issues for mutable objects in dataclasses initialization.

    This function is called explicitly instead of as a part of :func:`_process_mutable_types()` to prevent mapping
    proxy type i.e. a read only proxy for mapping objects. The error is thrown when using hierarchical data-classes
    for configuration.
    """
    for key in dir(obj):
        # skip dunder members
        if key.startswith("__"):
            continue
        # get data member
        value = getattr(obj, key)
        # check annotation
        ann = obj.__class__.__dict__.get(key)
        # duplicate data members that are mutable
        if not callable(value) and not isinstance(ann, property):
            try:
                setattr(obj, key, deepcopy(value))
            except AttributeError as e:
                from IPython import embed

                embed()


def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        # We have to treat torch tensors specially because `torch.tensor.__dict__` returns an empty
        # dict, which would mean that a torch.tensor would be stored as an empty dict. Instead we
        # want to store it directly as the tensor.
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # check if attribute is a list or tuple
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key

        # -- A) if key is present in the object ------------------------------------
        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)

            # -- 1) nested mapping → recurse ---------------------------
            if isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue

            # -- 2) iterable (list / tuple / etc.) ---------------------
            if isinstance(value, Iterable) and not isinstance(value, str):

                # ---- 2a) flat iterable → replace wholesale ----------
                if all(not isinstance(el, Mapping) for el in value):
                    out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                    if isinstance(obj, dict):
                        obj[key] = out_val
                    else:
                        setattr(obj, key, out_val)
                    continue

                # ---- 2b) existing value is None → abort -------------
                if obj_mem is None:
                    raise ValueError(
                        f"[Config]: Cannot merge list under namespace: {key_ns} because the existing value is None."
                    )

                # ---- 2c) length mismatch → abort -------------------
                if (
                    isinstance(obj_mem, Sized)
                    and isinstance(value, Sized)
                    and len(obj_mem) != len(value)
                ):
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )

                # ---- 2d) keep tuple/list parity & recurse ----------
                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True
                    # recursively call if iterable contains Mappings
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    # do not set value to obj, otherwise it overwrites the cfg class with the dict
                    if not set_obj:
                        continue

            # -- 3) callable attribute → resolve string --------------
            elif callable(obj_mem):
                # update function name
                value = string_to_callable(value)

            # -- 4) simple scalar / explicit None ---------------------
            elif value is None or isinstance(value, type(obj_mem)):
                pass

            # -- 5) type mismatch → abort -----------------------------
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )

            # -- 6) final assignment ---------------------------------
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)

        # -- B) if key is not present ------------------------------------
        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")


def _replace_class_with_kwargs(obj: object, **kwargs) -> object:
    """Return a new object replacing specified fields with new values.

    This is especially useful for frozen classes.  Example usage:

    .. code-block:: python

        @configclass(frozen=True)
        class C:
            x: int
            y: int

        c = C(1, 2)
        c1 = c.replace(x=3)
        assert c1.x == 3 and c1.y == 2

    Args:
        obj: The object to replace.
        **kwargs: The fields to replace and their new values.

    Returns:
        The new object.
    """
    return replace(obj, **kwargs)


def _validate(obj: object, prefix: str = "") -> list[str]:
    """Check the validity of configclass object.

    This function checks if the object is a valid configclass object. A valid configclass object contains no MISSING
    entries.

    Args:
        obj: The object to check.
        prefix: The prefix to add to the missing fields. Defaults to ''.

    Returns:
        A list of missing fields.

    Raises:
        TypeError: When the object is not a valid configuration object.
    """
    missing_fields = []

    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_validate(item, prefix=current_path))
        return missing_fields
    elif isinstance(obj, dict):
        obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return missing_fields

    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_validate(value, prefix=current_path))

    # raise an error only once at the top-level call
    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    return missing_fields


def _add_annotation_types(cls):
    """Add annotations to all elements in the dataclass.

    By definition in Python, a field is defined as a class variable that has a type annotation.

    In case type annotations are not provided, dataclass ignores those members when :func:`__dict__()` is called.
    This function adds these annotations to the class variable to prevent any issues in case the user forgets to
    specify the type annotation.

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos = (0.0, 0.0, 0.0)
           ^^
           If the function is NOT used, the following type-error is returned:
           TypeError: 'pos' is a field but has no type annotation
    """
    # get type hints
    hints = {}
    # iterate over class inheritance
    # we add annotations from base classes first
    for base in reversed(cls.__mro__):
        # check if base is object
        if base is object:
            continue
        # get base class annotations
        ann = base.__dict__.get("__annotations__", {})
        # directly add all annotations from base class
        hints.update(ann)
        # iterate over base class members
        # Note: Do not change this to dir(base) since it orders the members alphabetically.
        #   This is not desirable since the order of the members is important in some cases.
        for key in base.__dict__:
            # get class member
            value = getattr(base, key)
            # skip members
            if _skippable_class_member(key, value, hints):
                continue
            # add type annotations for members that don't have explicit type annotations
            # for these, we deduce the type from the default value
            if not isinstance(value, type):
                if key not in hints:
                    # check if var type is not MISSING
                    # we cannot deduce type from MISSING!
                    if value is MISSING:
                        raise TypeError(
                            f"Missing type annotation for '{key}' in class '{cls.__name__}'."
                            " Please add a type annotation or set a default value."
                        )
                    # add type annotation
                    hints[key] = type(value)
            elif key != value.__name__:
                # note: we don't want to add type annotations for nested configclass. Thus, we check if
                #   the name of the type matches the name of the variable.
                # since Python 3.10, type hints are stored as strings
                hints[key] = f"type[{value.__name__}]"

    # Note: Do not change this line. `cls.__dict__.get("__annotations__", {})` is different from
    #   `cls.__annotations__` because of inheritance.
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    cls.__annotations__ = hints


def _process_mutable_types(cls):
    """Initialize all mutable elements through :obj:`dataclasses.Field` to avoid unnecessary complaints.

    By default, dataclass requires usage of :obj:`field(default_factory=...)` to reinitialize mutable objects every time a new
    class instance is created. If a member has a mutable type and it is created without specifying the `field(default_factory=...)`,
    then Python throws an error requiring the usage of `default_factory`.

    Additionally, Python only explicitly checks for field specification when the type is a list, set or dict. This misses the
    use-case where the type is class itself. Thus, the code silently carries a bug with it which can lead to undesirable effects.

    This function deals with this issue

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           If the function is NOT used, the following value-error is returned:
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory
    """
    # note: Need to set this up in the same order as annotations. Otherwise, it
    #   complains about missing positional arguments.
    ann = cls.__dict__.get("__annotations__", {})

    # iterate over all class members and store them in a dictionary
    class_members = {}
    for base in reversed(cls.__mro__):
        # check if base is object
        if base is object:
            continue
        # iterate over base class members
        for key in base.__dict__:
            # get class member
            f = getattr(base, key)
            # skip members
            if _skippable_class_member(key, f):
                continue
            # store class member if it is not a type or if it is already present in annotations
            if not isinstance(f, type) or key in ann:
                class_members[key] = f
        # iterate over base class data fields
        # in previous call, things that became a dataclass field were removed from class members
        # so we need to add them back here as a dataclass field directly
        for key, f in base.__dict__.get("__dataclass_fields__", {}).items():
            # store class member
            if not isinstance(f, type):
                class_members[key] = f

    # check that all annotations are present in class members
    # note: mainly for debugging purposes
    if len(class_members) != len(ann):
        raise ValueError(
            f"In class '{cls.__name__}', number of annotations ({len(ann)}) does not match number of class members"
            f" ({len(class_members)}). Please check that all class members have type annotations and/or a default"
            " value. If you don't want to specify a default value, please use the literal `dataclasses.MISSING`."
        )
    # iterate over annotations and add field factory for mutable types
    for key in ann:
        # find matching field in class
        value = class_members.get(key, MISSING)
        # check if key belongs to ClassVar
        # in that case, we cannot use default_factory!
        origin = getattr(ann[key], "__origin__", None)
        if origin is ClassVar:
            continue
        # check if f is MISSING
        # note: commented out for now since it causes issue with inheritance
        #   of dataclasses when parent have some positional and some keyword arguments.
        # Ref: https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
        # TODO: check if this is fixed in Python 3.10
        # if f is MISSING:
        #     continue
        if isinstance(value, Field):
            setattr(cls, key, value)
        elif not isinstance(value, type):
            # create field factory for mutable types
            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)


def _skippable_class_member(key: str, value: Any, hints: Optional[dict] = None) -> bool:
    """Check if the class member should be skipped in configclass processing.

    The following members are skipped:

    * Dunder members: ``__name__``, ``__module__``, ``__qualname__``, ``__annotations__``, ``__dict__``.
    * Manually-added special class functions: From :obj:`_CONFIGCLASS_METHODS`.
    * Members that are already present in the type annotations.
    * Functions bounded to class object or class.
    * Properties bounded to class object.

    Args:
        key: The class member name.
        value: The class member value.
        hints: The type hints for the class. Defaults to None, in which case, the
            members existence in type hints are not checked.

    Returns:
        True if the class member should be skipped, False otherwise.
    """
    # skip dunder members
    if key.startswith("__"):
        return True
    # skip manually-added special class functions
    if key in _CONFIGCLASS_METHODS:
        return True
    # check if key is already present
    if hints is not None and key in hints:
        return True
    # skip functions bounded to class
    if callable(value):
        # FIXME: This doesn't yet work for static methods because they are essentially seen as function types.
        # check for class methods
        if isinstance(value, types.MethodType):
            return True
        # check for instance methods
        signature = inspect.signature(value)
        if "self" in signature.parameters or "cls" in signature.parameters:
            return True
    # skip property methods
    if isinstance(value, property):
        return True
    # Otherwise, don't skip
    return False


def _return_f(f: Any) -> Callable[[], Any]:
    """Returns default factory function for creating mutable/immutable variables.

    This function should be used to create default factory functions for variables.

    Example:

        .. code-block:: python

            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)
    """

    def _wrap():
        if isinstance(f, Field):
            if f.default_factory is MISSING:
                return deepcopy(f.default)
            else:
                return f.default_factory
        else:
            return deepcopy(f)

    return _wrap
