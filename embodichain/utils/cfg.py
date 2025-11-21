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

from __future__ import annotations

import functools
import inspect
import logging

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

from fvcore.common.config import BASE_KEY
from fvcore.common.config import CfgNode as _CfgNode
from iopath.common.file_io import PathManager as PathManagerBase
from yacs.config import _VALID_TYPES, _assert_with_logging

sep = "."
prefix = ""
PathManager = PathManagerBase()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _flatten_dict(
    src: Dict,
    prefix: Optional[str] = prefix,
    sep: Optional[str] = sep,
    dct: Optional[Dict] = {},
) -> Dict:
    """Traverse a dictionary and return all keys including nested ones.

    Args:
        src (Dict): an instance of :class:`Dict`.
        prefix (Optional[str], optional): [description]. Defaults to prefix.
        sep (Optional[str], optional): [description]. Defaults to sep.
        dct (Optional[Dict], optional): [description]. Defaults to {}.

    Returns:
        Dict: flatten dictionary with all keys.
    """
    items = []
    for k, v in src.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _dict_depth(d: Dict | CfgNode) -> int:
    """Calculate the maximal depth of dictionary

    Args:
        d (Dict): an instance of :class:`Dict`.

    Returns:
        int: maximal depth.
    """
    if isinstance(d, dict):
        # 如果d是空dict就直接给0
        return 1 + (max(map(_dict_depth, d.values())) if d else 0)
        # return 1 + (max(map(_dict_depth, d.values())) if d else 0)
    else:
        return 0
        # 无限递归最后肯定不是dict，也就是说肯定会raise error，这是不合理的
        # TypeError("Expected type is dict but {} is received".format(
        #     type(d).__name__))


# NOTE: given the new config system
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# they will stop adding new functionalities to default CfgNode.


# NOTE: maybe someday one require save config orderly, I have tried and find it not easy.
# there is a method making yaml.load() output ordered dict: https://tendcode.com/article/yaml_order/ ,
# but yacs.config.CfgNode is a subclass of :class:`Dict`, so it may hard to make a dict
# subclass has ordered key when initialize.
class CfgNode(_CfgNode):
    # counter records user visits of every attributes and is used in self.unvisited_keys()
    COUNTER = "__COUNTER__"
    CACHED_NAMES = "__CACHED_NAMES__"

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        """
        Args:
            init_dict (dict): the possibly-nested dictionary to initailize the CfgNode.
            key_list (list[str]): a list of names which index this CfgNode from the root.
                Currently only used for logging purposes.
            new_allowed (bool): whether adding new key is allowed when merging with
                other configs.
        """
        super(CfgNode, self).__init__(init_dict)
        # when self.load_cfg_from_file(), it consequently goes to cls(cfg_as_dict), where `init_dict` is not None
        # counter dict only contain flattened leaf node of a CfgNode rather than direct child node,
        # for example, counter dict of node
        # TOPKEYA:
        #     KEYA: "value1"
        #     KEYB:
        #         SUBKEYA: 1000
        #         SUBKEYB: 2000
        # has key ['TOPKEYA.KEYA', 'TOPKEYA.KEYB.SUBKEYA', 'TOPKEYA.KEYB.SUBKEYB'], but has no 'TOPKEYA' or 'TOPKEA.KEYB',
        # and the counter dict of node TOPKEYA has key ['KEYA', 'KEYB.SUBKEYA', 'KEYB.SUBKEYB'], but has no 'KEYB'.
        if init_dict is not None:
            self.__dict__[CfgNode.COUNTER] = _flatten_dict(init_dict)
            for key in self.__dict__[CfgNode.COUNTER].keys():
                self.__dict__[CfgNode.COUNTER][key] = 0
        else:
            self.__dict__[CfgNode.COUNTER] = {}
        self.__dict__[CfgNode.CACHED_NAMES] = []

        self.set_new_allowed(new_allowed)

    def __getattr__(self, name):
        if name in self:
            self.__dict__[CfgNode.CACHED_NAMES].append(name)
            concated_name = sep.join(self.__dict__[CfgNode.CACHED_NAMES])
            if concated_name in self.__dict__[CfgNode.COUNTER]:
                # only parent node of leaf CfgNode can reach here, and top level node can't
                self.__dict__[CfgNode.COUNTER][concated_name] += 1
                self.__dict__[CfgNode.CACHED_NAMES] = []
            return self[name]
        else:
            raise AttributeError(name)

    # TODO: overload __setattr__ to use `new_allowed` to avoid user manually add key by `cfg["key"]=value`.
    # Or is it necessary to do that? Because neither yacs and detectron2 make this feature.

    # TODO: When adding a new key, COUNTER does not contain an entry for the newly added key

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r", encoding="utf-8")

    @classmethod
    def load_cfg_from_file(
        cls,
        filename_or_str_content: Union[str, Path],
        new_allowed: bool = True,
        root_path: str = None,
    ) -> CfgNode:
        """load configration from a yaml file.
        Modified from function load_yaml_with_base() of fvcore.common.config.CfgNode.
        The original one do not support `NEW_ALLOWED` key, but I think sometime it will
        be needed, so we had better add it.

        Args:
            filename_or_str_content (Union[str, Path]): a yaml filename or yaml content string
            new_allowed (bool): whether adding new key is allowed when merging with
                other configs.
            root_path (str): Parent directory of `_BASE_` config. Usually _BASE_ is written
                as a relative path, the result will change if the path executing command change,
                and we directly use `root_path` as the actual parent directory of `_BASE_` config file
                to avoid this confusion.

        Returns:
            cfg: a :class:`CfgNode` instance.
        """
        is_file = PathManager.isfile(filename_or_str_content)
        if len(str(filename_or_str_content)) < 256 and str(
            filename_or_str_content
        ).endswith(".yaml"):
            # We assume if input is a yaml file path, it will not longer than 256
            # and it should ends with '.yaml'
            if is_file:
                with cls._open_cfg(filename_or_str_content) as file:
                    # load_cfg use yaml.safe_load() to prevent malicious code (see https://zhuanlan.zhihu.com/p/54332357);
                    # fvcore supports yaml.unsafe_load(), but I don't see any code use it both in detectron2 and fvcore,
                    # so I think use original load_cfg() in yacs is enough.
                    cfg = cls.load_cfg(file)
            else:
                msg = (
                    f"CfgNode: Input string: '{filename_or_str_content}' looks like"
                    " a yaml file path, but the file is not found on disk!"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
        else:
            # Otherwise the input is a yaml-format string
            cfg = cls.load_cfg(filename_or_str_content)

        if root_path is not None and hasattr(cfg, "_BASE_"):
            path = Path(root_path) / cfg._BASE_
            if not path.exists():
                raise ValueError("Path {} does not exist.".format(path))
            cfg._BASE_ = str(path)

        def _load_with_base(base_cfg_file: str) -> CfgNode:
            if base_cfg_file.startswith("~"):
                base_cfg_file = Path(base_cfg_file).expanduser()
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                if is_file:
                    # the path to base cfg is relative to the config file itself.
                    base_cfg_file = Path(filename_or_str_content).parent / base_cfg_file
            return cls.load_cfg_from_file(base_cfg_file, new_allowed=new_allowed)

        if BASE_KEY in cfg:
            if isinstance(cfg[BASE_KEY], list):
                base_cfg = cls(new_allowed=new_allowed)
                base_cfg_files = cfg[BASE_KEY]
                # NOTE: `new_allowed` of the new added key is default False, so after a "new_allowed" merge new keys from other config,
                # the new key is not `new_allowed`, which is unreasonable, so we manually update `new_allowed` of merged new keys
                for base_cfg_file in base_cfg_files:
                    base_cfg.merge_from_other_cfg(_load_with_base(base_cfg_file))
                    base_cfg.set_new_allowed(new_allowed)
            else:
                base_cfg_file = cfg[BASE_KEY]
                base_cfg = _load_with_base(base_cfg_file)
            del cfg[BASE_KEY]

            base_cfg.merge_from_other_cfg(cfg)
            return base_cfg

        cfg.set_new_allowed(new_allowed)
        return cfg

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CfgNode."""
        _merge_a_into_b(cfg_other, self, self, [])
        other_counter = cfg_other.__dict__[CfgNode.COUNTER]
        self.__dict__[CfgNode.COUNTER] = {
            **self.__dict__[CfgNode.COUNTER],
            **other_counter,
        }

    def dict(self):
        # NOTE: Without deepcopy, if value is a list, cfg.dict() will use a shallow copy of this list,
        # then change this list of cfg.dict() will lead to unexpected changeing of original cfg
        result = {}
        for key, value in deepcopy(self).items():
            if isinstance(value, CfgNode):
                result[key] = value.dict()
            else:
                result[key] = value

        return result

    def diff(self, other: CfgNode):
        """Show the difference between self and other `CfgNode`, helping user
        find Help users quickly identify the difference between them.

        Args:
            other (CfgNode): Another `CfgNode`.

        Returns:
            DeepDiff: A class containing difference, include adding, deleting and modifing.
        """
        from deepdiff import DeepDiff

        return DeepDiff(self, other)

    def dump(self, *args, **kwargs):
        """
        At present dump() can only ensure original CfgNode == the one after dump and reload,
        but can not ensure the order of their keys is consistent.

        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as fp:
            # set sort_key=False to keep writing order the same as original
            # input file rather than ordered by alphabetically;
            # set default_flow_style=None to keep list element written in one line
            # allow_unicode=True to support Chinese input
            self.dump(
                stream=fp, sort_keys=False, default_flow_style=None, allow_unicode=True
            )

    def depth(self):
        return _dict_depth(self)

    def unvisited_keys(self, inverse: Optional[bool] = False) -> List[str]:
        """Return all unvisited keys.

        Args:
            inverse (Optional[bool], optional): return all visited keys if `inverse` is True. Defaults to False.

        Returns:
            List[str]: list of all unvisited/visited keys.
        """
        self.__update_counter(self)
        condition = lambda x: x == 0 if not inverse else x > 0
        return [
            key
            for key, value in self.__dict__[CfgNode.COUNTER].items()
            if condition(value)
        ]

    def __update_counter(self, root: CfgNode, prefix=""):
        """Internal methods to recursively update counter for each keys.

        Args:
            root (CfgNode): Parent node of current CfgNode.
            prefix (str, optional): Concatenation of parent, grandparent and so on.
                For root CfgNode `prefix` is "", for a SUBKEY `prefix` may be "TOPKEYA.KEYB".
        """
        for key, kid_node in self.items():
            new_key = prefix + sep + key if prefix else key
            if isinstance(kid_node, dict) and _dict_depth(kid_node) > 0:
                kid_node.__update_counter(root, new_key)
            else:
                # a new_key of value "TOPKEYA.KEYB.SUBKEYA" lead to a1 slice_key
                # of value "['KEYB.SUBKEYA', 'TOPKEYA.KEYB.SUBKEYA']", which contain all parent keys
                sliced_keys = [
                    ".".join(new_key.split(".")[-k:])
                    for k in range(2, 1 + len(new_key.split(".")))
                ]
                # `self` is the father of `key`, and `root` is the father of `self`
                for root_key in root.__dict__[CfgNode.COUNTER].keys():
                    matched = any(
                        [sliced_key in root_key for sliced_key in sliced_keys]
                    )
                    if matched:
                        root.__dict__[CfgNode.COUNTER][root_key] = self.__dict__[
                            CfgNode.COUNTER
                        ][key]


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type or issubclass(original_type, replacement_type):
        return replacement

    # If either of them is None, allow type conversion to one of the valid types
    if (replacement_type == type(None) and original_type in _VALID_TYPES) or (
        original_type == type(None) and replacement_type in _VALID_TYPES
    ):
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for from_type, to_type in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        f"Key type mismatchs during merging config! Key: {full_key}, original: {original} of type {original_type}, new: {replacement} of type {replacement_type}."
    )


def _merge_a_into_b(a, b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in b:
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # Recursively merge dicts
            if isinstance(v, CfgNode):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])
                except BaseException:
                    raise
            else:
                b[k] = v
        elif b.is_new_allowed() or isinstance(b, MutableCfgNode):
            b[k] = v
        else:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                raise KeyError("Non-existent config key: {}".format(full_key))


class MutableCfgNode(CfgNode):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
        self.set_new_allowed(new_allowed)


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    # inspect.signature() obtains parameter list of function, such as (a, b=0, *c, d, e=1, **f)
    signature = inspect.signature(from_config_func)
    # cfg should be passed as the first parameter, whether it is a positional or keyword argument
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if (
        support_var_arg
    ):  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (_CfgNode, DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite

        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a": cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite

        # Usage 3: Decorator on any method of class. Needs an extra from_config argument:
        class A:
            @configurable(from_config=lambda cfg: {"a": cfg.A, "b": cfg.B})
            def a_func(self, a, b=2, c=3):
                pass
        insA = A()
        cfg = CfgNode.load_cfg('{"A": "2", "B": "3"}')
        a1 = insA.a_func(a=1, b=2)  # regular call
        a2 = insA.a_func(cfg)       # call with a cfg
        a3 = insA.a_func(cfg, b=3, c=4)  # call with extra overwrite

    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2 and 3. It must take `cfg`
            as its first argument.
    """
    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                )

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            params = inspect.signature(orig_func).parameters
            if "self" in params or "cls" in params:  # classmethod or instancemethod

                @functools.wraps(orig_func)
                def wrapped(
                    self, *args, **kwargs
                ):  # here `self` means actual `self` or `cls`
                    if _called_with_cfg(*args, **kwargs):
                        explicit_args = _get_args_from_config(
                            from_config, *args, **kwargs
                        )
                        return orig_func(self, **explicit_args)
                    else:
                        return orig_func(self, *args, **kwargs)

                wrapped.from_config = from_config
                return wrapped

            else:  # function or staticmethod

                @functools.wraps(orig_func)
                def wrapped(*args, **kwargs):
                    if _called_with_cfg(*args, **kwargs):
                        explicit_args = _get_args_from_config(
                            from_config, *args, **kwargs
                        )
                        return orig_func(**explicit_args)
                    else:
                        return orig_func(*args, **kwargs)

                wrapped.from_config = from_config
                return wrapped

        return wrapper
