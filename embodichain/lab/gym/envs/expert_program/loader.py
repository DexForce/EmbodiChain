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

"""Safe file and strict JSON loading for declarative Expert Programs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import yaml

from .cfg import ExpertProgramCfg
from .decoder import (
    ExpertProgramDecodeError,
    ExpertProgramValidationContext,
    decode_expert_program,
)

__all__ = [
    "MAX_EXPERT_PROGRAM_BYTES",
    "load_expert_program",
    "loads_expert_program_json",
    "parse_expert_program_json",
]

MAX_EXPERT_PROGRAM_BYTES = 4 * 1024 * 1024
"""Maximum serialized Expert Program size accepted by the file loader."""


class _StrictJsonValueError(ValueError):
    """Carry one stable strict-JSON failure into the public decode boundary."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build a JSON mapping while rejecting ambiguous duplicate keys."""
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise _StrictJsonValueError(
                "duplicate_json_key",
                f"Duplicate JSON key {key!r}.",
            )
        mapping[key] = value
    return mapping


def _reject_non_finite_json_constant(token: str) -> object:
    """Reject the non-standard NaN and Infinity JSON constants."""
    raise _StrictJsonValueError(
        "non_finite_number",
        f"Non-finite JSON number {token!r} is forbidden.",
    )


def _parse_finite_json_float(token: str) -> float:
    """Parse one JSON float while rejecting overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise _StrictJsonValueError(
            "non_finite_number",
            f"JSON number {token!r} is not finite.",
        )
    return value


def _validate_decoded_json_unicode(value: object) -> None:
    """Reject decoded JSON strings that cannot be represented as UTF-8."""
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise _StrictJsonValueError(
                "invalid_utf8",
                "Expert Program JSON contains an unpaired Unicode surrogate.",
            ) from error
        return
    if type(value) is list:
        for item in value:
            _validate_decoded_json_unicode(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_decoded_json_unicode(key)
            _validate_decoded_json_unicode(item)


def _loads_strict_json_value(
    text: str,
    *,
    max_bytes: int = MAX_EXPERT_PROGRAM_BYTES,
) -> object:
    """Parse one bounded JSON document into exact JSON-compatible values."""

    if type(text) is not str:
        raise TypeError("text must be exactly str.")
    if type(max_bytes) is not int:
        raise TypeError("max_bytes must be exactly int.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ExpertProgramDecodeError(
            "invalid_utf8",
            (),
            "Expert Program JSON must be valid UTF-8 text.",
        ) from error
    if len(payload) > max_bytes:
        raise ExpertProgramDecodeError(
            "input_too_large",
            (),
            f"Expert Program JSON exceeds the {max_bytes}-byte input limit.",
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
            parse_float=_parse_finite_json_float,
        )
        _validate_decoded_json_unicode(value)
        return value
    except _StrictJsonValueError as error:
        raise ExpertProgramDecodeError(error.code, (), error.message) from error
    except json.JSONDecodeError as error:
        raise ExpertProgramDecodeError(
            "invalid_json",
            (),
            "Invalid Expert Program JSON at "
            f"line {error.lineno}, column {error.colno}.",
        ) from error
    except RecursionError as error:
        raise ExpertProgramDecodeError(
            "input_too_deep",
            (),
            "Expert Program JSON exceeds the parser nesting limit.",
        ) from error
    except ValueError as error:
        raise ExpertProgramDecodeError(
            "invalid_json",
            (),
            "Expert Program JSON contains an invalid numeric value.",
        ) from error


def parse_expert_program_json(
    text: str,
    *,
    max_bytes: int = MAX_EXPERT_PROGRAM_BYTES,
) -> dict[str, object]:
    """Parse one bounded Expert Program JSON object without decoding its schema.

    This parse-only boundary lets a host-controlled frontend inspect or inject
    fields before calling :func:`decode_expert_program`. It rejects duplicate
    keys, non-finite numbers, trailing content, invalid Unicode, excessive
    nesting, oversized UTF-8 input, and non-object top-level values. It does
    not validate the Expert Program schema.

    Args:
        text: Untrusted JSON document text.
        max_bytes: Maximum accepted UTF-8 encoded input size.

    Returns:
        Exact JSON object mapping ready for explicit schema decoding.

    Raises:
        TypeError: If ``text`` or ``max_bytes`` has the wrong exact type.
        ValueError: If ``max_bytes`` is not positive.
        ExpertProgramDecodeError: If strict JSON parsing fails.
    """
    value = _loads_strict_json_value(text, max_bytes=max_bytes)
    if type(value) is not dict:
        raise ExpertProgramDecodeError(
            "expected_mapping",
            (),
            "Expected an object mapping.",
        )
    return value


def loads_expert_program_json(
    text: str,
    *,
    validation_context: ExpertProgramValidationContext | None = None,
    max_bytes: int = MAX_EXPERT_PROGRAM_BYTES,
) -> ExpertProgramCfg:
    """Strictly parse and decode one untrusted Expert Program JSON document.

    The input must be one plain JSON document. Markdown fences, trailing text,
    multiple documents, duplicate keys, non-finite numbers, and oversized input
    are rejected before the existing Expert Program decoder is called.

    Args:
        text: Untrusted JSON response text.
        validation_context: Optional provider-free static reference validator.
        max_bytes: Maximum UTF-8 encoded response size.

    Returns:
        Fully owned and internally validated Expert Program configuration.

    Raises:
        TypeError: If ``text`` or ``max_bytes`` has the wrong exact type.
        ValueError: If ``max_bytes`` is not positive.
        ExpertProgramDecodeError: If parsing or strict decoding fails.
    """
    data = parse_expert_program_json(text, max_bytes=max_bytes)
    return decode_expert_program(data, validation_context=validation_context)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML safe loader that also rejects ambiguous duplicate keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct one YAML mapping with unique, hashable keys."""
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def load_expert_program(
    path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    validation_context: ExpertProgramValidationContext | None = None,
) -> ExpertProgramCfg:
    """Safely load and strictly decode one JSON or YAML Expert Program file.

    Relative paths are resolved from ``base_dir`` when provided. Otherwise,
    they retain normal :class:`pathlib.Path` semantics and therefore resolve
    from the process working directory when opened.

    Args:
        path: JSON, YAML, or YML file to load.
        base_dir: Optional directory used to resolve a relative ``path``.
        validation_context: Optional provider-free static reference validator
            applied after decoding either serialized format.

    Returns:
        An owned, validated Expert Program configuration.

    Raises:
        FileNotFoundError: If the resolved path is not a regular file.
        ValueError: If the file is too large, has an unsupported extension, or
            contains ambiguous or invalid serialized data.
        ExpertProgramValidationError: If ``validation_context`` rejects an
            external reference.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    program_path = Path(path).expanduser()
    if base_dir is not None and not program_path.is_absolute():
        program_path = Path(base_dir).expanduser() / program_path
    if not program_path.is_file():
        raise FileNotFoundError(f"Expert Program path is not a file: {program_path}.")
    suffix = program_path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError(
            "Expert Program must use a .json, .yaml, or .yml extension; "
            f"got {program_path.name!r}."
        )

    payload = program_path.read_bytes()
    if len(payload) > MAX_EXPERT_PROGRAM_BYTES:
        raise ExpertProgramDecodeError(
            "input_too_large",
            (),
            "Expert Program exceeds the "
            f"{MAX_EXPERT_PROGRAM_BYTES}-byte input limit.",
        )
    text = payload.decode("utf-8")
    if suffix == ".json":
        return loads_expert_program_json(
            text,
            validation_context=validation_context,
        )
    try:
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid Expert Program YAML in {program_path}: {error}"
        ) from error
    return decode_expert_program(
        data,
        validation_context=validation_context,
    )
