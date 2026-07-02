from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SceneState:
    raw_input_json: Dict = field(default_factory=dict)
    table_size: Optional[Tuple[float, float]] = None
    table_semantics: str = ""
    coordinate_system: Dict = field(default_factory=dict)
    raw_object_dict: Dict[str, Dict] = field(default_factory=dict)
    init_layout: Dict[str, Dict] = field(default_factory=dict)
    optimization_model: Dict[str, Any] = field(default_factory=dict)
    optimized_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stack_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    asset_specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)


@dataclass
class Tempo_SceneState:
    raw_input_json: Dict = field(default_factory=dict)
    table_size: Optional[Tuple[float, float]] = None
    table_semantics: str = ""
    coordinate_system: Dict = field(default_factory=dict)
    raw_object_dict: Dict[str, Dict] = field(default_factory=dict)
    filtered_objects_info: Dict[str, Dict] = field(default_factory=dict)
    filtered_objects: Dict[str, Dict] = field(default_factory=dict)
    init_layout: Dict[str, Dict] = field(default_factory=dict)
    optimization_model: Dict[str, Any] = field(default_factory=dict)
    optimized_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stack_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_layout: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
