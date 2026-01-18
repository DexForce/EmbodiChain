# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from typing import Union, Dict, Any

# from dataclasses import asdict, is_dataclass

from embodichain.utils.cfg import CfgNode

from embodichain.toolkits.processor.component import EntityComponent
from embodichain.toolkits.processor.types import CFG_DEF_TYPE_KEYS


class EntityMetaClass(type):
    register_class = {}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "EntityBase":
            cls.register_class[name.upper()] = new_cls
        return new_cls


class EntityBase(metaclass=EntityMetaClass):
    """
    The base class for all entities in the scene.

    Args:
        name (str): The name of the entity.

    Attributes:
        name (str): The name of the entity.
        _components (Dict[type, EntityComponent]): A dictionary of components of type `EntityComponent`.
    """

    def __init__(self, name: str) -> None:
        self.name = name

        self._components: Dict[type, EntityComponent] = {}
        self._custom_properties: Dict[str, Any] = {}

    @classmethod
    def kind(cls) -> str:
        """
        The kind of the entity.

        Returns:
            str: A string representing the name of the class.
        """
        return cls.__name__

    def get_name(self) -> str:
        """
        Get the name of the entity.

        Returns:
            str: the name of the entity
        """
        return self.name

    def set_custom_property(self, **kwargs):
        for key, value in kwargs.items():
            self._custom_properties[key] = value

    def has_custom_property(self, *key: str):
        for k in key:
            if k not in self._custom_properties:
                return False
        return True

    def get_custom_property(self, key: str):
        if key not in self._custom_properties:
            return None
        return self._custom_properties[key]

    def get_custom_properties(self) -> Dict[str, Any]:
        return self._custom_properties

    def remove_custom_property(self, *key: str):
        for k in key:
            if k in self._custom_properties:
                del self._custom_properties[k]

    def add_component(self, *component: EntityComponent):
        """
        Adds one or more components to the entity.

        Args:
            *component (EntityComponent): One or more components to be added.

        Raises:
            AssertionError: If any of the components are not instances of EntityComponent or not direct subclasses of EntityComponent.

        Description:
            This method adds one or more components to the entity. If a component of the same type already exists, it will be replaced.

        Note:
            Currently, the method only accepts components that are instances of EntityComponent and direct subclasses of EntityComponent.

        Example:
            >>> entity = Entity()
            >>> entity.add_component(ScaleComponent(), SpatializationComponenet())
        """
        # The existing component of the same type will be replaced.
        for comp in component:
            assert isinstance(
                comp, EntityComponent
            ), f"Component {type(comp)} must be an instance of EntityComponent"
            assert (
                EntityComponent in comp.__class__.__bases__
            ), f"Component {type(comp)} must be a direct subclass of EntityComponent"
            # we only accept dataclass type component
            # assert is_dataclass(comp)
            self._components[type(comp)] = comp

    def has_component(self, comp_type: type) -> bool:
        """
        Check if the entity has a component of a specific type.

        Args:
            comp_type (type): The type of component to check for.

        Returns:
            bool: True if the entity has a component of the specified type, False otherwise.
        """
        return comp_type in self._components

    def get_component(self, comp_type: type) -> Union[None, EntityComponent]:
        """
        Get the component of the specified type from the entity.

        Args:
            comp_type (type): The type of component to retrieve.

        Returns:
            Union[None, EntityComponent]: The component of the specified type, or None if it does not exist.
        """
        return self._components.get(comp_type, None)

    def remove_component(self, comp_type: type):
        """
        Remove a component of the specified type from the entity.

        Parameters:
            comp_type (type): The type of component to remove.

        Raises:
            ValueError: If the component of the specified type does not exist.

        Returns:
            None
        """
        if comp_type in self._components:
            del self._components[comp_type]
        else:
            raise ValueError(f"{comp_type} component does not exist")

    def save(self) -> Dict:
        """
        Saves the components of the entity to a dictionary.

        Returns:
            Dict: A dictionary containing the names of the component types as keys and the saved components as values.
        """
        results = {}
        for comp_type, comp in self._components.items():
            # # only save the components that are decorated by dataclass
            # if is_dataclass(comp):
            results[comp_type.__name__] = comp.save()
        return results

    @classmethod
    def from_config(cls, cfg: Union[str, CfgNode]) -> "EntityBase":
        if isinstance(cfg, CfgNode):
            if "NAME" not in cfg:
                raise ValueError(f"Config should contains key NAME, but got {cfg}.")
            name = cfg.pop("NAME")
        elif isinstance(cfg, str):
            name = cfg
        else:
            raise TypeError(f"Config should be a string or CfgNode, but got {cfg}.")
        return cls(name)


def build_entity_from_config(
    cfg: Union[str, CfgNode], entity_type: str = None
) -> EntityBase:
    if isinstance(cfg, str) and entity_type is None:
        err_msg = f"'entity_type' is required when 'cfg' is a string."
        raise ValueError(err_msg)
    type_key = None
    if entity_type is None:
        for tk in CFG_DEF_TYPE_KEYS:
            if tk in cfg:
                if type_key is not None:
                    raise ValueError(
                        f"Config should only contains one of keys {CFG_DEF_TYPE_KEYS}, but got {cfg}."
                    )
                type_key = tk
        if type_key is None:
            raise ValueError(
                f"Config should contains one of keys {CFG_DEF_TYPE_KEYS}, but got {cfg}."
            )
        type_key = cfg.pop(type_key)
    else:
        type_key = entity_type
    register_class = EntityBase.register_class
    if isinstance(type_key, str):
        type_key = type_key.upper()
        if type_key not in register_class:
            raise ValueError(f"Class {type_key} is not registered")
    else:
        raise TypeError(f"Class type {type_key} is not a string")
    return register_class[type_key].from_config(cfg)
