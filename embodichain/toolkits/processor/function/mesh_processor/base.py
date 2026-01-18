import abc
from typing import List, Dict

from embodichain.utils.cfg import CfgNode

from embodichain.toolkits.processor.entity import MeshEntity


class MeshProcessorMetaClass(type):
    register_class = {}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "MeshProcessor":
            cls.register_class[name.upper()] = new_cls
        return new_cls


class MeshProcessor(metaclass=MeshProcessorMetaClass):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def apply(self, meshes: List[MeshEntity]) -> List[MeshEntity]:
        pass


class MeshProcessorList(List[MeshProcessor]):
    def apply(self, meshes: List[MeshEntity]) -> List[MeshEntity]:
        for processor in self:
            meshes = processor.apply(meshes)
        return meshes


def build_mesh_processors(config: CfgNode) -> MeshProcessorList:
    processors = MeshProcessorList()
    for name, cfg in config.items():
        if cfg is None:
            cfg = {}
        processor = MeshProcessor.register_class[name.upper()](**cfg)
        processors.append(processor)
    return processors
