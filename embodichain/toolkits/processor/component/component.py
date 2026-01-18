# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import abc
import hashlib
import numpy as np
import open3d as o3d
from typing import List, Dict, Any
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from scipy.spatial.transform import Rotation

from embodichain.utils.cfg import CfgNode
import dexsim.utility as dexutils
from embodichain.toolkits.processor.types import CFG_DEF_TYPE_KEYS
from dexsim.kit.meshproc.generate_thicker_acd import get_pc_thickness


class BaseMetaClass(type):
    register_class = {}

    def __new__(cls, name: str, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "EntityComponent":
            cls.register_class[name.upper()] = new_cls
        return new_cls


class EntityComponent(metaclass=BaseMetaClass):
    """An abstract class for entity components.

    EntityComponent 只是单纯的数据类，没有其他功能行为。 因此我们可以用 `@dataclass` 来装饰他们。

    在使用 `@dataclass` 装饰器时，有几点需要注意：
    1. 使用 eq=False 避免使用 dataclass 默认的 __eq__ 方法，而是使用基类的 __eq__ 方法，
       特别是当数据类中有 np.ndarray 类型的变量 比较时会出现错误
    2. 使用 fronzen=True 避免多个 entity 共享同一个组件实例的时候，
       修改某个组件的属性会影响到其他 entity 的问题。这样做的话，一旦一个组件被初始化，其属性就不能再被修改了，只能通过 new 方法创建新的组件实例。
       当一个 Component 可能会被多个 entity 共享时，就应该使用 frozen=True 来避免这个问题。
    """

    def __eq__(self, other):
        """
        用于检查两个相同 EntityComponent 是否相等。

        重写 dataclass 的 __eq__ 方法，是增加 EntityComponent 中有 np.ndarray 类型变量的比较。

        Args:
            other (object): The object to compare with.
        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if isinstance(other, self.__class__):
            for k, v in self.__dict__.items():
                if k not in other.__dict__:
                    return False
                if isinstance(v, np.ndarray):
                    if not np.array_equal(v, other.__dict__[k]):
                        return False
                elif v != other.__dict__[k]:
                    return False
            return True
        return False

    def new(self, **kwargs):
        """
        根据传入的参数更新组件的属性，并返回一个新的组件实例。

        一个组件可能会被多个 entity 共享，所以在更新组件的属性时，如果需要对不同的 entity 有不同的属性值，
        使用该接口进行参数更新，而不是直接修改组件的属性。

        Args:
            **kwargs: Keyword arguments to set the attributes of the new instance.

        Raises:
            ValueError: If any of the keyword arguments are not valid fields of the class.

        Returns:
            An instance of the class with the given keyword arguments set as attributes.

        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"{k} is not a valid field")
            # TODO check the type of value
            # setattr(self, k, v)
        for field in fields(self):
            if field.name not in kwargs:
                kwargs[field.name] = getattr(self, field.name)
        return type(self)(**kwargs)

    def save(self) -> Dict:
        """
        保存当前组件实例的数据。

        该函数首先检查当前实例是否为 dataclass，如果是，就使用 `asdict` 函数来保存数据。如果不是，就抛出一个 `NotImplementedError` 异常。
        即当前我们仅支持 dataclass 类型的组件实例。

        Returns:
            Dict: The data of the current instance as a dictionary.

        Raises:
            NotImplementedError: If the current instance is not a dataclass.
        """
        if not is_dataclass(self):
            raise NotImplementedError
        return asdict(self)

    @classmethod
    def from_config(cls, cfg: CfgNode):
        if not is_dataclass(cls):
            raise NotImplementedError
        data_fields = fields(cls)
        if not isinstance(cfg, dict):
            if len(data_fields) > 1:
                raise ValueError(f"Config should be a dict, but got {cfg}.")
            return cls(cfg)
        else:
            params = {}
            for key, val in cfg.items():
                params[key.lower()] = val
            return cls(**params)


def build_component_from_config(cfg: CfgNode, name: str = None) -> EntityComponent:
    type_key = None
    if name is None:
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
        type_key = name
    register_class = EntityComponent.register_class
    if isinstance(type_key, str):
        type_key = type_key.upper()
        if type_key not in register_class:
            raise ValueError(f"Class {type_key} is not registered")
    else:
        raise TypeError(f"Class type {type_key} is not a string")
    try:
        return register_class[type_key].from_config(cfg)
    except Exception as e:
        raise ValueError(f"Failed to build component {type_key}, {e}")


@dataclass(eq=False)
class AxisAlignedBoundingBox(EntityComponent):
    min_bound: np.ndarray
    max_bound: np.ndarray

    def is_close(
        self, other: "AxisAlignedBoundingBox", threshold: float = 1e-3
    ) -> bool:
        return np.allclose(
            self.min_bound, other.min_bound, atol=threshold
        ) and np.allclose(self.max_bound, other.max_bound, atol=threshold)


@dataclass(eq=False)
class OrientedBoundingBox(EntityComponent):
    center: np.ndarray
    extent: np.ndarray
    R: Rotation


@dataclass(eq=False, frozen=True)
class TriangleComponent(EntityComponent):
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_uvs: np.ndarray = np.empty((0, 3, 2))
    vertex_uvs: np.ndarray = np.empty((0, 2))
    vertex_colors: np.ndarray = np.empty((0, 3))  # or 4
    vertex_normals: np.ndarray = np.empty((0, 3))
    texture: np.ndarray = np.empty((0, 0, 3))  # hwc
    mesh_fpath: str = None
    optional_params: Dict[str, Any] = field(default_factory=dict)

    def md5_hash(self) -> str:
        md5 = hashlib.md5()
        hash_attr_keys = ["vertices", "triangles", "mesh_fpath"]
        for key in hash_attr_keys:
            val = getattr(self, key)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                md5.update(val.tobytes())
            elif isinstance(val, str):
                md5.update(val.encode())

        return md5.hexdigest()

    @classmethod
    def from_config(cls, cfg: CfgNode):
        if "MESH_FPATH" not in cfg:
            raise ValueError(f"Config should contains key MESH_FPATH, but got {cfg}.")
        mesh_fpath = cfg.MESH_FPATH
        mesh = cls.__from_fpath(mesh_fpath)

        optional_params = {}
        if "OPTIONAL_PARAMS" in cfg and cfg.OPTIONAL_PARAMS is not None:
            for key, val in cfg.OPTIONAL_PARAMS.items():
                optional_params[key.lower()] = val
        return cls(
            vertices=mesh.vertices,
            triangles=mesh.triangles,
            triangle_uvs=mesh.triangle_uvs,
            vertex_uvs=mesh.vertex_uvs,
            vertex_colors=mesh.vertex_colors,
            vertex_normals=mesh.vertex_normals,
            texture=mesh.texture,
            mesh_fpath=mesh_fpath,
            optional_params=optional_params,
        )

    @classmethod
    def from_fpath(cls, mesh_fpath: str):
        mesh = cls.__from_fpath(mesh_fpath)

        return cls(
            vertices=mesh.vertices,
            triangles=mesh.triangles,
            triangle_uvs=mesh.triangle_uvs,
            vertex_uvs=mesh.vertex_uvs,
            vertex_colors=mesh.vertex_colors,
            vertex_normals=mesh.vertex_normals,
            texture=mesh.texture,
            mesh_fpath=mesh_fpath,
        )

    @staticmethod
    def __from_fpath(mesh_fpath: str):
        from dexsim.kit.meshproc.mesh_io import load_mesh

        mesh = load_mesh(mesh_fpath)
        if isinstance(mesh, List):
            msg = f"Mesh file {mesh_fpath} contains multiple meshes. Only the first mesh will be used."
            dexutils.log_warning(msg)
            mesh = mesh[0]
        # vertices = mesh.vertices
        # triangles = mesh.triangles
        if mesh.vertex_uvs is None:
            mesh.vertex_uvs = np.empty((0, 2))
        if mesh.triangle_uvs is None:
            mesh.triangle_uvs = np.empty((0, 3, 2))
        if mesh.vertex_colors is None:
            mesh.vertex_colors = np.empty((0, 3))
        if mesh.vertex_normals is None:
            mesh.vertex_normals = np.empty((0, 3))
        if mesh.texture is None:
            mesh.texture = np.empty((0, 0, 3))
        return mesh

    def get_thickness(self):
        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.triangles),
        )
        surface_pc_o3d = mesh_o3d.sample_points_uniformly(number_of_points=3000)
        surface_pc = np.array(surface_pc_o3d.points)
        thickness, standard_pose = get_pc_thickness(surface_pc)
        return thickness


@dataclass()
class VisualComponent(EntityComponent):
    is_visual: bool = True


@dataclass(eq=False)
class ScaleComponent(EntityComponent):
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])

    def __post_init__(self):
        if self.scale is not None:
            self.scale = np.array(self.scale)


@dataclass(eq=False)
class SpatializationComponenet(EntityComponent):
    location: np.ndarray = np.array([0, 0, 0], dtype=np.float32)
    rotation: Rotation = Rotation.from_matrix(np.eye(3))

    def __post_init__(self):
        if self.location is not None:
            self.location = np.array(self.location, dtype=np.float32)
        if self.rotation is not None and not isinstance(self.rotation, Rotation):
            self.rotation = Rotation.from_euler("xyz", self.rotation, degrees=True)

    def save(self) -> Dict:
        return_dict = asdict(self)
        return_dict["rotation"] = self.rotation.as_euler("xyz", degrees=True)
        return return_dict

    def get_pose(self) -> np.ndarray:
        pose = np.eye(4)
        if self.location is not None:
            pose[:3, 3] = self.location
        if self.rotation is not None:
            pose[:3, :3] = self.rotation.as_matrix()
        return pose
