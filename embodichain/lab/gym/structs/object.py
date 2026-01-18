import numpy as np
import open3d as o3d
import os
import hashlib

from typing import List, Dict, Union

from embodichain.data import get_data_path
from embodichain.utils import logger

from embodichain.toolkits.processor.function.mesh_processor.base import (
    MeshProcessorList,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGenerator,
)


def generate_pickpose_sampler(
    file_name: str, mesh: o3d.t.geometry.TriangleMesh, params: dict
) -> None:
    logger.log_info(f"Generating object mesh {file_name} pick poses.")
    if os.path.exists(file_name):
        try:
            if os.path.exists(file_name) and not file_name.endswith(".dae"):
                pickpose_sampler = AntipodalGenerator(
                    mesh,
                    **params,
                    unique_id=hashlib.md5(file_name.encode()).hexdigest(),
                )
            elif file_name.endswith(".dae"):
                pickpose_sampler = None
            else:
                logger.log_warning(
                    f"Failed to build AntipodalGenerator because {file_name} is invalid!"
                )
        except Exception as e:
            logger.log_warning(f"Failed to build AntipodalGenerator: {str(e)}")
    else:
        logger.log_warning(
            f"Failed to build AntipodalGenerator cause {file_name} unvalid!"
        )
    return pickpose_sampler


# TODO: We should refactor the Object to support Group object design.
class Object:
    name: str = "Undefined"
    description: str = "Undefined"
    pick_poses: List[np.ndarray] = None
    pose: np.ndarray
    parts: List["Object"]
    articulations: Dict[str, np.ndarray]
    mesh: o3d.geometry.TriangleMesh
    scale: Union[List, np.ndarray] = [1, 1, 1]
    mesh_file: str  # to be depracted.
    active_state: bool = False
    unit: str = "m"
    pickpose_sampler: AntipodalGenerator = None
    pickpose_sampler_params: dict = None
    # for object group
    folder_path: str
    mesh_processor: MeshProcessorList = None

    def get_mesh_file(self):
        if hasattr(self, "mesh_file"):
            return self.mesh_file
        else:
            obj_cad_files = [
                file
                for file in os.listdir(self.folder_path)
                if file.startswith("mesh_processed_") is False
            ]

            target_file = np.random.choice(obj_cad_files)

            return self.select_mesh_file_from_folder(target_file)

    def select_mesh_file_from_folder(self, target_file: str):
        from embodichain.toolkits.processor.component import TriangleComponent
        from embodichain.toolkits.processor.entity import MeshEntity

        cache_fpath = os.path.join(self.folder_path, f"mesh_processed_{target_file}")
        if os.path.exists(cache_fpath) is False:
            tri_component = TriangleComponent.from_fpath(
                os.path.join(self.folder_path, target_file)
            )
            mesh_entity = MeshEntity("mesh", tri_component)
            mesh = self.mesh_processor.apply([mesh_entity])[0]
            mesh.save_mesh(cache_fpath)

        if self.pickpose_sampler_params is not None:
            mesh = o3d.t.io.read_triangle_mesh(cache_fpath)
            self.pickpose_sampler = generate_pickpose_sampler(
                cache_fpath, mesh, self.pickpose_sampler_params
            )

        return cache_fpath

    @staticmethod
    def from_folder(path: str, obj_data: dict) -> "Object":
        from embodichain.toolkits.processor.function.mesh_processor import (
            build_mesh_processors,
        )

        obj = Object()
        obj.folder_path = path
        obj.description = obj_data.get("description", "Undefined")
        obj.name = obj_data.get("name", "Undefined")
        obj.unit = obj_data.get("unit", "m")
        obj.pickpose_sampler_params = obj_data.get("auto_pickpose_generator", None)
        obj.pose = obj_data.get("pose", np.eye(4))
        mesh_processor_config = obj_data.get("mesh_processor", None)
        if mesh_processor_config is not None:
            obj.mesh_processor = build_mesh_processors(mesh_processor_config)

        return obj

    @staticmethod
    def from_mesh(path: str, downsample: bool = False, local_dir: str = "") -> "Object":
        r"""Create an Object instance from a mesh file.

        Args:
            path (str): The file path to the mesh key, value in obj_related[k].items():
                setattr(obj, key, value)
            objs.append(obj)

        # Order the objects based file.
            downsample (bool, optional): Whether to downsample the mesh. Defaults to False.

        Returns:
            Object: An `Object` instance containing the mesh data.
        """
        obj = Object()

        if not os.path.exists(path):
            if local_dir is not None and local_dir != "":
                path = os.path.join(local_dir, path)
            else:
                path = get_data_path(path)

        mesh = o3d.io.read_triangle_mesh(path)
        if downsample:
            obj.mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=4000)
        else:
            obj.mesh = mesh
        obj.mesh_file = path
        return obj

    @staticmethod
    def from_urdf(path: str) -> List["Object"]:
        r"""Create a list of Object instances from a URDF file.

        This method reads a URDF (Unified Robot Description Format) file, extracts
        the geometry data, and returns a list of `Object` instances representing
        the visual elements described in the URDF.

        Args:
            path (str): The file path to the URDF file.

        Returns:
            List[Object]: A list of `Object` instances representing the visual elements.
        """
        import pinocchio
        import copy

        data_path = copy.deepcopy(path)
        if not os.path.exists(data_path):
            data_path = get_data_path(path)
        package_dirs = [os.path.dirname(data_path)]

        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
            data_path, package_dirs=package_dirs
        )

        urdf_dir = os.path.dirname(data_path)
        objs = []

        # Parse the geometry data from URDF
        for geom in visual_model.geometryObjects.tolist():
            if hasattr(geom, "meshPath"):
                mesh_path = geom.meshPath
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(urdf_dir, mesh_path)
                obj = Object.from_mesh(mesh_path)
                obj.name = geom.name
                objs.append(obj)

        return objs

    @staticmethod
    def _save_mesh_or_urdf(obj: "Object", new_file_name: str):
        r"""Save the mesh or URDF file with error handling.

        Args:
            obj (Object): The object containing the mesh or URDF data.
            new_file_name (str): The new file path where the data will be saved.
        """
        try:
            if new_file_name.endswith(".urdf"):
                obj.save_as_urdf(new_file_name)
            else:
                o3d.io.write_triangle_mesh(new_file_name, obj.mesh)
        except Exception as e:
            logger.log_error(f"Failed to save the file {new_file_name}: {str(e)}")

    @staticmethod
    def _generate_new_filename(file_path: str, extension: str) -> str:
        r"""Generate a new filename with a specified extension.

        Args:
            file_path (str): The original file path.
            extension (str): The new extension to append to the filename.

        Returns:
            str: The generated file path with the new extension.
        """
        _, file_extension = os.path.splitext(file_path)
        return os.path.join(
            os.path.dirname(file_path),
            os.path.basename(file_path).split(".")[0] + f"_{extension}{file_extension}",
        )

    @staticmethod
    def _apply_common_settings(obj: "Object", obj_data: dict, file_path: str):
        r"""Apply common settings such as unit conversion and pose generation to the object.

        Args:
            obj (Object): The object to which settings will be applied.
            obj_data (dict): Dictionary containing object-specific configuration data.
            file_path (str): Object file path.
        """
        if "unit" in obj_data and obj_data["unit"] == "mm":
            obj.mesh.scale(1e-3, center=np.zeros((3)))
            obj.unit = "m"
            obj_data.pop("unit")
            new_file_name = Object._generate_new_filename(obj.mesh_file, extension="m")
            Object._save_mesh_or_urdf(obj, new_file_name)
            obj.mesh_file = new_file_name

        obj.scale = obj_data.get("scale", [1, 1, 1])
        from dexsim.utility.meshproc import scale_trianglemesh

        obj.mesh = scale_trianglemesh(obj.mesh, obj.scale)

        if "auto_pickpose_generator" in obj_data:
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
            obj.pickpose_sampler = generate_pickpose_sampler(
                obj.mesh_file, mesh, obj_data["auto_pickpose_generator"]
            )

        for key, value in obj_data.items():
            setattr(obj, key, value)

    @staticmethod
    def from_config(
        path: Union[str, Dict], downsample: bool = False, local_dir: str = ""
    ) -> List["Object"]:
        r"""Create a list of Object instances from a configuration file.

        Args:
            path (Union[str, Dict]): The file path to the configuration file or a dictionary containing the configuration.
            downsample (bool, optional): Whether to downsample the mesh. Defaults to False.

        Returns:
            List[Object]: A list of `Object` instances as specified in the configuration file.
        """
        from embodichain.utils.utility import load_json

        if isinstance(path, str):
            config = load_json(path)
        else:
            config = path

        obj_related = config["obj_list"]
        objs = []

        for k, obj_data in obj_related.items():
            if obj_data.get("mesh_file", None) is not None:
                file = obj_data.pop("mesh_file")
                file_ext = os.path.splitext(file)[-1].lower()

                if local_dir is not None and local_dir != "":
                    data_file_path = os.path.join(local_dir, file)
                else:
                    data_file_path = get_data_path(file)

                if file_ext == ".urdf":
                    urdf_objs = Object.from_urdf(file)
                    for obj in urdf_objs:
                        Object._apply_common_settings(obj, obj_data, data_file_path)
                    objs.extend(urdf_objs)
                else:
                    obj = Object.from_mesh(file, downsample, local_dir=local_dir)
                    Object._apply_common_settings(obj, obj_data, data_file_path)
                    objs.append(obj)
            else:
                folder_path = obj_data.get("folder_path", None)
                if folder_path is None:
                    logger.log_error(
                        f"Object configuration {k} does not contain a valid mesh file or folder path."
                    )
                obj = Object.from_folder(folder_path, obj_data)
                objs.append(obj)

        # TODO: to be improved.
        if len(objs) == len(obj_related):
            order = [int(k) for k in obj_related.keys()]
            return [objs[i] for i in np.argsort(order)]
        else:
            return objs
