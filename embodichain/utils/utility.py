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


import cv2
import pickle
import argparse
import time
import torch
import functools
import open3d as o3d
import numpy as np

from tqdm import tqdm
from PIL import Image
from functools import wraps
from typing import Dict, List, Tuple, Optional, Callable, Any

from embodichain.utils.string import callable_to_string


@functools.lru_cache(maxsize=None)  # memoization
def get_func_tag(tagName):
    return TagDecorator(tagName)


# https://stackoverflow.com/questions/41834530/how-to-make-python-decorators-work-like-a-tag-to-make-function-calls-by-tag
class TagDecorator(object):
    def __init__(self, tagName):
        self.functions = {}
        self.tagName = tagName

    def __str__(self):
        return "<TagDecorator {tagName}>".format(tagName=self.tagName)

    def __call__(self, f):
        class_name = f.__qualname__.split(".")[0]
        if class_name in self.functions.keys():
            self.functions[class_name].update({f.__name__: f})
        else:
            self.functions.update({class_name: {f.__name__: f}})
        return f


def set_attributes_for_class(self, params=None):
    if params:
        for k, v in params.items():
            if k != "self" and not k.startswith("_"):
                setattr(self, k, v)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        # log_warning(
        #     f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        # )
        return result  # 返回被装饰函数的执行结果

    return wrapper


from embodichain.utils.logger import log_warning, log_error


def snake_to_camel(name):
    import re

    name = re.sub("_([a-zA-Z])", lambda m: (m.group(1).upper()), name)
    name = re.sub("-+", "_", name)
    return name


def convert_bytes(d):
    if isinstance(d, dict):
        return {convert_bytes(k): convert_bytes(v) for k, v in d.items()}
    if isinstance(d, list):
        return [convert_bytes(i) for i in d]
    if isinstance(d, bytes):
        return d.decode("UTF-8")
    return d


def pad_to_chunk(x: np.ndarray, chunk_size: int) -> np.ndarray:
    if x.shape[0] < chunk_size:

        if len(x.shape) <= 2:
            x = np.concatenate(
                [
                    x,
                    np.tile(
                        x[-1:],
                        (chunk_size - x.shape[0], 1),
                    ),
                ],
                axis=0,
            )
        elif len(x.shape) == 3 or len(x.shape) == 4:
            x = np.concatenate(
                [
                    x,
                    np.tile(
                        x[-1:],
                        (
                            (chunk_size - x.shape[0], 1, 1, 1)
                            if len(x[:1].shape) == 4
                            else (chunk_size - x.shape[0], 1, 1)
                        ),
                    ),
                ],
                axis=0,
            )
        else:
            raise ValueError("Unsupported shape {}.".format(x.shape))

    assert x.shape[0] == chunk_size, "shape {} vs chunk_size {}.".format(
        x.shape, chunk_size
    )
    return x


def dict2args(d: Dict) -> argparse.ArgumentParser:
    args = argparse.Namespace(**d)
    return args


def parser2dict(args) -> Dict:
    return vars(args)


def change_nested_dict(dict, keys, mode: str = "update", value=None):
    """
    Update or delete a nested dictionary at a specific key.

    Args:
        dict (dict): The dictionary to update.
        keys (tuple): Tuple of keys to the target value.
        mode (str): Whether to delete or remove the given key-value pair.
        value: The new value to set.

    Returns:
        dict: The updated dictionary.
    """
    if mode == "update":
        if value is None:
            log_error("The value to be updated is None, please check.")
        else:
            if len(keys) == 1:
                dict[keys[0]] = value
            else:
                change_nested_dict(dict[keys[0]], keys[1:], "update", value)
    elif mode == "delete":
        if value is not None:
            log_warning(
                f"Under mode 'delete' only the keys to be removed need to be provided. But got a not-None vlaue {value}."
            )
        if len(keys) == 1:
            del dict[keys[0]]
        else:
            change_nested_dict(dict[keys[0]], keys[1:], "delete")
    else:
        log_error(f"Mode '{mode}; is noet realized yet.")

    return dict


def set_texture_to_material(material, texture: np.ndarray, env, type: str = "color"):
    if type == "color":
        # TODO: Currently, create texture for base color map without alpha has error.
        # should be fixed in the future.
        if texture.shape[-1] == 3:
            texture = np.concatenate(
                [texture, np.ones_like(texture[..., :1]) * 255], axis=-1
            )

        color_texture = env.create_color_texture(texture, has_alpha=True)
        if color_texture:
            material.get_inst().set_base_color_map(color_texture)
    else:
        log_error(f"Unsupported texture type: {type}. Only 'color' is supported.")


def get_random_real_image(base_path: str, read: bool = True) -> np.ndarray:
    import os, random

    # 随机选择一个子文件夹
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    selected_subfolder = random.choice(subfolders)

    # 随机选择一个图片文件
    image_files = [
        f.path
        for f in os.scandir(selected_subfolder)
        if f.is_file() and f.path.endswith((".png", ".jpg", ".jpeg"))
    ]
    selected_image_file = random.choice(image_files)

    # 读取图片
    if read:
        real_image = cv2.imread(selected_image_file)
        return real_image
    else:
        return selected_image_file


def read_all_folder_images(base_path: str) -> List[np.ndarray]:
    """Read all images from all subfolders under the base path.

    Args:
        base_path (str): The base directory containing subfolders with images.

    Returns:
        List[np.ndarray]: A list of images read from the subfolders.
    """
    import os

    images = []
    # 遍历所有子文件夹
    # First, collect all image files
    image_files = []
    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(subdir, file))

    # Then process with progress bar
    for image_path in tqdm(image_files, desc="Loading images"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            images.append(image)
    return images


def reset_all_seeds(seed: int = 0):
    import torch
    import random
    import open3d as o3d

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    o3d.utility.random.seed(seed)


def do_process_decorator(
    pre_process: Optional[bool] = True, post_process: Optional[bool] = True
):
    """A decorator to decorate :meth:`inference`. Usage and example is comming soon.

    Args:
        pre_process (Optional[bool], optional): whether do pre-process. Defaults to True.
        post_process (Optional[bool], optional): whether do post-process. Defaults to True.
    """

    def inner_decorator(func: Callable):
        def main_wrapper(self, *args, **kwargs):
            if pre_process:
                input = getattr(self, "pre_process")(*args, **kwargs)
            if isinstance(input, dict):
                ret = func(self, input)
            else:
                ret = func(self, *input)
            if post_process:
                output = getattr(self, "post_process")(*ret)
            return output

        return main_wrapper

    return inner_decorator


def pad_img_list(img_list, max_len):
    while len(img_list) < max_len:
        img_list.append(None)


def get_right_name(name: str):
    return name + "_r"


def read_video(video_path: str):
    video = cv2.VideoCapture(video_path)
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    length = total_frame_count
    fps = video.get(cv2.CAP_PROP_FPS)
    return video, fps, length


def create_video_writer(
    video_path: str, resolution: Tuple[int, int], fps: int
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 用于mp4格式的生成
    video_vis = cv2.VideoWriter(
        video_path,
        fourcc,
        fps,
        (resolution[1], resolution[0]),
    )
    return video_vis


def update_array(
    mat: np.ndarray, vec: np.ndarray, first_is_latest: bool = True
) -> np.ndarray:
    if first_is_latest:
        mat[1:, :] = mat[:-1, :]
        mat[0, :] = vec
        return mat
    else:
        mat[:-1, :] = mat[1:, :]
        mat[-1, :] = vec
        return mat


def save_pkl(path: str, content):
    with open(path, "wb") as f:  # open a text file
        pickle.dump(content, f)  # serialize the list


def load_pkl(
    path: str,
):
    with open(path, "rb") as f:
        content = pickle.load(f)
    return content


def save_json(path: str, data):
    import json

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> Dict:
    import json

    with open(path) as f:
        config = json.load(f)
    return config


def load_txt(path: str) -> str:
    with open(path, "r") as f:
        contents = f.read().strip()
    return contents


def encode_image(image: np.ndarray, format: str = "png"):
    import base64

    image_encode = cv2.imencode(f".{format}", image)[1]
    base64_image = base64.b64encode(image_encode).decode("utf-8")
    return base64_image


def inv_transform(transform: np.ndarray) -> np.ndarray:
    """inverse transformation

    Args:
        transform (np.array): [np.array of size [4 x 4]]

    Returns:
        np.array: [np.array of size [4 x 4]]
    """
    r = transform[:3, :3]
    t = transform[:3, 3].T
    inv_r = r.T
    inv_t = -inv_r @ t
    inv_pose = np.eye(4, dtype=np.float32)
    inv_pose[:3, :3] = inv_r
    inv_pose[:3, 3] = inv_t
    return inv_pose


def scale_image(image, scale=0.5):
    import cv2

    h, w = image.shape[:2]
    if image.dtype == np.uint8:
        return cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale),
            ),
        )
    elif image.dtype == np.bool_:

        image = image.astype(np.uint8)
        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale),
            ),
        )
        return image.astype(np.bool_)


def padding_by_longest_edge(img: np.ndarray) -> np.ndarray:
    w, h, c = img.shape[:3]
    e = np.maximum(w, h)
    ret = np.zeros((e, e, c)).astype(img.dtype)
    ret[:w, :h] = img
    return ret


def center_crop(img: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
    return crop_img


def postprocess_small_regions(
    masks: np.ndarray,
    min_area: int,
    max_area: int,
) -> List[int]:
    """Filter masks based on area constraints.

    Args:
        masks: Array of binary masks or list of masks.
        min_area: Minimum area threshold (exclusive - areas must be strictly greater).
        max_area: Maximum area threshold (inclusive - areas can equal this value).

    Returns:
        List of indices for masks that meet the area constraints (min_area < area <= max_area).
    """
    n = len(masks) if isinstance(masks, list) else masks.shape[0]
    # Use list comprehension for more efficient filtering
    # Logic: area > min_area and area <= max_area (original behavior preserved)
    return [
        i for i in range(n) if min_area < masks[i].astype(np.uint8).sum() <= max_area
    ]


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    from torchvision.ops import masks_to_boxes
    import torch

    bbox = (
        masks_to_boxes(torch.from_numpy(mask).unsqueeze(0))
        .squeeze(0)
        .numpy()
        .astype(np.int16)
    )
    return bbox


def remove_overlap_mask(
    masks: List[np.ndarray], keep_inner_threshold: float = 0.5, eps: float = 1e-5
) -> List[int]:
    keep_ids = []

    # Pre-compute areas once for efficiency
    areas = np.array([mask.astype(np.uint8).sum() for mask in masks])

    for i, maskA in enumerate(masks):
        keep = True
        for j, maskB in enumerate(masks):
            if i == j:
                # 同一个mask，跳过
                continue
            if areas[i] > areas[j]:
                # 大的包裹mask不能被过滤
                continue

            # 计算交集
            intersection = (maskA * maskB).sum()
            # 计算maskA的覆盖比例
            overlap_ratio = intersection / (areas[i] + eps)
            # maskA被maskB覆盖的面积比例达到threshold，不保留
            if overlap_ratio >= keep_inner_threshold:
                keep = False
                break

        if keep:
            keep_ids.append(i)

    return keep_ids


def encode_image_from_path(image_path: str):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def check_shared_memory_exists(name):
    from multiprocessing import shared_memory

    try:
        shm = shared_memory.SharedMemory(name=name)
        return True
    except FileNotFoundError:
        return False


def get_class_instance(module_name, class_name, *args, **kwargs):
    """Get an instance of a class from a module.

    Args:
        module_name (str): The name of the module to import.
        class_name (str): The name of the class to instantiate.

    Returns:
        object: An instance of the specified class.
    """
    import importlib

    # Import the module
    module = importlib.import_module(module_name)
    # Get the class from the module
    cls = getattr(module, class_name)
    return cls


def key_in_nested_dict(d: Dict, key: str) -> bool:
    """Check if a key exists in a nested dictionary.

    Args:
        d (Dict): A dictionary that may contain nested dictionaries.
        key (str): The key to search for in the dictionary.

    Returns:
        bool: True if the key exists in the dictionary or any of its nested dictionaries, False otherwise.
    """
    if key in d:
        return True
    for value in d.values():
        if isinstance(value, dict):  # Check if the value is a nested dictionary
            if key_in_nested_dict(
                value, key
            ):  # Recursively check the nested dictionary
                return True
    return False


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


def get_mesh_md5(mesh: o3d.t.geometry.TriangleMesh) -> str:
    """get mesh md5 unique key

    Args:
        mesh (o3d.geometry.TriangleMesh): mesh

    Returns:
        str: mesh md5 value.
    """
    import hashlib

    vert = np.array(mesh.vertex.positions.numpy(), dtype=float)
    face = np.array(mesh.triangle.indices.numpy(), dtype=float)
    mix = np.vstack([vert, face])
    return hashlib.md5(np.array2string(mix).encode()).hexdigest()
