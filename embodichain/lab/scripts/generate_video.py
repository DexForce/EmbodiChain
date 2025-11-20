# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from embodichain.utils.logger import log_info, log_warning

import h5py
import argparse
import numpy as np
import os

from tqdm import tqdm
from dexsim.utility import images_to_video
from typing import Dict, Callable, Tuple
from embodichain.utils.visualizer import draw_keypoints, draw_action_distribution
from embodichain.data.enum import EefType, JointType, Modality, PrivilegeType
from embodichain.data.data_engine.unified_state import ActionIndicesGenerator


class VideoCreator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _sub_function(
        images,
        output_path,
        video_key,
        exteroceptions: Dict = None,
        multiplier: int = 1,
        drawer: Callable = lambda x: x,
    ):
        for key in images.keys():
            imgs = images[key]
            if imgs is None:
                log_warning(f"No images found for key: {key}. Skipping.")
                continue
            img_list = []
            for i in tqdm(range(imgs.shape[0])):
                image_i = drawer(imgs[i] * multiplier)
                if exteroceptions is not None and len(exteroceptions[key]) != 0:
                    image_i = draw_keypoints(
                        image_i, exteroceptions[key][i].reshape(-1, 2)
                    )
                img_list.append(image_i)

            images_to_video(img_list, output_path, f"{key}_{video_key}")

    @staticmethod
    def monocular_save(
        observations: Dict,
        video_key: str,
        output_path: str,
        multiplier: int = 1,
        drawer: Callable = lambda x: x,
        draw_exteroception: bool = True,
    ):
        images = observations[video_key]
        if (
            PrivilegeType.EXTEROCEPTION.value in observations.keys()
            and draw_exteroception
        ):
            exteroceptions = observations[PrivilegeType.EXTEROCEPTION.value]
        else:
            exteroceptions = None
        VideoCreator._sub_function(
            images,
            output_path,
            video_key,
            exteroceptions,
            multiplier,
            drawer,
        )


def visualize_data_dict(f: Dict, output_path: str):
    observations = f["observations"]

    if PrivilegeType.MASK.value in observations.keys():
        VideoCreator.monocular_save(
            observations,
            PrivilegeType.MASK.value,
            output_path,
            255,
            draw_exteroception=False,
        )

    if Modality.GEOMAP.value in observations.keys():
        from embodichain.utils.utility_3d import gen_disp_colormap

        VideoCreator.monocular_save(
            observations,
            Modality.GEOMAP.value,
            output_path,
            1,
            lambda x: (gen_disp_colormap(x).transpose(1, 2, 0) * 255).astype(np.uint8),
            draw_exteroception=False,
        )

    VideoCreator.monocular_save(observations, Modality.IMAGES.value, output_path)


def main(args):

    data_path = args.data_path
    output_path = args.output_path
    assert data_path.endswith(".hdf5"), "Data path must have format of .hdf5"
    with h5py.File(data_path, "r") as f:
        from embodichain.data.data_engine.data_dict_extractor import (
            CompressedVideoHDF5,
        )
        import hdfdict

        data = hdfdict.load(data_path)
        data = CompressedVideoHDF5(output_path).safe_filter(data)

        visualize_data_dict(data, output_path)
        robot_meta = data["robot_meta"]
        arm_dofs = robot_meta["arm_dofs"][()]
        indices_generator = ActionIndicesGenerator(arm_dofs)

        actions = f[Modality.ACTIONS.value][()]
        key_names = indices_generator.global_mapping.mapping_from_name_to_indices.keys()
        log_info(f"Arm dofs: {arm_dofs}", color="green")
        indices_dict = {}
        for key_name in key_names:
            indices_dict[key_name] = indices_generator.get([key_name])
        draw_action_distribution(actions, indices_dict, output_path, smooth=args.smooth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data file.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output video file.",
        default="./outputs",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="whether smooth joints.",
    )
    args = parser.parse_args()

    main(args)
