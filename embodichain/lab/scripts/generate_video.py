from embodichain.utils.logger import log_info, log_warning

try:
    import h5ffmpeg as hf
except Exception as e:
    log_warning("Fail to import h5ffmpeg.")
import h5py
import argparse
import numpy as np
import os
from tqdm import tqdm
from dexsim.utility import images_to_video
from typing import Dict, Callable, Tuple
from embodichain.utils.visualizer import draw_keypoints, draw_action_distribution
from embodichain.data.enum import EefType, JointType, Modality, PrivilegeType
from embodichain.data.data_engine.indices_unifier import ActionIndicesGenerator


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
    os.makedirs(output_path, exist_ok=True)
    assert data_path.endswith(".hdf5"), "Data path must have format of .hdf5"
    with h5py.File(data_path, "r") as f:
        from embodichain.data.data_engine.data_dict_extractor import (
            CompressedVideoHDF5,
        )
        import hdfdict

        data = hdfdict.load(data_path)
        data = CompressedVideoHDF5(
            output_path, chunks=data["chunks"].item()
        ).safe_filter(data)

        # NOTE: DO NOT USE THIS IN SCRIPT, IT IS FOR DEBUGGING PURPOSES ONLY
        # slice_id = 20
        # data_copy = hdfdict.load(data_path)
        # data_copy = CompressedVideoHDF5(output_path).safe_filter(data_copy, slice_id=slice_id)

        # a = data["observations"]["images"]["cam_high"][slice_id]
        # b = data_copy["observations"]["images"]["cam_high"]
        # print(a, b.shape)
        # delta = a-b[slice_id]
        # print(np.linalg.norm(delta))

        visualize_data_dict(data, output_path)
        if "robot_meta" in data.keys():
            log_warning("Simulation data.")
            robot_meta = data["robot_meta"]
            arm_dofs = robot_meta["arm_dofs"][()]
            actions = f[Modality.ACTIONS.value][()]
        else:
            from embodichain.data.data_engine.datasets.sim_real_unified_dict_dataset import (
                RobotRealDataRouter,
            )

            log_warning("Real data.")
            actions, _, _, _, _ = RobotRealDataRouter(
                robot_name=args.robot_name
            ).realdata2simdata(f)
        indices_generator = ActionIndicesGenerator(arm_dofs)

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
    parser.add_argument("--robot_name", default="DexforceW1", type=str)
    args = parser.parse_args()

    main(args)
