#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import h5py
from tqdm import tqdm
from embodichain.data.enum import (
    Modality,
    PrivilegeType,
    ActionMode,
    JointType,
    TeleoperationData,
    EefType,
    CameraName,
    EndEffector,
)

# Import CompressedVideoHDF5 class
from embodichain.data.data_engine.data_dict_extractor import CompressedVideoHDF5
from embodichain.utils.logger import log_warning, log_info

DEFAULT_IMAGE_NUM_IN_A_CHUNK = 5


def load_metadata(metadata_path: Path) -> List[Dict]:
    """Load metadata JSONL file"""
    meta = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if (
                TeleoperationData.HEAD_CAMERA.value
                in entry[TeleoperationData.CAMERA_TYPE_KEY.value]
                and TeleoperationData.LEFT_PLACE.value
                in entry[TeleoperationData.CAMERA_TYPE_KEY.value]
            ):
                meta.append(entry)
    meta.sort(key=lambda x: x[TeleoperationData.TIMESTAMP_KEY.value])
    return meta


def load_qpos(qpos_path: Path, joint_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load qpos data from JSON file"""
    with open(qpos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data[TeleoperationData.FRAMES.value]
    frames.sort(key=lambda x: x[TeleoperationData.TIMESTAMP_KEY.value])

    ts = np.array(
        [f[TeleoperationData.TIMESTAMP_KEY.value] for f in frames], dtype=np.float64
    )
    qpos = np.stack(
        [
            np.array(
                [f[TeleoperationData.DATA.value].get(k, 0.0) for k in joint_keys],
                dtype=np.float32,
            )
            for f in frames
        ],
        axis=0,
    )

    return ts, qpos


def interp_qpos(
    qpos_ts: np.ndarray, qpos: np.ndarray, target_ts: np.ndarray
) -> np.ndarray:
    """Interpolate qpos to target timestamps"""
    D = qpos.shape[1]
    out = np.zeros((len(target_ts), D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(target_ts, qpos_ts, qpos[:, d])
    return out


def load_images_batch(
    data_dir: Path,
    meta: List[Dict],
    camera_type: str = TeleoperationData.HEAD_CAMERA.value,
) -> Dict[str, np.ndarray]:
    """Load images in batches to avoid memory issues"""
    image_data = {}

    # Initialize image arrays
    left_images = []
    right_images = []

    # Load images in batches
    for i, m in tqdm(
        enumerate(meta), desc=f"Loading {camera_type} images", total=len(meta)
    ):
        # Load left eye image
        pic_name = m[TeleoperationData.IMAGE_PATH_KEY.value].split("/")[-1]
        left_img_path = (
            data_dir / camera_type / TeleoperationData.LEFT_PLACE.value / pic_name
        )
        if os.path.exists(str(left_img_path)):
            left_img = cv2.imread(str(left_img_path))
            left_images.append(left_img)

        # Load corresponding right eye image
        # Infer right eye path from left eye path
        right_img_path = (
            data_dir / camera_type / TeleoperationData.RIGHT_PLACE.value / pic_name
        )

        if os.path.exists(str(right_img_path)):
            right_img = cv2.imread(str(right_img_path))
            right_images.append(right_img)

    if len(left_images) == 0:
        log_warning(
            "Camera type {} for {} is empty.".format(
                camera_type, TeleoperationData.LEFT_PLACE.value
            )
        )
    if len(right_images) == 0:
        log_warning(
            "Camera type {} for {} is empty.".format(
                camera_type, TeleoperationData.RIGHT_PLACE.value
            )
        )

    if len(left_images) > 0 and left_images[0] is not None:
        left_images = np.array(left_images).astype(np.uint8)
        if camera_type == TeleoperationData.HEAD_CAMERA.value:
            image_data[
                TeleoperationData.CAM_HIGH_PREFIX.value
                + "_"
                + TeleoperationData.LEFT_PLACE.value
            ] = left_images
        elif camera_type == TeleoperationData.HAND_CAMERA.value:
            image_data[
                TeleoperationData.CAM_HAND_PREFIX.value
                + "_"
                + TeleoperationData.LEFT_PLACE.value
            ] = left_images
    if len(right_images) > 0 and right_images[0] is not None:
        right_images = np.array(right_images).astype(np.uint8)
        if camera_type == TeleoperationData.HEAD_CAMERA.value:
            image_data[
                TeleoperationData.CAM_HIGH_PREFIX.value
                + "_"
                + TeleoperationData.RIGHT_PLACE.value
            ] = right_images
        elif camera_type == TeleoperationData.HAND_CAMERA.value:
            image_data[
                TeleoperationData.CAM_HAND_PREFIX.value
                + "_"
                + TeleoperationData.RIGHT_PLACE.value
            ] = right_images

    return image_data


def save_aligned_trajectory_json(
    original_qpos_path: Path,
    meta: List[Dict],
    qpos_interp: np.ndarray,
    joint_keys: List[str],
    output_path: Path,
):
    """Save aligned trajectory JSON file, maintaining original format but keeping only data corresponding to video frames"""

    # Load original qpos file to get complete structure
    with open(original_qpos_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Create aligned frame data
    aligned_frames = []

    for i, (m, qpos_values) in tqdm(
        enumerate(zip(meta, qpos_interp)),
        desc="Saving aligned trajectory JSON",
        total=len(meta),
    ):
        frame_data = {
            "frame_id": i,  # Start numbering from 0
            "timestamp": float(m[TeleoperationData.TIMESTAMP_KEY.value]),
            "data": {},
        }

        # Fill joint data
        for j, key in enumerate(joint_keys):
            frame_data["data"][key] = float(qpos_values[j])

        # Keep other fields from original data (if they exist)
        if (
            TeleoperationData.FRAMES.value in original_data
            and len(original_data[TeleoperationData.FRAMES.value]) > 0
        ):
            original_frame = original_data[TeleoperationData.FRAMES.value][
                0
            ]  # Use first frame as template
            for key in original_frame[TeleoperationData.DATA.value]:
                if (
                    key not in joint_keys
                ):  # Only keep additional fields not in joint_keys
                    frame_data["data"][key] = original_frame[
                        TeleoperationData.DATA.value
                    ][key]

        aligned_frames.append(frame_data)

    # Build complete aligned trajectory JSON
    aligned_data = {TeleoperationData.FRAMES.value: aligned_frames}

    # Save aligned JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_data, f, indent=2, ensure_ascii=False)

    log_info(f"💾 Saved aligned trajectory JSON to: {output_path}")
    return output_path


def convert_to_compressed_hdf5(
    data_dir: Path,
    output_path: Path,
    joint_keys: List[str],
    camera_types: List[str] = [
        TeleoperationData.HEAD_CAMERA.value,
        TeleoperationData.HAND_CAMERA.value,
    ],
):
    """Convert raw teleoperation data to compressed HDF5 format"""
    os.makedirs(output_path.parent, exist_ok=True)

    # Check necessary files
    metadata_path = data_dir / TeleoperationData.METADATA_FILE.value
    qpos_pattern = TeleoperationData.QPOS_PATTERN.value

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Find qpos files
    qpos_files = list(data_dir.glob(qpos_pattern))
    if not qpos_files:
        raise FileNotFoundError(f"No qpos files found matching pattern: {qpos_pattern}")

    qpos_path = qpos_files[0]  # Take the first matching file

    log_info(f"📥 Loading metadata from: {metadata_path}")
    meta = load_metadata(metadata_path)
    ts_meta = np.array(
        [m[TeleoperationData.TIMESTAMP_KEY.value] for m in meta], dtype=np.float64
    )

    log_info(f"📥 Loading qpos data from: {qpos_path.name}")
    ts_q, qpos = load_qpos(qpos_path, joint_keys)

    log_info(f"🔄 Interpolating qpos to image timestamps...")
    qpos_interp = interp_qpos(ts_q, qpos, ts_meta)
    aligned_json_path = output_path.parent / f"aligned_{qpos_path.name}"
    save_aligned_trajectory_json(
        qpos_path, meta, qpos_interp, joint_keys, aligned_json_path
    )

    for j, key in enumerate(joint_keys):
        if EndEffector.GRIPPER.value.lower() in key.lower():
            qpos_interp[:, j] = qpos_interp[:, j] / 1000.0
        if EndEffector.DEXTROUSHAND.value.lower() in key.lower():
            qpos_interp[:, j] = qpos_interp[:, j] / 100.0

    log_info("Trajectory length: {}.".format(qpos_interp.shape[0]))

    # Prepare data dictionary
    data_dict = {
        TeleoperationData.OBSERVATIONS.value: {
            TeleoperationData.IMAGES.value: {},
            TeleoperationData.QPOS.value: qpos_interp,
        },
        TeleoperationData.ACTION.value: qpos_interp,
    }

    # Load images for all camera types
    camera_images = load_images_batch(
        data_dir, meta, TeleoperationData.HEAD_CAMERA.value
    )
    data_dict[TeleoperationData.OBSERVATIONS.value][
        TeleoperationData.IMAGES.value
    ].update(camera_images)

    camera_images = load_images_batch(
        data_dir, meta, TeleoperationData.HAND_CAMERA.value
    )
    data_dict[TeleoperationData.OBSERVATIONS.value][
        TeleoperationData.IMAGES.value
    ].update(camera_images)

    # Use CompressedVideoHDF5 for compressed saving
    log_info(f"💾 Saving compressed HDF5 to: {output_path}")
    cvhdf5 = CompressedVideoHDF5(
        str(output_path),
        chunks=int(qpos_interp.shape[0] / DEFAULT_IMAGE_NUM_IN_A_CHUNK),
    )
    cvhdf5.dump(
        data_dict,
        video_names=[TeleoperationData.IMAGES.value],  # Only include images
        dtypes=[np.uint8],  # Corresponding data types
    )

    log_info(f"✅ Successfully converted to compressed HDF5: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw teleoperation data to compressed HDF5 format using CompressedVideoHDF5"
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to the raw teleoperation data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file path (default: <data_dir>/compressed_data.hdf5)",
    )
    parser.add_argument(
        "--joint_keys",
        type=str,
        nargs="+",
        default=[
            "ANKLE",
            "KNEE",
            "BUTTOCK",
            "WAIST",  # 3
            "NECK1",
            "NECK2",
            "LEFT_J1",  # 6
            "LEFT_J2",
            "LEFT_J3",
            "LEFT_J4",
            "LEFT_J5",
            "LEFT_J6",
            "LEFT_J7",
            "LEFT_GRIPPER",
            "RIGHT_J1",  # 14
            "RIGHT_J2",
            "RIGHT_J3",
            "RIGHT_J4",
            "RIGHT_J5",
            "RIGHT_J6",
            "RIGHT_J7",
            "RIGHT_GRIPPER",
            "LEFT_HAND_THUMB1",  # 22
            "LEFT_HAND_THUMB2",
            "LEFT_HAND_INDEX",
            "LEFT_HAND_MIDDLE",
            "LEFT_HAND_RING",
            "LEFT_HAND_PINKY",
            "RIGHT_HAND_THUMB1",  # 28
            "RIGHT_HAND_THUMB2",
            "RIGHT_HAND_INDEX",
            "RIGHT_HAND_MIDDLE",
            "RIGHT_HAND_RING",
            "RIGHT_HAND_PINKY",
        ],
        help="List of joint keys in the qpos data",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=[
            TeleoperationData.HEAD_CAMERA.value,
            TeleoperationData.HAND_CAMERA.value,
        ],
        help="Camera types to include (default: head, hand)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    output_path = (
        Path(args.output) if args.output else data_dir / "_compressed_data.hdf5"
    )

    try:
        convert_to_compressed_hdf5(
            data_dir=data_dir,
            output_path=output_path,
            joint_keys=args.joint_keys,
            camera_types=args.cameras,
        )
    except Exception as e:
        log_info(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
