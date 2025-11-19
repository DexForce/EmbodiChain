import os
import yaml
import torch
from typing import Dict
import pytest
from embodichain.agents.dexforce_vla import simple_build_model
from embodichain.data.enum import Modality, PrivilegeType
from embodichain.agents.dexforce_vla.train.utils import ConfigParser
from embodichain import embodichain_dir


def run_datasets(
    mode,
):
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    from embodichain.data.dataset import PourWaterHDF5, EMBODICHAIN_DEFAULT_DATA_ROOT

    dataset = PourWaterHDF5()

    from omegaconf.omegaconf import OmegaConf

    # Read the config
    # https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#config-node-interpolation
    config = OmegaConf.load(
        os.path.join(
            embodichain_dir,
            "agents",
            "dexforce_vla",
            "configs",
            "base.yaml",
        )
    )
    config = OmegaConf.to_container(config, resolve=True)

    from embodichain.agents.dexforce_vla.train.vla_dataset import (
        DataCollatorForVLAConsumerDataset,
        VLAConsumerDataset,
    )

    # Dataset and DataLoaders creation:
    train_dataset = VLAConsumerDataset(
        config=config["dataset"],
        data_path=os.path.join(
            EMBODICHAIN_DEFAULT_DATA_ROOT, "extract", "PourWaterHDF5"
        ),
        camera_used=ConfigParser.get_camera_used(config),
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        state_history_len=config["common"]["state_history_len"],
        chunk_size=config["common"]["test_action_chunk_size"],
        image_size=config["common"]["image_size"],
        replace_bg_prob=0,
        state_noise_snr=10,
        mode=mode,
        use_precomp_lang_embed=True,
        batch_size=1,
    )
    gt_size = {
        Modality.IMAGES.value: (
            config["common"]["img_history_size"],
            config["common"]["num_cameras"],
            config["common"]["image_size"],
            config["common"]["image_size"],
            3,
        ),
        Modality.GEOMAP.value: (
            config["common"]["img_history_size"],
            config["common"]["num_cameras"],
            config["common"]["image_size"],
            config["common"]["image_size"],
            3,
        ),
        PrivilegeType.EXTEROCEPTION.value: (
            config["common"]["img_history_size"],
            config["common"]["num_cameras"],
            config["common"]["test_action_chunk_size"],
            30,
            2,
        ),
        PrivilegeType.MASK.value: (
            1,
            config["common"]["num_cameras"],
            config["common"]["image_size"],
            config["common"]["image_size"],
            1,
        ),
    }
    for d in train_dataset:
        for key, val in d.items():
            if key in gt_size:
                assert (
                    val.shape == gt_size[key]
                ), f"{key} size {val.shape} vs gt size {gt_size[key]}"

        break
    device = "cuda"
    data_collator = DataCollatorForVLAConsumerDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, collate_fn=data_collator, num_workers=0
    )

    for batch in train_dataloader:
        for key, val in batch.items():
            batch[key] = val.to(device)
        break


if __name__ == "__main__":
    run_datasets("training")
    run_datasets("evaluation")
