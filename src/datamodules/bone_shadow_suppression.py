from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from src.datamodules.datasets import InpaintingDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


class BonesXRayDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            # During testing on MNIST, transformations should be the same
            # because we split initial train + test into train + valid + test
            transforms_train = TransformsWrapper(self.transforms.get("train"))

            transforms_test = TransformsWrapper(
                self.transforms.get("valid_test_predict")
            )
            train_set = InpaintingDataset(
                data_path = self.cfg_datasets.get("data_path"),
                target_path = self.cfg_datasets.get("target_path"),  
                mask_path=self.cfg_datasets.get("mask_path"),
                read_mode = "uint16",
                transforms=transforms_train,
                include_names=True
            )
            seed = self.cfg_datasets.get("seed")
            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=train_set,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )
            
            self.train_set.transforms = transforms_train
            self.valid_set.transforms = transforms_test
            self.test_set.transforms = transforms_test
            # print(self.train_set.transforms)
            # print(self.valid_set.transforms)
            # print(self.test_set.transforms)
        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
