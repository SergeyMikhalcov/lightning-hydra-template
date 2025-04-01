from typing import Any, Dict, Optional
from omegaconf import DictConfig

from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule
from src.datamodules.datasets import BreastPairedDataset


class BreastDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            # During testing on MNIST, transformations should be the same
            # because we split initial train + test into train + valid + test
            transforms_train = TransformsWrapper(self.transforms.get("train"))
            transforms_test = TransformsWrapper(
                self.transforms.get("valid_test_predict")
            )
            self.train_set = BreastPairedDataset(
                self.cfg_datasets.get("train_csv_path"),
                self.cfg_datasets.get("target_column"),
                self.cfg_datasets.get("dcm_path_col"),
                self.cfg_datasets.get("id_column"),
                self.cfg_datasets.get("order"),
                transforms=transforms_train,
            )
            self.valid_set = BreastPairedDataset(
                self.cfg_datasets.get("val_csv_path"),
                self.cfg_datasets.get("target_column"),
                self.cfg_datasets.get("dcm_path_col"),
                self.cfg_datasets.get("id_column"),
                self.cfg_datasets.get("order"),
                transforms=transforms_test,
            )
            self.test_set = BreastPairedDataset(
                self.cfg_datasets.get("test_csv_path"),
                self.cfg_datasets.get("target_column"),
                self.cfg_datasets.get("dcm_path_col"),
                self.cfg_datasets.get("id_column"),
                self.cfg_datasets.get("order"),
                transforms=transforms_test,
            )
        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
