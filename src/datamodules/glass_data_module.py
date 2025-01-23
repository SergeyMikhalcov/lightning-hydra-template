from typing import Any, Dict, Optional
from omegaconf import DictConfig

from src.datamodules.datasets import InpaintingDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


class GlassDataModule(SingleDataModule):
    def __init__(self, 
                 datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )
        
    def setup(self, stage: Optional[str] = None) -> None:
        if not self.train_set and not self.valid_set and not self.test_set:
            transforms_train = TransformsWrapper(self.transforms.get("train"))

            transforms_test = TransformsWrapper(
            self.transforms.get("valid_test_predict")
            )
        self.train_set = InpaintingDataset(
                data_path = self.cfg_datasets.get("train_data_path"),
                target_path = self.cfg_datasets.get("train_target_path"),  
                mask_path=None,
                read_mode = "pillow",
                transforms=transforms_train,
                include_names=True
            )
        
        self.valid_set = InpaintingDataset(
                data_path = self.cfg_datasets.get("val_data_path"),
                target_path = self.cfg_datasets.get("val_target_path"),  
                mask_path=None,
                read_mode = "pillow",
                transforms=transforms_test,
                include_names=True
            )
            
        self.train_set.transforms = transforms_train
        self.valid_set.transforms = transforms_test
            # self.test_set.transforms = transforms_test
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
