import io
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
    ) -> None:
        """BaseDataset.

        Args:
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
        """

        self.read_mode = read_mode
        self.to_gray = to_gray
        self.transforms = transforms

    def _read_image_(self, image: Any) -> np.ndarray:
        """Read image from source.

        Args:
            image (Any): Image source. Could be str, Path or bytes.

        Returns:
            np.ndarray: Loaded image.
        """

        if self.read_mode == "pillow":
            if not isinstance(image, (str, Path)):
                image = io.BytesIO(image)
            image = np.asarray(Image.open(image).convert("RGB"))
        elif self.read_mode == "cv2":
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.read_mode == "uint16":
            image = np.asarray(Image.open(image)).astype(np.float32)
        elif self.read_mode == "mask":
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        else:
            raise NotImplementedError("use pillow or cv2 or uint16 or mask")
        if self.to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    
    def _read_image_with_mode(self, image: Any, read_mode: str) -> np.ndarray:
        """Read image from source.

        Args:
            image (Any): Image source. Could be str, Path or bytes.

        Returns:
            np.ndarray: Loaded image.
        """

        if read_mode == "pillow":
            if not isinstance(image, (str, Path)):
                image = io.BytesIO(image)
            image = np.asarray(Image.open(image).convert("RGB"))
        elif read_mode == "cv2":
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif read_mode == "uint16":
            image = np.asarray(Image.open(image)).astype(np.float32)
        elif read_mode == "mask":
            image = np.asarray(Image.open(image))            
        else:
            raise NotImplementedError("use pillow or cv2 or uint16 or mask")
        return image


    def _process_image_(self, image: np.ndarray, mask=None) -> torch.Tensor:
        """Process image, including transforms, etc.

        Args:
            image (np.ndarray): Image in np.ndarray format.
            mask (np.ndarray): Mask in np.ndarray format.

        Returns:
            torch.Tensor: Image prepared for dataloader.
        """

        if self.transforms:
            if mask is not None:
                output = self.transforms(image=image, mask=mask)
                output = (output['image'], output['mask'])
            else:
                output = self.transforms(image=image)["image"]

        else:
            if len(image.shape) < 3:
                image = np.expand_dims(image, 2)
            output = torch.from_numpy(image).permute(2, 0, 1)

        return output

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
