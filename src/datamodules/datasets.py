import os
import json
import random
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import pydicom as pcm

from src.datamodules.components.dataset import BaseDataset
from src.datamodules.components.h5_file import H5PyFile
from src.datamodules.components.parse import parse_image_paths


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        json_path: Optional[str] = None,
        txt_path: Optional[str] = None,
        data_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
        include_names: bool = False,
        shuffle_on_load: bool = True,
        label_type: str = "torch.LongTensor",
        **kwargs: Any,
    ) -> None:
        """ClassificationDataset.

        Args:
            json_path (:obj:`str`, optional): Path to annotation json.
            txt_path (:obj:`str`, optional): Path to annotation txt.
            data_path (:obj:`str`, optional): Path to HDF5 file or images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            include_names (bool): If True, then `__getitem__` method would return image
                name/path value with key `name`. Default to False.
            shuffle_on_load (bool): Deterministically shuffle the dataset on load
                to avoid the case when Dataset slice contains only one class due to
                annotations dict keys order. Default to True.
            label_type (str): Label torch.tensor type. Default to torch.FloatTensor.
            kwargs (Any): Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms, read_mode, to_gray)
        if (json_path and txt_path) or (not json_path and not txt_path):
            raise ValueError("Requires json_path or txt_path, but not both.")
        elif json_path:
            json_path = Path(json_path)
            if not json_path.is_file():
                raise RuntimeError(f"'{json_path}' must be a file.")
            with open(json_path) as json_file:
                self.annotation = json.load(json_file)
        else:
            txt_path = Path(txt_path)
            if not txt_path.is_file():
                raise RuntimeError(f"'{txt_path}' must be a file.")
            self.annotation = {}
            with open(txt_path) as txt_file:
                for line in txt_file:
                    _, label, path = line[:-1].split("\t")
                    self.annotation[path] = label

        self.keys = list(self.annotation)
        if shuffle_on_load:
            random.Random(shuffle_on_load).shuffle(self.keys)

        self.include_names = include_names
        self.label_type = label_type

        data_path = "" if data_path is None else data_path
        self.data_path = data_path = Path(data_path)
        self.data_file = None
        if data_path.is_file():
            if data_path.suffix != ".h5":
                raise RuntimeError(f"'{data_path}' must be a h5 file.")
            self.data_file = H5PyFile(str(data_path), **kwargs)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image = self._process_image_(image)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        if self.include_names:
            return {"image": image.float(), "label": label, "name": key}
        return {"image": image.float(), "label": label}

    def get_weights(self) -> List[float]:
        label_list = [self.annotation[key] for key in self.keys]
        weights = 1.0 / np.bincount(label_list)
        return weights.tolist()
    
    
class InpaintingDataset(BaseDataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        target_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "uint16",
        to_gray: bool = False,
        include_names: bool = True,
        label_type: str = "torch.LongTensor",
        **kwargs: Any,
    ) -> None:
        """InpaintingDataset.

        Args:
            data_path (:obj:`str`, optional): Path to HDF5 file or images source dir.
            target_path (:obj:`str`, optional): Path to target images with the same filenames.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            include_names (bool): If True, then `__getitem__` method would return image
                name/path value with key `name`. Default to False.
            shuffle_on_load (bool): Deterministically shuffle the dataset on load
                to avoid the case when Dataset slice contains only one class due to
                annotations dict keys order. Default to True.
            label_type (str): Label torch.tensor type. Default to torch.FloatTensor.
            kwargs (Any): Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms, read_mode, to_gray)
        self.images = []
        self.targets = []
        self.masks = []
        
        for a in os.walk(data_path):
            if len(a[2]):
                for img in a[2]:
                    self.images.append(os.path.join(a[0], img))
                    self.targets.append(os.path.join(a[0],
                                                     img).replace(data_path,
                                                                  target_path))
                    if mask_path:
                        self.masks.append(os.path.join(a[0],
                                                     img).replace(data_path,
                                                                  mask_path))

        # for img_name in os.listdir(data_path):
        #     if img_name in os.listdir(target_path):
        #         self.images.append(os.path.join(data_path, img_name))
        #         self.targets.append(os.path.join(target_path, img_name))
        #     if mask_path and img_name in os.listdir(mask_path):
        #         self.masks.append(os.path.join(mask_path, img_name))
        
        #self.transforms = transforms
        #print(self.transforms)
        self.include_names = include_names
        self.label_type = label_type

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path = self.images[index]
        target_path = self.targets[index]
        image = self._read_image_(img_path)
        target = self._read_image_with_mode(target_path, read_mode='mask')
        image, target = self._process_image_(image=image, mask=target)
        output = {"image": image.float(), "target": np.expand_dims(target, axis=0)}
        if self.include_names:
            output["name"] = img_path
        if len(self.masks):
            mask_path = self.masks[index]
            mask = self._read_image_(mask_path)
            mask[mask==255] = 1.0
            output["mask"] = torch.Tensor(mask).unsqueeze(0).unsqueeze(0)
        # print(output["image"].shape, output["target"].shape)
        return output


class ClassificationVicRegDataset(ClassificationDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image1, image2 = np.copy(image), np.copy(image)
        # albumentations returns random augmentation on each __call__
        z1 = self._process_image_(image1)
        z2 = self._process_image_(image2)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        return {"z1": z1.float(), "z2": z2.float(), "label": label}


class NoLabelsDataset(BaseDataset):
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        dir_paths: Optional[List[str]] = None,
        txt_paths: Optional[List[str]] = None,
        json_paths: Optional[List[str]] = None,
        dirname: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
        include_names: bool = False,
    ) -> None:
        """NoLabelsDataset.

        Args:
            file_paths (:obj:`List[str]`, optional): List of files.
            dir_paths (:obj:`List[str]`, optional): List of directories.
            txt_paths (:obj:`List[str]`, optional): List of TXT files.
            json_paths (:obj:`List[str]`, optional): List of JSON files.
            dirname (:obj:`str`, optional): Images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            include_names (bool): If True, then `__getitem__` method would return image
                name/path value with key `name`. Default to False.
        """

        super().__init__(transforms, read_mode, to_gray)
        if file_paths or dir_paths or txt_paths:
            self.keys = parse_image_paths(
                file_paths=file_paths, dir_paths=dir_paths, txt_paths=txt_paths
            )
        elif json_paths:
            self.keys = []
            for json_path in json_paths:
                with open(json_path) as json_file:
                    data = json.load(json_file)
                for path in data.keys():
                    self.keys.append(path)
        else:
            raise ValueError("Requires data_paths or json_paths.")
        self.dirname = Path(dirname if dirname else "")
        self.include_names = include_names

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        path = self.dirname / Path(key)
        image = self._read_image_(path)
        image = self._process_image_(image)
        if self.include_names:
            return {"image": image, "name": key}
        return {"image": image}

    def __len__(self) -> int:
        return len(self.keys)


class BreastPairedDataset(BaseDataset):

    class InputType(Enum):
        ClassIndex = 0
        Logits = 1

    ClasName2Index = {'A': 0,
                      'B': 1,
                      'C': 2,
                      'D': 3
                      }
    Index2Classname = {0: 'A',
                       1: 'B',
                       2: 'C',
                       3: 'D'
                       }
    Logits2Index = np.argmax
    n_classes = 4

    def __init__(self, csv_path: str, target_column: Union[str, List[str]],
                 dcm_path_col: Union[str, List[str]], id_column: str,
                 order: List[str] = ['CC', 'MLO'],
                 transforms=None,
                 use_brut_invariant: bool = False,
                 one_of_postproc: bool = False,
                 replace_names: bool = True) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path, index_col=0)
        assert id_column in self.df.columns, ValueError(f'Column {id_column} not in DataFrame.')
        self.id_column = id_column
        self.researchs_ids = [uid for uid in self.df[id_column].unique() if
                              len(self.df[self.df[id_column] == uid]) == len(
                                  order)
                              ]
        self.order = order
        self.dcm_path_col = dcm_path_col
        self.target_col = target_column
        self.transforms = transforms
        self.brut_inv = use_brut_invariant
        self.one_of_aug = one_of_postproc
        self.replace_names = replace_names

    @staticmethod
    def read_dcm(path2dcm):
        dcm = pcm.dcmread(path2dcm)
        w, h = int(dcm.Rows), int(dcm.Columns)
        raw_1d = np.frombuffer(dcm[0x00310410].value, dtype=np.uint16)
        raw = np.reshape(raw_1d, (h, w))
        return raw

    def __getitem__(self, index):
        research_df = self.df[self.df[self.id_column] == self.researchs_ids[index]]
        if isinstance(self.dcm_path_col, str):
            cc_path = research_df[research_df['ViewPosition'] ==
                                  'CC'][self.dcm_path_col].iloc[0]
            mlo_path = research_df[research_df['ViewPosition'] ==
                                   'MLO'][self.dcm_path_col].iloc[0]
            if self.replace_names:
                cc_path = cc_path.replace('aserver-images',
                                          'media').replace('\\', '/')
                mlo_path = mlo_path.replace('aserver-images',
                                            'media').replace('\\', '/')
            cc_img = np.array(Image.open(Path(cc_path)))
            mlo_img = np.array(Image.open(Path(mlo_path)))
        elif (isinstance(self.dcm_path_col, Iterable) and not self.one_of_aug):
            cc_path = research_df[research_df['ViewPosition'] ==
                                  'CC'][self.dcm_path_col].apply(
                lambda x: '/'.join(x), axis=1).iloc[0]
            mlo_path = research_df[research_df['ViewPosition'] ==
                                   'MLO'][self.dcm_path_col].apply(
                lambda x: '/'.join(x), axis=1).iloc[0]
            if self.replace_names:
                cc_path = cc_path.replace('aserver-images',
                                          'media').replace('\\', '/')
                mlo_path = mlo_path.replace('aserver-images',
                                            'media').replace('\\', '/')
            cc_img = self.read_dcm(Path(cc_path))
            mlo_img = self.read_dcm(Path(mlo_path))
        elif (isinstance(self.dcm_path_col, Iterable) and self.one_of_aug):
            postproc = random.choice(self.dcm_path_col)
            cc_path = research_df[research_df['ViewPosition'] ==
                                  'CC'][postproc].iloc[0]
            mlo_path = research_df[research_df['ViewPosition'] ==
                                   'MLO'][postproc].iloc[0]
            if self.replace_names:
                cc_path = cc_path.replace('aserver-images',
                                          'media').replace('\\', '/')
                mlo_path = mlo_path.replace('aserver-images',
                                            'media').replace('\\', '/')
            cc_img = np.array(Image.open(Path(cc_path)))
            mlo_img = np.array(Image.open(Path(mlo_path)))
        target = research_df[self.target_col].iloc[0]
        if type(target) == str:
            target = torch.tensor(self.ClasName2Index[target], dtype=torch.long)
        else:
            target = torch.tensor(target)
        if self.transforms is not None:
            augmentation = self.transforms(image=cc_img)
            cc_img = augmentation['image']
            augmentation = self.transforms(image=mlo_img)
            mlo_img = augmentation['image']
        if self.brut_inv:
            if random.random() > 0.5:
                cc_img, mlo_img = 1-cc_img, 1-mlo_img
        stacked = torch.concat((cc_img, mlo_img), dim=0)

        return {'image': stacked,
                'target': target,
                'paths': {
                    'cc_path': cc_path,
                    'mlo_path': mlo_path
                    }
                }

    def __len__(self):
        return len(self.researchs_ids)