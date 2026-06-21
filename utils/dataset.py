import os
import random

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset
from torchvision import transforms


def read_manifest(path):
    images = []
    labels = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("Invalid manifest line in {}: {}".format(path, line))
            if len(parts) >= 3:
                images.append((parts[0], parts[1]))
                labels.append(parts[2])
            else:
                images.append(parts[0])
                labels.append(parts[1])
    return images, labels


def read_array(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    del dataset
    return data


def channels_first_to_last(array):
    if array.ndim == 3 and array.shape[0] <= 16 and array.shape[1] > 16 and array.shape[2] > 16:
        return np.transpose(array, (1, 2, 0))
    return array


def channels_are_same(array):
    if array.ndim != 3 or array.shape[2] < 2:
        return False
    first = array[..., 0]
    return all(np.array_equal(first, array[..., index]) for index in range(1, array.shape[2]))


def normalize_image_array(array, image_mode="auto"):
    array = channels_first_to_last(np.asarray(array))
    mode = (image_mode or "auto").lower()
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if mode in ["gray", "grayscale", "single", "single_channel"]:
        if array.ndim == 3:
            if channels_are_same(array):
                array = array[..., 0]
            else:
                gray = array[..., :3].astype(np.float32).mean(axis=2)
                if np.issubdtype(array.dtype, np.integer):
                    gray = np.rint(gray).astype(array.dtype)
                array = gray
        return array
    if mode in ["rgb", "three_channel"]:
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=2)
        return array[..., :3] if array.ndim == 3 and array.shape[2] > 3 else array
    if mode == "auto" and array.ndim == 3 and array.shape[2] in [3, 4] and channels_are_same(array):
        array = array[..., 0]
    return array


def normalize_label_array(label, label_mode="auto"):
    label = np.asarray(label)
    if label.ndim == 3:
        label = channels_first_to_last(label)
        label = label[..., 0] if label.ndim == 3 else label
    mode = (label_mode or "auto").lower()
    if mode in ["binary", "bin", "positive"]:
        return (label > 0).astype(np.int64)
    if mode == "auto":
        unique_values = np.unique(label)
        if unique_values.size <= 2 and unique_values.max(initial=0) > 1:
            return (label > 0).astype(np.int64)
    return label.astype(np.int64)


def image_to_tensor(path, image_mode="auto"):
    image = normalize_image_array(read_array(path), image_mode=image_mode)
    return transforms.ToTensor()(np.asarray(image)).to(torch.float32)


def combine_paired_tensors(tensors, paired_input_mode="stack"):
    if len(tensors) == 1:
        return tensors[0]
    mode = (paired_input_mode or "stack").lower()
    before = tensors[0]
    after = tensors[1]
    diff = torch.abs(after - before)
    if mode in ["stack", "ab", "pair", "paired"]:
        return torch.cat(tensors, dim=0)
    if mode in ["stack_diff", "ab_diff", "diff_abs", "abs_diff"]:
        return torch.cat([before, after, diff], dim=0)
    if mode in ["post_diff", "b_diff", "after_diff"]:
        return torch.cat([after, diff], dim=0)
    if mode in ["stack_signed_diff", "ab_signed_diff", "signed_diff"]:
        return torch.cat([before, after, after - before], dim=0)
    if mode in ["stack_diff_signed", "ab_diff_signed"]:
        return torch.cat([before, after, diff, after - before], dim=0)
    raise ValueError("Unsupported paired_input_mode: {}".format(paired_input_mode))


def spatial_augmentation(image, label, augmentation):
    if not augmentation or not augmentation.get("enable", False):
        return image, label
    if random.random() < float(augmentation.get("hflip_prob", 0.0)):
        image = torch.flip(image, dims=[2])
        label = torch.flip(label, dims=[1])
    if random.random() < float(augmentation.get("vflip_prob", 0.0)):
        image = torch.flip(image, dims=[1])
        label = torch.flip(label, dims=[0])
    if random.random() < float(augmentation.get("rot90_prob", 0.0)):
        k = random.randint(1, 3)
        image = torch.rot90(image, k=k, dims=[1, 2])
        label = torch.rot90(label, k=k, dims=[0, 1])
    return image, label


class FloodDataset(Dataset):
    def __init__(
        self,
        manifest,
        transform,
        train=True,
        image_mode="auto",
        label_mode="auto",
        paired_input_mode="stack",
        augmentation=None,
    ):
        self.images, self.labels = read_manifest(manifest)
        self.transform = transform
        self.train = train
        self.image_mode = image_mode
        self.label_mode = label_mode
        self.paired_input_mode = paired_input_mode
        self.augmentation = augmentation or {}

    def __getitem__(self, index):
        image_item = self.images[index]
        label_path = self.labels[index]
        if isinstance(image_item, (tuple, list)):
            image = combine_paired_tensors(
                [image_to_tensor(path, self.image_mode) for path in image_item],
                paired_input_mode=self.paired_input_mode,
            )
        else:
            image = image_to_tensor(image_item, self.image_mode)
        label = torch.from_numpy(normalize_label_array(read_array(label_path), self.label_mode)).long()
        if self.train:
            image, label = spatial_augmentation(image, label, self.augmentation)
        image = self.transform(image)
        return image, label, os.path.basename(label_path)

    def __len__(self):
        return len(self.images)
