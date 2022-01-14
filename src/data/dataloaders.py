__all__ = ["CorruptedMNIST", "dataloaders"]

from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv
from torch.utils.data import DataLoader, Dataset


def _find_data_path():
    return Path(find_dotenv(), "..", "data", "processed")


class CorruptedMNIST(Dataset):
    def __init__(self, mode="train") -> None:
        super().__init__()
        parent_path = _find_data_path()
        images, labels = [], []
        for path in parent_path.glob(f"{mode}*.npz"):
            npz = np.load(path)
            images.append(npz["images"])
            labels.append(npz["labels"])
        self.images = np.concatenate(images)
        self.labels = np.concatenate(labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx])
        label = torch.LongTensor(self.labels[[idx]])
        return image, label


def dataloaders(batch_size=64):
    train_dataloader = DataLoader(CorruptedMNIST(), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(CorruptedMNIST("test"), batch_size=batch_size)
    return train_dataloader, test_dataloader
