"""
LFW dataloading
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = "tight"


class LFWDataset(Dataset):
    def __init__(self, transform) -> None:
        base_path = Path(find_dotenv(), "..", "data", "external", "lfw-deepfunneled")
        self.images = []
        for image_path in base_path.glob("**/*.jpg"):
            self.images.append(image_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.images[index])
        return self.transform(image)


def get_timing(dataloader, batches_to_check):
    res = []
    for _ in range(5):
        start = time.time()
        for batch_idx, _ in enumerate(dataloader):
            if batch_idx > batches_to_check:
                break
        end = time.time()
        res.append(end - start)
    res = np.array(res)
    return res.mean(), res.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=None, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)
    parser.add_argument("-get_timing_plot", action="store_true")

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        nrow = 8
        batch = iter(dataloader).next()
        grid = list(make_grid(batch[:nrow, :, :, :], nrow))
        fig, axs = plt.subplots(ncols=len(grid), squeeze=False)
        for i, img in enumerate(grid):
            img = to_pil_image(img.detach())
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        save_path = Path(find_dotenv(), "..", "models", "lfw", "lfw.png")
        fig.savefig(save_path)

    if args.get_timing:
        # lets do some repetitions
        mean, std = get_timing(dataloader, args.batches_to_check)
        print(f"Timing: {mean}+-{std}")

    if args.get_timing_plot:
        res = []
        for num_workers, _ in enumerate(range(8), 1):
            print(f">>> Loading data with {num_workers} threads.")
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            res.append(get_timing(dataloader, args.batches_to_check))
        res = np.array(res)
        fig, ax = plt.subplots()
        ax.errorbar(np.arange(8) + 1, res[:, 0], yerr=res[:, 1])
        ax.set_ylabel("Average Timing")
        ax.set_xlabel("# threads")
        save_path = Path(find_dotenv(), "..", "models", "lfw", "lfw-timing.png")
        fig.savefig(save_path)
