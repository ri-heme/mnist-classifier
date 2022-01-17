"""
LFW dataloading
"""
import argparse
import enum
import time
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dotenv import find_dotenv


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

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=None, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        from torchvision.transforms.functional import to_pil_image
        plt.rcParams["savefig.bbox"] = 'tight'
        nrow = 8
        batch = iter(dataloader).next()
        grid = list(make_grid(batch[:nrow,:,:,:], nrow))
        fig, axs = plt.subplots(ncols=len(grid), squeeze=False)
        for i, img in enumerate(grid):
            img = to_pil_image(img.detach())
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        save_path = Path(find_dotenv(), "..", "models", "lfw", "lfw.png")
        fig.savefig(save_path)
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')
