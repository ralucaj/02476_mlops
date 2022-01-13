"""
LFW dataloading
"""
import argparse
import glob
import time
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.path_to_folder = path_to_folder
        self.filenames = []
        for filename in glob.iglob(self.path_to_folder + '**/*.jpg', recursive=True):
            self.filenames.append(filename)
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = os.path.join(
            self.path_to_folder, self.filenames[idx]
        )
        img = Image.open(image_path)
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='/mnt/c/Users/raluca/Documents/lfw-deepfunneled/', type=str)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        plt.rcParams["savefig.bbox"] = 'tight'
        imgs = next(iter(dataloader))
        print(len(imgs))

        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = transforms.functional.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig('reports/figures/data_viz.png')

    if args.get_timing:
        # lets do so repetitions
        workers = [1, 2, 4, 8]
        timings_avg = []
        timings_std = []
        for num_workers in workers:
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False,
                            num_workers=num_workers)
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > 100:
                        break
                end = time.time()

                res.append(end - start)

            res = np.array(res)
            timings_avg.append(np.mean(res))
            timings_std.append(np.std(res))
            print(f'Workers: {num_workers}, Timing: {np.mean(res)}+-{np.std(res)}')
        plt.errorbar(workers, timings_avg, timings_std)
        plt.savefig('reports/figures/error_bars.png')