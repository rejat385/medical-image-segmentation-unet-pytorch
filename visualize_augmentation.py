
import random

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from dataset import PolypDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def print_img(image, mask, type):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Medical Image")

    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"results/aug_example_{type}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    
    dataset = PolypDataset(size=128)

    # Define image augmentations

    test_image = dataset[999]
    img = test_image[0]
    msk = test_image[1]

    print_img(img, msk, "original")

    transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])

    for i in range(1,10):
        k = random.randint(0,3)
        img_a = torch.rot90(img, k, dims=(-2, -1))
        msk_a = tv_tensors.Mask(torch.rot90(msk, k, dims=(-2, -1)))
        img_a, msk_a = transforms(img_a, msk_a)
        print_img(img_a, msk_a, f"aug_{i}")