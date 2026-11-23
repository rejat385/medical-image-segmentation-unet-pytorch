import random

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class PolypDataset(Dataset):
    def __init__(self, root="data/Kvasir-SEG", size=128, augment=False):
        self.root = Path(root)
        self.size = size
        self.augment = augment

        image_dir = self.root / "images"
        mask_dir = self.root / "masks"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Check dataset folders inside: {self.root}"
            )

        self.pairs = []

        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for: {image_path.name}"
                )

            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError("No image/mask pairs found!")

        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        with Image.open(image_path) as im:
            image = im.convert("RGB").resize(
                (self.size, self.size),
                Image.Resampling.BILINEAR
            )

        with Image.open(mask_path) as m:
            mask = m.convert("L").resize(
                (self.size, self.size),
                Image.Resampling.NEAREST
            )

        image = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        mask = np.asarray(
            mask, dtype=np.float32
        ) / 255.0

        # JPEG masks may have compression artifacts.
        mask = (mask >= 0.5).astype(np.float32)

        # Image: HWC -> CHW
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        # Mask: HW -> 1HW
        mask = torch.from_numpy(
            mask[None, ...].copy()
        )

        if self.augment:
            k = random.randint(0,3)
            image = torch.rot90(image, k, dims=(-2, -1))
            mask = tv_tensors.Mask(torch.rot90(mask, k, dims=(-2, -1)))
            image, mask = self.transforms(image, mask)


        return image, mask


if __name__ == "__main__":
    dataset = PolypDataset(augment=True)

    image, mask = dataset[0]

    assert image.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    assert image.dtype == torch.float32
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    print("Number of samples:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(),
          image.max().item())
    print("Mask values:", torch.unique(mask))

    val_dataset = PolypDataset(augment=False)

    image1, mask1 = val_dataset[0]
    image2, mask2 = val_dataset[0]

    assert torch.equal(image1, image2)
    assert torch.equal(mask1, mask2)

    print("Validation dataset is deterministic.")
