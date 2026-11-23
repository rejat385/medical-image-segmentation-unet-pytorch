
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import PolypDataset
from models.unet import UNet

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    total = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + smooth) / (total + smooth)
    return 1 - dice.mean()


def dice_score(logits, targets, smooth=1e-6):
    preds = (torch.sigmoid(logits) >= 0.5).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    total = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + smooth) / (total + smooth)
    return dice


def main():
    dataset = PolypDataset(size=128, augment=False)

    train_set, val_set, test_set = random_split(
        dataset,
        [800, 100, 100],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Save split indices for reproducible test evaluation.
    torch.save(
        {
            "train": train_set.indices,
            "val": val_set.indices,
            "test": test_set.indices
        },
        "checkpoints/split_indices.pt"
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )

    model = UNet().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    history = {"train_loss": [], "val_dice": []}
    best_dice = -1.0

    print("Device:", DEVICE)
    print("Training images:", len(train_set))
    print("Validation images:", len(val_set))
    print("Test images:", len(test_set))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, masks in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = (
                bce(logits, masks)
                + dice_loss(logits, masks)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_set)

        model.eval()
        dice_values = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)
                dice_values.extend(
                    dice_score(logits, masks).cpu().tolist()
                )

        val_dice = float(np.mean(dice_values))

        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                "checkpoints/unet_best.pth"
            )

    print(f"\nBest validation Dice: {best_dice:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_dice"])
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
