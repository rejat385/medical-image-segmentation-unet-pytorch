
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
import wandb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

config = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "image_size": 128,
    "optimizer": "Adam",
    "loss": "BCE + Dice",
}


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

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.last_bce = None
        self.last_dice = None

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice = dice_loss(logits, targets)

        self.last_bce = bce_loss.detach()
        self.last_dice = dice.detach()

        return bce_loss + dice
    
def main():
    # tell wandb to get started
    with wandb.init(project="medical-image-segmentation-unet-pytorch", name="unet-baseline", config=config):      

        dataset = PolypDataset(size=128)

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
            CHECKPOINT_DIR / "split_indices.pt"
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

        criterion = BCEDiceLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )

        history = {"train_loss": [], "val_dice": []}
        best_dice = -1.0

        print("Device:", DEVICE)
        print("Training images:", len(train_set))
        print("Validation images:", len(val_set))
        print("Test images:", len(test_set))

        wandb.watch(model, criterion, log="all", log_freq=10)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_dice = 0.0

            for images, masks in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS}"
            ):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                optimizer.zero_grad()

                logits = model(images)

                loss = criterion(logits, masks)

                bce_loss = criterion.last_bce
                dice_loss_val = criterion.last_dice

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_bce += bce_loss.item() * images.size(0)
                total_dice += dice_loss_val.item() * images.size(0)

            train_loss = total_loss / len(train_set)
            bce_avg = total_bce / len(train_set)
            dice_avg = total_dice / len(train_set)
        
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
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    CHECKPOINT_DIR / "unet_best.pth"
                )

            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": train_loss,
                "train/bce_loss": bce_avg,
                "train/dice_loss": dice_avg,
                "val/dice_score": val_dice,
                "val/best_dice": best_dice
            })

        print(f"\nBest validation Dice: {best_dice:.4f}")

        wandb.run.summary["best_val_dice"] = best_dice
        wandb.run.summary["best_epoch"] = best_epoch

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(history["train_loss"])
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")

        axes[1].plot(history["val_dice"])
        axes[1].set_title("Validation Dice")
        axes[1].set_xlabel("Epoch")

        plt.tight_layout()
        plot_path = RESULTS_DIR / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        wandb.log({
            "training_curves": wandb.Image(str(plot_path))
        })
        plt.show()


if __name__ == "__main__":
    main()
