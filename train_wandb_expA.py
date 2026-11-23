
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import PolypDataset
from models.unet import UNet
from train import dice_loss, dice_score
from train_wandb import BCEDiceLoss
import wandb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 50
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
    "LR_Scheduler": "ReduceLROnPlateau"
}

    
def main():
    # tell wandb to get started
    with wandb.init(project="medical-image-segmentation-unet-pytorch", name="unet-expA", config=config):      

        dataset = PolypDataset(size=128, augment=False)

        splits = torch.load(
            "checkpoints/split_indices.pt",
            weights_only=True
        )

        train_set = Subset(dataset, splits["train"])
        val_set = Subset(dataset, splits["val"])

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

        # 4. Adaptive learning-rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )

        history = {"train_loss": [], "val_dice": [], "learning_rate": []}
        best_dice = -1.0
        best_epoch = 0

        print("Device:", DEVICE)
        print("Training images:", len(train_set))
        print("Validation images:", len(val_set))

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

            # Save best model BEFORE scheduler update

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    CHECKPOINT_DIR / "unet_expA_best.pth"
                )

            # Adjust learning rate based on validation
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]["lr"]
            
            history["train_loss"].append(train_loss)
            history["val_dice"].append(val_dice)
            history["learning_rate"].append(current_lr)

            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": train_loss,
                "train/bce_loss": bce_avg,
                "train/dice_loss": dice_avg,
                "val/dice_score": val_dice,
                "val/best_dice": best_dice,
                "lr": current_lr
            })

            print(
                f"Epoch {epoch+1:02d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        artifact = wandb.Artifact(
            name="unet-best-model-expA",
            type="model",
            metadata={
                "best_val_dice": best_dice,
                "best_epoch": best_epoch,
                "architecture": "U-Net",
                "dataset": "Kvasir-SEG"
            }
        )

        artifact.add_file(
            str(CHECKPOINT_DIR / "unet_expA_best.pth")
        )
        artifact.add_file(
            str(CHECKPOINT_DIR / "split_indices.pt")
        )

        print("\nExperiment A finished")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation Dice: {best_dice:.4f}")

        # 7. Save results
        np.savez(
            RESULTS_DIR / "experiment_A_history.npz",
            train_loss=history["train_loss"],
            val_dice=history["val_dice"],
            learning_rate=history["learning_rate"]
        )
        artifact.add_file(str(RESULTS_DIR / "experiment_A_history.npz"))
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_dice"] = best_dice
        wandb.run.summary["best_epoch"] = best_epoch

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["train_loss"])
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")

        axes[1].plot(history["val_dice"])
        axes[1].set_title("Validation Dice")
        axes[1].set_xlabel("Epoch")

        axes[2].plot(history["learning_rate"])
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / "training_curves_expA.png"
        plt.savefig(plot_path, dpi=150)
        wandb.log({
            "training_curves": wandb.Image(str(plot_path))
        })
        plt.show()


if __name__ == "__main__":
    main()
