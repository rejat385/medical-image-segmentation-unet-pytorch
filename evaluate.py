
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from dataset import PolypDataset
from models.unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_metrics(pred, target, eps=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_area = pred.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3))

    dice = (
        (2 * intersection + eps)
        / (pred_area + target_area + eps)
    )

    union = pred_area + target_area - intersection
    iou = (intersection + eps) / (union + eps)

    return dice, iou


def main():
    dataset = PolypDataset(size=128)

    splits = torch.load(
        "checkpoints/split_indices.pt",
        weights_only=True
    )
    test_set = Subset(dataset, splits["test"])

    loader = DataLoader(
        test_set, batch_size=8, shuffle=False
    )

    model = UNet().to(DEVICE)
    model.load_state_dict(
        torch.load(
            "checkpoints/unet_best.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )
    model.eval()

    all_dice = []
    all_iou = []

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(DEVICE))
            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            dice, iou = calculate_metrics(
                predictions,
                masks.to(DEVICE)
            )

            all_dice.extend(dice.cpu().tolist())
            all_iou.extend(iou.cpu().tolist())

    print(f"Test Dice: {np.mean(all_dice):.4f}")
    print(f"Test IoU:  {np.mean(all_iou):.4f}")

    # Visualize one held-out image
    image, mask = test_set[0]

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(DEVICE))
        pred = (
            torch.sigmoid(logits)[0, 0].cpu().numpy()
            >= 0.5
        ).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Medical Image")

    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/prediction_example.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
