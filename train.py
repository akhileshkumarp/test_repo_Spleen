import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    NiftiSegmentationDataset,
    compute_dice,
    load_dataset_json,
    load_nifti,
    normalize_ct,
    sliding_window_inference,
    split_train_val,
)
from unet3d import UNet3D


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    targets_onehot = torch.zeros_like(probs)
    targets_onehot.scatter_(1, targets[:, None, ...], 1)
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * targets_onehot, dims)
    union = torch.sum(probs + targets_onehot, dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def validate(model, val_items, data_dir, patch_size, overlap, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for item in val_items:
            image_path = os.path.join(data_dir, item["image"])
            label_path = os.path.join(data_dir, item["label"])
            image, _, _ = load_nifti(image_path)
            label, _, _ = load_nifti(label_path)
            image = normalize_ct(image)
            pred = sliding_window_inference(image, model, patch_size, overlap, device)
            dices.append(compute_dice(pred > 0, label > 0))
    return float(np.mean(dices)) if dices else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset_json(args.data_dir)
    train_items, val_items = split_train_val(data["training"], args.val_ratio, args.seed)

    train_dataset = NiftiSegmentationDataset(
        train_items, args.data_dir, tuple(args.patch_size), training=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet3D(in_channels=1, out_channels=2, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = ce_loss(logits, labels) + dice_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_dice = validate(
            model, val_items, args.data_dir, tuple(args.patch_size), args.overlap, device
        )
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_dice={val_dice:.4f}")

        checkpoint_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_path)
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.out_dir, "checkpoint_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)


if __name__ == "__main__":
    main()
