import json
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset_json(data_dir):
    json_path = os.path.join(data_dir, "dataset.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_train_val(items, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img.affine, img.header


def normalize_ct(volume, clip_min=-100.0, clip_max=300.0):
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)
    return volume


def pad_to_shape(arr, target_shape):
    pad = []
    for dim, target in zip(arr.shape, target_shape):
        total = max(target - dim, 0)
        before = total // 2
        after = total - before
        pad.append((before, after))
    return np.pad(arr, pad, mode="constant", constant_values=0)


def random_crop_3d(image, label, patch_size):
    image = pad_to_shape(image, patch_size)
    label = pad_to_shape(label, patch_size)
    z, y, x = image.shape
    pz, py, px = patch_size
    sz = random.randint(0, z - pz)
    sy = random.randint(0, y - py)
    sx = random.randint(0, x - px)
    image = image[sz : sz + pz, sy : sy + py, sx : sx + px]
    label = label[sz : sz + pz, sy : sy + py, sx : sx + px]
    return image, label


def center_crop_3d(image, patch_size):
    image = pad_to_shape(image, patch_size)
    z, y, x = image.shape
    pz, py, px = patch_size
    sz = (z - pz) // 2
    sy = (y - py) // 2
    sx = (x - px) // 2
    return image[sz : sz + pz, sy : sy + py, sx : sx + px]


class NiftiSegmentationDataset(Dataset):
    def __init__(self, items, data_dir, patch_size, training=True):
        self.items = items
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.training = training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = os.path.join(self.data_dir, item["image"])
        label_path = os.path.join(self.data_dir, item["label"])
        image, _, _ = load_nifti(image_path)
        label, _, _ = load_nifti(label_path)
        image = normalize_ct(image)
        if self.training:
            image, label = random_crop_3d(image, label, self.patch_size)
            if random.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                label = np.flip(label, axis=0).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=2).copy()
        else:
            image = center_crop_3d(image, self.patch_size)
            label = center_crop_3d(label, self.patch_size)
        image = torch.from_numpy(image[None, ...])
        label = torch.from_numpy(label.astype(np.int64))
        return image, label


def compute_dice(pred, target, eps=1e-6):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    inter = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * inter + eps) / (union + eps)


def sliding_window_inference(volume, model, patch_size, overlap, device):
    model.eval()
    volume = pad_to_shape(volume, patch_size)
    z, y, x = volume.shape
    pz, py, px = patch_size
    sz = max(int(pz * (1 - overlap)), 1)
    sy = max(int(py * (1 - overlap)), 1)
    sx = max(int(px * (1 - overlap)), 1)

    out_prob = np.zeros((2, z, y, x), dtype=np.float32)
    count = np.zeros((z, y, x), dtype=np.float32)

    with torch.no_grad():
        for zz in range(0, z - pz + 1, sz):
            for yy in range(0, y - py + 1, sy):
                for xx in range(0, x - px + 1, sx):
                    patch = volume[zz : zz + pz, yy : yy + py, xx : xx + px]
                    patch_t = torch.from_numpy(patch[None, None, ...]).to(device)
                    logits = model(patch_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    out_prob[:, zz : zz + pz, yy : yy + py, xx : xx + px] += probs
                    count[zz : zz + pz, yy : yy + py, xx : xx + px] += 1.0

    out_prob = out_prob / np.maximum(count[None, ...], 1.0)
    pred = np.argmax(out_prob, axis=0).astype(np.uint8)
    return pred
