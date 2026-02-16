import argparse
import os

import nibabel as nib
import torch

from data_utils import load_dataset_json, load_nifti, normalize_ct, sliding_window_inference
from unet3d import UNet3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--checkpoint", default="outputs/checkpoint_best.pt")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out_dir", default="outputs/preds")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=1, out_channels=2, base_channels=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    data = load_dataset_json(args.data_dir)
    for image_rel in data["test"]:
        image_path = os.path.join(args.data_dir, image_rel)
        image, affine, header = load_nifti(image_path)
        image = normalize_ct(image)
        pred = sliding_window_inference(
            image, model, tuple(args.patch_size), args.overlap, device
        )

        base = os.path.basename(image_rel).replace(".nii.gz", "")
        out_path = os.path.join(args.out_dir, f"{base}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred.astype("uint8"), affine, header), out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
