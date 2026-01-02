"""Test single image"""

from main import process_image, load_image, load_mask, load_ground_truth
import os

DRIVE_PATH = "DRIVE"

img = "DRIVE/training/images/21_training.tif"
mask = "DRIVE/training/mask/21_training_mask.gif"
gt = "DRIVE/training/1st_manual/21_manual1.gif"

if all(os.path.exists(p) for p in [img, mask, gt]):
    m, _, _ = process_image(img, mask, gt, save_dir="DRIVE/output/test")
    print(f"Sensitivity: {m['Sensitivity']:.4f}")
    print(f"Specificity: {m['Specificity']:.4f}")
    print(f"Accuracy:    {m['Accuracy']:.4f}")
    print(f"PPV:         {m['PPV']:.4f}")
else:
    print("Files not found!")
