from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from src.dataset import load_and_inspect_oct2017, create_leakage_free_splits

# Standard ImageNet statistics required by pretrained torchvision backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """
    Returns domain-safe transforms for OCT retinal scans:
    - Train: Resize + HorizontalFlip + Small Rotation (±10°) + Mild Brightness/Contrast Jitter
      (No vertical flip, preserving natural retinal layer anatomy)
    - Eval (Val/Test): Deterministic Resize + ImageNet Normalization
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": train_transform, "eval": eval_transform}


class OCTDataset(Dataset):
    """PyTorch Dataset wrapper that converts grayscale OCT images to 3-channel RGB."""

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        image = item["image"].convert("RGB")
        label = int(item["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_class_weights(train_split, num_classes: int = 4) -> torch.Tensor:
    """Computes inverse-frequency class weights to counteract class imbalance."""
    counts = Counter(train_split["label"])
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    quick_debug: bool = False,
):
    """
    Builds Train, Validation, and Test PyTorch DataLoaders plus class weights.
    Set quick_debug=True to use a small subset (2000 train / 500 val) for fast CPU checks.
    """
    raw_dataset, label_names = load_and_inspect_oct2017()
    train_split, val_split, test_split = create_leakage_free_splits(raw_dataset)

    if quick_debug:
        train_split = train_split.shuffle(seed=42).select(range(2000))
        val_split = val_split.shuffle(seed=42).select(range(500))
        print(f"[Quick Debug Mode] Using {len(train_split)} train and {len(val_split)} val samples.")

    class_weights = compute_class_weights(train_split, num_classes=len(label_names))
    tfms = get_transforms(img_size=img_size)

    train_ds = OCTDataset(train_split, transform=tfms["train"])
    val_ds = OCTDataset(val_split, transform=tfms["eval"])
    test_ds = OCTDataset(test_split, transform=tfms["eval"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights, label_names


def save_preprocessed_preview(train_loader, label_names, output_path: str = "preprocessed_batch.png"):
    """Denormalizes one batch and saves an 8-image grid to verify preprocessing."""
    images, labels = next(iter(train_loader))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip((img * std) + mean, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Class: {label_names[labels[i].item()]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved preprocessed batch preview to '{output_path}'")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, weights, classes = get_dataloaders(batch_size=32)
    batch_imgs, batch_lbls = next(iter(train_loader))

    print(f"\nBatch Tensor Shape : {batch_imgs.shape} (Batch, Channels, Height, Width)")
    print(f"Batch Labels Shape : {batch_lbls.shape}")
    print(f"Pixel Value Range  : [{batch_imgs.min():.3f}, {batch_imgs.max():.3f}]")
    print(f"Class Weights      : {[round(w, 3) for w in weights.tolist()]}")

    save_preprocessed_preview(train_loader, classes)