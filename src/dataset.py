from collections import Counter
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def load_and_inspect_oct2017(dataset_name: str = "zacharielegault/Kermany2017-OCT"):
    """Loads OCT2017 directly from Hugging Face and audits classes and image formats."""
    print(f"Downloading/loading '{dataset_name}' from Hugging Face...")
    dataset = load_dataset(dataset_name)

    label_names = dataset["train"].features["label"].names
    print(f"Splits available: {list(dataset.keys())}")
    print(f"Class mapping: {dict(enumerate(label_names))}")

    # 1. Count images per disease category
    train_counts = Counter(dataset["train"]["label"])
    test_counts = Counter(dataset["test"]["label"])

    print("\n--- Class Distribution ---")
    for idx, name in enumerate(label_names):
        print(f"{name:<8} | Train: {train_counts[idx]:<6} | Test: {test_counts[idx]:<5}")

    # 2. Check image dimensions and color channels (sample first 500 scans)
    modes = Counter()
    sizes = Counter()
    for i in range(500):
        img = dataset["train"][i]["image"]
        modes[img.mode] += 1
        sizes[img.size] += 1

    print(f"\nSampled Color Modes: {dict(modes)}")
    print(f"Top 5 Image Resolutions (W, H): {sizes.most_common(5)}")

    return dataset, label_names


def create_leakage_free_splits(dataset, n_splits: int = 5, seed: int = 42):
    """
    Splits the training data 80/20 into Train and Validation sets using a fixed seed.
    Keeps the official Test split completely untouched until final evaluation.
    """
    train_labels = dataset["train"]["label"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(range(len(train_labels)), train_labels))

    train_split = dataset["train"].select(train_idx)
    val_split = dataset["train"].select(val_idx)
    test_split = dataset["test"]

    print(
        f"\nLeakage-Free Splits -> "
        f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}"
    )
    return train_split, val_split, test_split


def save_class_samples(train_split, label_names, output_path: str = "oct_samples.png"):
    """Saves a 1x4 visual grid with one sample OCT image from each category."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    samples = {}

    for item in train_split:
        lbl = item["label"]
        if lbl not in samples:
            samples[lbl] = item["image"]
        if len(samples) == 4:
            break

    for idx in range(4):
        axes[idx].imshow(samples[idx], cmap="gray")
        axes[idx].set_title(f"{label_names[idx]}\nSize: {samples[idx].size}, Mode: {samples[idx].mode}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved sample grid to '{output_path}'")


if __name__ == "__main__":
    raw_dataset, classes = load_and_inspect_oct2017()
    train_ds, val_ds, test_ds = create_leakage_free_splits(raw_dataset)
    save_class_samples(train_ds, classes)