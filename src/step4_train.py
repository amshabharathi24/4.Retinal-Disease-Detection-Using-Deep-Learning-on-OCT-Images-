import os
import time
import json
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from src.preprocessing import get_dataloaders
from src.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


def save_training_curves(history, output_path: str = "models/training_history.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], marker="o", label="Train Macro-F1")
    axes[1].plot(epochs, history["val_f1"], marker="s", label="Val Macro-F1")
    axes[1].set_title("Macro-F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Train Retinal OCT Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["custom_cnn", "mobilenet_v2", "resnet18", "vgg16", "densenet121"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience in epochs")
    parser.add_argument("--quick_debug", action="store_true", help="Use 2000 train / 500 val samples for fast CPU verification")
    parser.add_argument("--unfreeze", action="store_true", help="Fine-tune entire backbone instead of head-only")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining '{args.model}' on {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | Patience: {args.patience}")

    train_loader, val_loader, _, class_weights, class_names = get_dataloaders(
        batch_size=args.batch_size,
        quick_debug=args.quick_debug,
    )

    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=not args.unfreeze,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join("models", f"best_{args.model}.pth")

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_f1)
        elapsed = time.time() - start_t

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        print(
            f"Epoch [{epoch}/{args.epochs}] ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, F1: {tr_f1:.4f} | "
            f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_macro_f1": round(va_f1, 4),
                    "val_accuracy": round(va_acc, 4),
                },
                checkpoint_path,
            )
            print(f"  --> New best checkpoint saved to '{checkpoint_path}' (Val Macro-F1: {best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  --> Early stopping triggered after {epoch} epochs (no Val Macro-F1 improvement for {args.patience} epochs).")
                break

    with open(os.path.join("models", f"history_{args.model}.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_training_curves(history, output_path=os.path.join("models", f"curves_{args.model}.png"))
    print(f"\nTraining complete! Best Validation Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()