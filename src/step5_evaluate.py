import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.preprocessing import get_dataloaders
from src.model import build_model


@torch.no_grad()
def run_evaluation(checkpoint_path: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    print(f"Loaded '{checkpoint_path}' | Backbone: {model_name} | Best Val F1: {checkpoint['val_macro_f1']}")

    _, _, test_loader, _, _ = get_dataloaders(batch_size=batch_size, quick_debug=False)

    model = build_model(model_name=model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_targets.extend(labels.tolist())

    test_acc = accuracy_score(all_targets, all_preds)
    test_macro_f1 = f1_score(all_targets, all_preds, average="macro")
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)

    print("\n--- Official 1,000-Image Test Set Evaluation ---")
    print(f"Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Test Macro-F1 : {test_macro_f1:.4f}\n")
    print(report_str)

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Test Confusion Matrix ({model_name} | Acc: {test_acc*100:.1f}%)")
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")
    plt.tight_layout()

    cm_path = os.path.join("models", f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to '{cm_path}'")

    metrics_path = os.path.join("models", f"eval_{model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "test_accuracy": round(test_acc, 4),
                "test_macro_f1": round(test_macro_f1, 4),
                "classification_report": report_dict,
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print(f"Saved evaluation metrics to '{metrics_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained OCT model on Test split")
    parser.add_argument("--checkpoint", type=str, default="models/best_mobilenet_v2.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_evaluation(checkpoint_path=args.checkpoint, batch_size=args.batch_size)