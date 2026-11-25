import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.preprocessing import get_transforms
from src.model import build_model


class GradCAM:
    """Computes Grad-CAM heatmaps for MobileNetV2, ResNet18, or DenseNet121."""

    def __init__(self, model, model_name: str):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Disable inplace=True on ReLU/ReLU6 layers so backward hooks never fail on MobileNetV2
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                module.inplace = False

        name = model_name.lower()
        if name in ("mobilenet_v2", "custom_cnn"):
            target_layer = self.model.features[-1]
        elif name == "resnet18":
            target_layer = self.model.layer4[-1]
        elif name == "densenet121":
            target_layer = self.model.features.denseblock4
        else:
            raise ValueError(f"Grad-CAM not configured for '{model_name}'")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """Generates a normalized [0, 1] heatmap (H, W) for the target class."""
        for param in self.model.parameters():
            param.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, probs, class_idx


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blends a jet colormap Grad-CAM heatmap onto a resized RGB OCT image."""
    base_img = pil_image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    base_arr = np.array(base_img, dtype=np.float32) / 255.0

    colored_map = plt.get_cmap("jet")(heatmap)[:, :, :3]
    blended = np.clip((1 - alpha) * base_arr + alpha * colored_map, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("models/best_mobilenet_v2.pth", map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    class_names = ckpt["class_names"]

    model = build_model(model_name=model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    from datasets import load_dataset
    ds = load_dataset("zacharielegault/Kermany2017-OCT", split="test")
    sample_img = ds[0]["image"].convert("RGB")
    actual_label = class_names[ds[0]["label"]]

    tfms = get_transforms(img_size=224)["eval"]
    tensor_in = tfms(sample_img).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model_name)
    heatmap, probs, pred_idx = gradcam.generate(tensor_in)
    overlay_img = overlay_heatmap(sample_img, heatmap)

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "gradcam_sample.png")
    overlay_img.save(out_path)

    print(f"Actual Class    : {actual_label}")
    print(f"Predicted Class : {class_names[pred_idx]} ({probs[pred_idx]*100:.2f}% confidence)")
    print(f"Saved Grad-CAM overlay preview to '{out_path}'")