import torch
import torch.nn as nn
from torchvision import models


class CustomRetinalCNN(nn.Module):
    """
    Custom CNN built from scratch for 4-class OCT retinal disease classification
    (Document Step 4 & Step 6: Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout).
    """

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    """
    Builds either CustomRetinalCNN (from scratch) or a pretrained ImageNet backbone
    (MobileNetV2, ResNet18, VGG16, DenseNet121) adapted for 4-class OCT classification.
    """
    name = model_name.lower()

    if name == "custom_cnn":
        return CustomRetinalCNN(num_classes=num_classes, dropout_rate=0.35)

    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: custom_cnn, mobilenet_v2, resnet18, vgg16, densenet121."
        )

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_batch = torch.randn(4, 3, 224, 224).to(device)

    for arch in ["custom_cnn", "mobilenet_v2", "resnet18", "vgg16", "densenet121"]:
        net = build_model(arch, num_classes=4, pretrained=True, freeze_backbone=True).to(device)
        out = net(dummy_batch)
        total_params = sum(p.numel() for p in net.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(
            f"{arch:<13} | Output: {list(out.shape)} | "
            f"Total Params: {total_params:.2f}M | Trainable Params: {trainable_params:,}"
        )