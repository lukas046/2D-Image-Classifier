import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # depends on input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,16,H/2,W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B,32,H/4,W/4)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def load_model(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "weights.pth")
    model = SimpleCNN(num_classes=22)
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


transform = T.Compose([T.ToTensor(), T.Resize((224, 224))])


def guess(model, image: torch.Tensor):
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, class_label = torch.max(probs, dim=1)

    return class_label.item(), confidence.item()
