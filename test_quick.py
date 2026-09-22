"""
Quick test to verify MNIST recognition works without interactive mode
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import os

# Model architecture
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)

# Test function
def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'mnist_model.pth'

    if not os.path.exists(model_path):
        print("모델이 아직 훈련 중입니다. 잠시 후 다시 시도해주세요.")
        return

    # Load model
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test on sample from test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("테스트 샘플 5개로 모델 검증:")
    print("-" * 50)

    correct = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        if i >= 5:
            break

        image = image.to(device)
        output = model(image)
        prob = torch.softmax(output, dim=1)
        pred = prob.argmax(item=1).item()
        conf = prob[0][pred].item()

        is_correct = "✓" if pred == label.item() else "✗"
        print(f"샘플 {i+1}: 실제={label.item()}, 예측={pred}, 확률={conf*100:.2f}% {is_correct}")

        total += 1
        if pred == label.item():
            correct += 1

    print("-" * 50)
    print(f"정확도: {correct}/{total} ({correct*100//total}%)")
    print("\n✓ 모델 훈련이 완료되었습니다!")
    print("이제 'python mnist_handwriting_recognition.py'를 실행하여")
    print("손글씨 인식 모드를 시작하세요.")

if __name__ == "__main__":
    test_model()
