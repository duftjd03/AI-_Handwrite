"""
MNIST Handwriting Recognition System (PyTorch Version)
Trains a neural network on MNIST dataset and provides real-time handwriting recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import Image, ImageDraw


# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(batch_size=128):
    """
    Load MNIST dataset and preprocess it for training.

    Returns:
        tuple: (train_loader, test_loader, device) - data loaders and device
    """
    print("Loading MNIST dataset...")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Test data: {len(test_dataset)} samples")

    return train_loader, test_loader, device


# ============================================================================
# PART 2: MODEL ARCHITECTURE
# ============================================================================

class MNISTNet(nn.Module):
    """Neural network for digit classification."""

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

        x = self.fc4(x)
        return x


def build_model(device):
    """
    Build and compile the neural network model.

    Returns:
        nn.Module: PyTorch model
    """
    model = MNISTNet().to(device)
    return model


# ============================================================================
# PART 3: MODEL TRAINING
# ============================================================================

def train_model(model, train_loader, test_loader, device, epochs=15, learning_rate=0.001):
    """
    Train the neural network model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Torch device (cuda or cpu)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        dict: Training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("\nTraining model...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


# ============================================================================
# PART 4: MODEL EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test data.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Torch device
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.

    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    axes[0].plot(history['train_acc'], label='Training Accuracy')
    axes[0].plot(history['val_acc'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(history['train_loss'], label='Training Loss')
    axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("\nTraining history plot saved as 'training_history.png'")
    plt.close()


# ============================================================================
# PART 5: HANDWRITING INPUT AND RECOGNITION
# ============================================================================

def create_drawing_canvas():
    """
    Create an interactive canvas for drawing handwritten digits.
    User can draw with mouse and press SPACE to recognize or Q to quit.

    Returns:
        numpy.ndarray: 280x280 image array
    """
    image_size = 280
    image = Image.new('L', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)

    print("\n" + "="*60)
    print("HANDWRITING RECOGNITION MODE")
    print("="*60)
    print("Instructions:")
    print("  - Left-click and drag to draw a digit")
    print("  - Press 'SPACE' to recognize the digit")
    print("  - Press 'C' to clear the canvas")
    print("  - Press 'Q' to quit")
    print("="*60 + "\n")

    canvas = np.array(image)

    cv2.namedWindow('Draw Digit', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Draw Digit', 560, 560)

    drawing = False
    last_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas

        x = x // 2
        y = y // 2

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if last_point:
                draw.line([last_point, (x, y)], fill='black', width=10)
                last_point = (x, y)
            canvas = np.array(image)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    cv2.setMouseCallback('Draw Digit', mouse_callback)

    while True:
        display_canvas = cv2.resize(canvas, (560, 560), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Draw Digit', display_canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            cv2.destroyAllWindows()
            return canvas

        elif key == ord('c'):
            image = Image.new('L', (image_size, image_size), 'white')
            draw = ImageDraw.Draw(image)
            canvas = np.array(image)
            print("Canvas cleared!")

        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return canvas


def preprocess_user_input(image_array):
    """
    Preprocess user-drawn image for model prediction.

    Args:
        image_array: numpy array (typically 280x280)

    Returns:
        tuple: (preprocessed_tensor, normalized_image)
    """
    # Resize to 28x28
    if image_array.shape != (28, 28):
        image_resized = cv2.resize(image_array, (28, 28))
    else:
        image_resized = image_array

    # Invert colors
    image_inverted = 255 - image_resized

    # Normalize to [0, 1]
    image_normalized = image_inverted.astype('float32') / 255.0

    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)

    return image_tensor, image_normalized


def predict_digit(model, image_array, device):
    """
    Predict the digit in the given image.

    Args:
        model: Trained PyTorch model
        image_array: 280x280 numpy array
        device: Torch device

    Returns:
        tuple: (predicted_digit, confidence, normalized_image)
    """
    image_tensor, image_normalized = preprocess_user_input(image_array)
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_digit = torch.max(probabilities, 1)

    return predicted_digit.item(), confidence.item(), image_normalized


def display_prediction(predicted_digit, confidence, image_normalized):
    """
    Display the recognized digit and confidence score.

    Args:
        predicted_digit: Predicted digit (0-9)
        confidence: Confidence score (0-1)
        image_normalized: Normalized image array
    """
    print("\n" + "="*60)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("="*60 + "\n")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image_normalized, cmap='gray')
    ax.set_title(f"Recognized Digit: {predicted_digit} (Confidence: {confidence*100:.2f}%)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=100)
    print("Prediction result saved as 'prediction_result.png'")
    plt.close()


def interactive_recognition(model, device):
    """
    Main function for interactive handwriting recognition.

    Args:
        model: Trained PyTorch model
        device: Torch device
    """
    while True:
        canvas = create_drawing_canvas()

        if canvas is None:
            print("\nExiting recognition mode...")
            break

        predicted_digit, confidence, image_normalized = predict_digit(model, canvas, device)
        display_prediction(predicted_digit, confidence, image_normalized)

        continue_choice = input("Draw another digit? (yes/no): ").lower().strip()
        if continue_choice not in ['yes', 'y']:
            print("\nExiting recognition mode...")
            break


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """Main function to orchestrate the entire pipeline."""
    model_path = 'mnist_model.pth'

    # Check if trained model exists
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from '{model_path}'...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Load data
        train_loader, test_loader, device = load_and_preprocess_data()

        # Build model
        print("\nBuilding model architecture...")
        model = build_model(device)
        print(model)

        # Train model
        history = train_model(model, train_loader, test_loader, device, epochs=15)

        # Evaluate model
        evaluate_model(model, test_loader, device)

        # Plot training history
        plot_training_history(history)

        # Save model
        print(f"\nSaving model to '{model_path}'...")
        torch.save(model.state_dict(), model_path)
        print("Model saved successfully!")

    # Start interactive recognition
    print("\n" + "="*60)
    print("MNIST HANDWRITING RECOGNITION SYSTEM READY")
    print("="*60)
    interactive_recognition(model, device)

    print("\nThank you for using MNIST Handwriting Recognition System!")


if __name__ == "__main__":
    main()
