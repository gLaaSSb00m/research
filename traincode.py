import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from vit_unet_torch import ViTUNet
import torch.nn as nn
import torch.optim as optim
import librosa
import matplotlib.pyplot as plt

# Load preprocessed spectrogram data
noisy_mag = np.load('./spectrogram_data/noisy_mag.npy')
voice_mag = np.load('./spectrogram_data/voice_mag.npy')

# Normalize the spectrograms
noisy_mag = noisy_mag / np.max(noisy_mag)
voice_mag = voice_mag / np.max(voice_mag)

# Convert to PyTorch tensors
noisy_mag_tensor = torch.tensor(noisy_mag, dtype=torch.float32)
voice_mag_tensor = torch.tensor(voice_mag, dtype=torch.float32)

# Add channel dimension and fix shape
noisy_mag_tensor = noisy_mag_tensor.unsqueeze(1).squeeze(-1)  # Shape: [batch_size, 1, height, width]
voice_mag_tensor = voice_mag_tensor.unsqueeze(1).squeeze(-1)

print(noisy_mag_tensor.shape)  # Should be [batch_size, 1, height, width]

class SpectrogramDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# Create dataset and dataloader
dataset = SpectrogramDataset(noisy_mag_tensor, voice_mag_tensor)

# Split dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.2 * len(dataset))    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Define batch size
batch_size = 8

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = ViTUNet().to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
epochs = 50  # Number of epochs
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (noisy, clean) in enumerate(train_loader):
        # Move data to the same device as the model
        noisy, clean = noisy.to(device), clean.to(device)

        # Forward pass
        output = model(noisy)

        # Compute loss
        loss = criterion(output, clean)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_train_loss += loss.item()

        # Calculate accuracy (threshold-based for regression tasks)
        predicted = output.detach()
        correct_train += torch.sum(torch.abs(predicted - clean) < 0.1).item()
        total_train += clean.numel()

        # Show progress percentage
        progress = (batch_idx + 1) / len(train_loader) * 100
        print(f"\rEpoch {epoch + 1}/{epochs} - {progress:.2f}% complete", end="")

    # Calculate average training loss and accuracy
    train_loss = epoch_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Forward pass
            output = model(noisy)

            # Compute loss
            loss = criterion(output, clean)
            epoch_val_loss += loss.item()

            # Calculate accuracy (threshold-based for regression tasks)
            predicted = output.detach()
            correct_val += torch.sum(torch.abs(predicted - clean) < 0.1).item()
            total_val += clean.numel()

    # Calculate average validation loss and accuracy
    val_loss = epoch_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch metrics
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
    print(f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}")

# Save the trained model
torch.save(model.state_dict(), "vit_unet_speech_separation.pth")

# Plot loss and accuracy curves
plt.figure(figsize=(12, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# Save the plot
plt.savefig('training_curves.png')
plt.show()

# Inference phase
# Prepare test data (example: loading a test spectrogram)
test_noisy_mag = np.load('./spectrogram_data/test_noisy_mag.npy')
test_noisy_mag = test_noisy_mag / np.max(test_noisy_mag)  # Normalize
test_noisy_mag_tensor = torch.tensor(test_noisy_mag, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension and move to device

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    enhanced_mag = model(test_noisy_mag_tensor)

# Convert the output back to numpy
enhanced_mag = enhanced_mag.squeeze(1).cpu().numpy()

# Load the phase spectrogram
noisy_phase = np.load('spectrogram_data/noisy_phase.npy')

# Reconstruct the complex spectrogram
enhanced_complex = enhanced_mag * np.exp(1j * noisy_phase)

# Perform inverse STFT
enhanced_audio = librosa.istft(enhanced_complex)

# Save the enhanced audio
sf.write('enhanced_audio.wav', enhanced_audio, samplerate=8000)