import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
import librosa
import soundfile as sf
import os
import matplotlib.pyplot as plt

# Preprocessing Configuration
SAMPLE_RATE = 8000
FRAME_LENGTH = 8064
HOP_LENGTH_FRAME = 8064
N_FFT = 255
HOP_LENGTH = 63
SNR = 0
DIM_SQUARE_SPEC = int(N_FFT / 2) + 1  # 128

# Directories
VOICE_DIR = './speech/'
NOISE_DIR = './noise/'
SPECTROGRAM_DIR = './spectrogram_data/'
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs('./sound/', exist_ok=True)

# Preprocessing Functions
def audio_to_audio_frame_stack(audio, frame_length, hop_length):
    total_samples = len(audio)
    frames = []
    for start in range(0, total_samples - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)
    return np.vstack(frames)

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame):
    list_sound_array = []
    for file in list_audio_files:
        y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
        list_sound_array.append(audio_to_audio_frame_stack(y, frame_length, hop_length_frame))
    return np.vstack(list_sound_array)

def mixed_voice_with_noise(voice, noise, nb_samples, frame_length, snr):
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))
    for i in range(nb_samples):
        prod_voice[i, :] = voice[i, :]
        norm_voice = np.linalg.norm(voice[i, :])
        norm_noise = np.linalg.norm(noise[i, :])
        if norm_noise > 0 and norm_voice > 0:
            prod_noise[i, :] = (noise[i, :] / norm_noise) * (norm_voice * 10 ** (-snr / 20))
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]
    return prod_voice, prod_noise, prod_noisy_voice

def calculate_stft(audio, n_fft, hop_length_fft):
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    mag, phase = librosa.magphase(stft_audio)
    return mag, phase

def extract_stft_features(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    nb_audio = numpy_audio.shape[0]
    mag = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=np.float32)
    phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=np.complex64)
    for i in range(nb_audio):
        mag[i, :, :], phase[i, :, :] = calculate_stft(numpy_audio[i], n_fft, hop_length_fft)
    return mag, phase

# Preprocess Dataset
list_noise_files = os.listdir(NOISE_DIR)
list_voice_files = os.listdir(VOICE_DIR)
noise = audio_files_to_numpy(NOISE_DIR, list_noise_files, SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH_FRAME)
voice = audio_files_to_numpy(VOICE_DIR, list_voice_files, SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH_FRAME)

# Ensure same number of samples
l = min(voice.shape[0], noise.shape[0])
voice = voice[:l]
noise = noise[:l]
nb_samples = voice.shape[0]

# Mix voice and noise
prod_voice, prod_noise, prod_noisy = mixed_voice_with_noise(voice, noise, nb_samples, FRAME_LENGTH, SNR)
voice_reshaped = prod_voice.reshape(1, nb_samples * FRAME_LENGTH)
noise_reshaped = prod_noise.reshape(1, nb_samples * FRAME_LENGTH)
noisy_reshaped = prod_noisy.reshape(1, nb_samples * FRAME_LENGTH)

# Save audio files
sf.write('./sound/voice.wav', voice_reshaped[0, :], samplerate=SAMPLE_RATE)
sf.write('./sound/noise.wav', noise_reshaped[0, :], samplerate=SAMPLE_RATE)
sf.write('./sound/noisy.wav', noisy_reshaped[0, :], samplerate=SAMPLE_RATE)

# Extract STFT features
voice_mag, voice_phase = extract_stft_features(prod_voice, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)
noise_mag, noise_phase = extract_stft_features(prod_noise, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)
noisy_mag, noisy_phase = extract_stft_features(prod_noisy, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)

# Save spectrograms
np.save(os.path.join(SPECTROGRAM_DIR, 'voice_mag.npy'), voice_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'voice_phase.npy'), voice_phase)
np.save(os.path.join(SPECTROGRAM_DIR, 'noise_mag.npy'), noise_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'noise_phase.npy'), noise_phase)
np.save(os.path.join(SPECTROGRAM_DIR, 'noisy_mag.npy'), noisy_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'noisy_phase.npy'), noisy_phase)

# Model Definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Increased dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, C, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class ViTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(1024, 1024),
            nn.Dropout(0.4)
        )
        self.bottleneck = DoubleConv(2048, 1024)
        self.cbam = CBAM(1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        vit_input = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')
        vit_input = vit_input.repeat(1, 3, 1, 1)
        vit_features = self.vit(vit_input)
        vit_features = self.vit_proj(vit_features)
        B, C = vit_features.shape
        H, W = x5.shape[2], x5.shape[3]
        vit_features = vit_features.view(B, C, 1, 1).expand(-1, -1, H, W)
        x5 = torch.cat([x5, vit_features], dim=1)
        x5 = self.bottleneck(x5)
        x5 = self.cbam(x5)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.outc(x)

# Dataset Class
class SpectrogramDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# Load and Normalize Spectrograms
noisy_mag = np.load(os.path.join(SPECTROGRAM_DIR, 'noisy_mag.npy'))
voice_mag = np.load(os.path.join(SPECTROGRAM_DIR, 'voice_mag.npy'))
noisy_phase = np.load(os.path.join(SPECTROGRAM_DIR, 'noisy_phase.npy'))

max_mag = max(np.max(noisy_mag), np.max(voice_mag))
noisy_mag = noisy_mag / max_mag
voice_mag = voice_mag / max_mag

noisy_mag_tensor = torch.tensor(noisy_mag, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1, height, width]
voice_mag_tensor = torch.tensor(voice_mag, dtype=torch.float32).unsqueeze(1)

# Create Dataset and DataLoaders
dataset = SpectrogramDataset(noisy_mag_tensor, voice_mag_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added weight decay

# SNR Metric
def compute_snr(pred, target, eps=1e-8):
    signal = torch.sum(target ** 2, dim=(1, 2, 3))
    noise = torch.sum((target - pred) ** 2, dim=(1, 2, 3))
    snr = 10 * torch.log10(signal / (noise + eps))
    return torch.mean(snr)

# Training Loop
epochs = 50
train_losses = []
val_losses = []
train_snrs = []
val_snrs = []

for epoch in range(epochs):
    # Training Phase
    model.train()
    epoch_train_loss = 0.0
    epoch_train_snr = 0.0
    for batch_idx, (noisy, clean) in enumerate(train_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_snr += compute_snr(output, clean).item()
        progress = (batch_idx + 1) / len(train_loader) * 100
        print(f"\rEpoch {epoch + 1}/{epochs} - {progress:.2f}% complete", end="")
    
    train_loss = epoch_train_loss / len(train_loader)
    train_snr = epoch_train_snr / len(train_loader)
    train_losses.append(train_loss)
    train_snrs.append(train_snr)

    # Validation Phase
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_snr = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            epoch_val_loss += loss.item()
            epoch_val_snr += compute_snr(output, clean).item()
    
    val_loss = epoch_val_loss / len(val_loader)
    val_snr = epoch_val_snr / len(val_loader)
    val_losses.append(val_loss)
    val_snrs.append(val_snr)

    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.6f}, Train SNR: {train_snr:.6f}")
    print(f"Val Loss: {val_loss:.6f}, Val SNR: {val_snr:.6f}")

# Save Model
torch.save(model.state_dict(), "vit_unet_speech_separation.pth")

# Plot Metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_snrs, label='Train SNR')
plt.plot(range(1, epochs + 1), val_snrs, label='Validation SNR')
plt.xlabel('Epochs')
plt.ylabel('SNR (dB)')
plt.title('SNR Curve')
plt.legend()

plt.savefig('training_curves.png')

# Inference
test_noisy_mag = np.load(os.path.join(SPECTROGRAM_DIR, 'noisy_mag.npy'))
test_noisy_mag = test_noisy_mag / max_mag
test_noisy_mag_tensor = torch.tensor(test_noisy_mag, dtype=torch.float32).unsqueeze(1).to(device)

model.eval()
with torch.no_grad():
    enhanced_mag = model(test_noisy_mag_tensor)
enhanced_mag = enhanced_mag.squeeze(1).cpu().numpy()

enhanced_complex = enhanced_mag * np.exp(1j * noisy_phase)
enhanced_audio = librosa.istft(enhanced_complex)
sf.write('enhanced_audio.wav', enhanced_audio, samplerate=SAMPLE_RATE)