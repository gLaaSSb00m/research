import numpy as np
import librosa
import soundfile as sf
import os
import gc

# Preprocessing Configuration
SAMPLE_RATE = 8000
FRAME_LENGTH = 8064
HOP_LENGTH_FRAME = 8064
N_FFT = 255
HOP_LENGTH = 63
SNR = 0
DIM_SQUARE_SPEC = int(N_FFT / 2) + 1  # 128
BATCH_SIZE = 1000  # Adjustable batch size to manage memory
CHUNK_SIZE = 800000  # Chunk size for audio writing to manage large files

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

def extract_stft_features_in_batches(numpy_audio, dim_square_spec, n_fft, hop_length_fft, batch_size=BATCH_SIZE, output_dir=SPECTROGRAM_DIR):
    os.makedirs(output_dir, exist_ok=True)
    nb_audio = numpy_audio.shape[0]
    mag_files = []
    phase_files = []
    for batch_idx, start in enumerate(range(0, nb_audio, batch_size)):
        end = min(start + batch_size, nb_audio)
        batch = numpy_audio[start:end]
        mag = np.zeros((len(batch), dim_square_spec, dim_square_spec), dtype=np.float64)
        phase = np.zeros((len(batch), dim_square_spec, dim_square_spec), dtype=np.complex128)
        for i, audio in enumerate(batch):
            mag[i, :, :], phase[i, :, :] = calculate_stft(audio, n_fft, hop_length_fft)
        mag_file = os.path.join(output_dir, f'mag_batch_{batch_idx}.npy')
        phase_file = os.path.join(output_dir, f'phase_batch_{batch_idx}.npy')
        np.save(mag_file, mag)
        np.save(phase_file, phase)
        mag_files.append(mag_file)
        phase_files.append(phase_file)
        del mag, phase
        gc.collect()
    print(f"STFT features saved in batches to {output_dir}")
    return mag_files, phase_files

def write_audio_in_chunks(data, filename, samplerate, chunk_size=CHUNK_SIZE):
    total_samples = data.shape[1]
    with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=1, subtype='FLOAT') as file:
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = data[0, start:end].reshape(-1)
            file.write(chunk.astype(np.float64))  # Ensure float64 format

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

# Save audio files in chunks
try:
    write_audio_in_chunks(voice_reshaped, './sound/voice.wav', SAMPLE_RATE)
    write_audio_in_chunks(noise_reshaped, './sound/noise.wav', SAMPLE_RATE)
    write_audio_in_chunks(noisy_reshaped, './sound/noisy.wav', SAMPLE_RATE)
except Exception as e:
    print(f"Error writing audio file: {e}")
    raise

# Extract STFT features in batches
voice_mag_files, voice_phase_files = extract_stft_features_in_batches(prod_voice, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)
noise_mag_files, noise_phase_files = extract_stft_features_in_batches(prod_noise, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)
noisy_mag_files, noisy_phase_files = extract_stft_features_in_batches(prod_noisy, DIM_SQUARE_SPEC, N_FFT, HOP_LENGTH)

# Clean up large arrays to free memory
del prod_voice, prod_noise, prod_noisy, voice, noise
gc.collect()

# Load and combine batches into single arrays
voice_mag = np.concatenate([np.load(f) for f in voice_mag_files])
voice_phase = np.concatenate([np.load(f) for f in voice_phase_files])
noise_mag = np.concatenate([np.load(f) for f in noise_mag_files])
noise_phase = np.concatenate([np.load(f) for f in noise_phase_files])
noisy_mag = np.concatenate([np.load(f) for f in noisy_mag_files])
noisy_phase = np.concatenate([np.load(f) for f in noisy_phase_files])

# Save combined spectrograms
np.save(os.path.join(SPECTROGRAM_DIR, 'voice_mag.npy'), voice_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'voice_phase.npy'), voice_phase)
np.save(os.path.join(SPECTROGRAM_DIR, 'noise_mag.npy'), noise_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'noise_phase.npy'), noise_phase)
np.save(os.path.join(SPECTROGRAM_DIR, 'noisy_mag.npy'), noisy_mag)
np.save(os.path.join(SPECTROGRAM_DIR, 'noisy_phase.npy'), noisy_phase)
