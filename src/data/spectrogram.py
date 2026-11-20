import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft
import sys

def create_spectrogram(file_path):

    data = sf.read(file_path)

    audio, sample_rate = data

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Apply the Short-Time Fourier Transform
    frequencies, times, Zxx = stft(
        audio,
        fs=sample_rate,
        nperseg=1024,
        noverlap=512
    )

    # Convert the Fourier Transform into a magnitude spectrogram for the model to interpret
    spectrogram = np.abs(Zxx)

    # Return the magnitude spectrogram and the original complex STFT for reconstruction
    return spectrogram, Zxx


# Used for testing only
# Upload a .wav file to the directory and run 'python3 spectrogram.py filename.wav' to produce a spectrogram of the audio
if __name__ == "__main__":

    file_path = sys.argv[1]

    spectrogram, Zxx = create_spectrogram(file_path)

    print("Spectrogram shape:", spectrogram.shape)

    plt.imshow(
        20 * np.log10(spectrogram + 1e-10),
        aspect="auto",
        origin="lower",
        cmap="magma"
    )

    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.colorbar(label="Magnitude (dB)")

    plt.savefig("spectrogram.png")
    plt.close()