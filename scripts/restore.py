import sys
import numpy as np
import torch
import soundfile as sf
import librosa
import yaml
from tqdm import tqdm

from src.rl.policy import Policy
from src.data.spectrogram import create_spectrogram
from src.denoiser.apply import apply_gpu
from pathlib import Path

cfg = yaml.safe_load(open("configs.yaml"))

SR = cfg["sample_rate"]
OVERLAP = 0.25

policy = Policy(8)

policy.load_state_dict(
    torch.load("results/policy.pt", map_location="cpu")
)

policy.eval()


def restore_file(file_path):
    audio, sample_rate = sf.read(file_path, dtype="float32")

    mono = audio.ndim == 1

    if mono:
        restored_audio = restored_audio[:, None]

    segment_length = sample_rate * cfg["segment_seconds"]
    overlap = int(segment_length * OVERLAP)
    step = segment_length - overlap

    restored_audio = np.zeros_like(audio)
    weights = np.zeros((len(audio), 1), dtype=np.float32)

    for start in tqdm(range(0, len(audio), step)):

        segment = audio[start:start + segment_length]

        original_length = len(segment)

        segment = np.pad(
            segment,
            ((0, segment_length - original_length), (0, 0))
        )

        policy_input = segment.mean(axis=1)

        if sample_rate != SR:
            policy_input = librosa.resample(
                policy_input,
                orig_sr=sample_rate,
                target_sr=SR
            )

        with torch.no_grad():
            spec = create_spectrogram(policy_input, SR)[0]
            action = policy.act(spec)[0].numpy()

        # Crossfade overlapping segments
        window = np.ones(segment_length, dtype=np.float32)

        if start > 0:
            window[:overlap] = np.linspace(0, 1, overlap)

        if start + segment_length < len(audio):
            window[-overlap:] = np.linspace(1, 0, overlap)

        window = window[:original_length]

        for channel in range(audio.shape[1]):

            restored = apply_gpu(
                segment[:, channel],
                sample_rate,
                action
            )

            restored_audio[start:start + original_length, channel] += (
                restored[:original_length] * window
            )

        weights[start:start + original_length, 0] += window

        if start + segment_length >= len(audio):
            break

    restored_audio /= np.maximum(weights, 1e-8)

    channels = audio.shape[1]

    if mono:
        audio = audio[:, 0]
        restored_audio = restored_audio[:, 0]

    output_path = Path(file_path).with_name(f"{Path(file_path).stem}_restored.wav")

    sf.write(output_path, restored_audio, sample_rate, subtype="FLOAT")

    print("Restoration complete.")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Original: {file_path}")
    print(f"Restored: {output_path}")

# Run from repo root: python -m scripts.restore <audio_file>
if __name__ == "__main__":
    restore_file(sys.argv[1])