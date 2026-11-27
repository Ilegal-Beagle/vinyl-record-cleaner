import numpy as np
import noisereduce
import torch
from src.denoiser.gpu_declick import batch_gpu_declick

def to_settings(a: np.ndarray, sr: int) -> dict:
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)

    n_fft = 2 ** int(round(8 + a[5] * 3 + np.log2(sr / 22050)))
    hop = n_fft // 4

    min_freq_hz = 2 * sr / n_fft + 1
    min_time_ms = 1000 * hop / sr + 1

    return {
        "prop_decrease": float(a[0]),
        "sigmoid_slope_nonstationary": 5.0 + a[1] * 10.0,
        "freq_mask_smooth_hz": min_freq_hz + a[2] * (1000.0 - min_freq_hz),
        "time_mask_smooth_ms": min_time_ms + a[3] * (200.0 - min_time_ms),
        "thresh_n_mult_nonstationary": 1.0 + a[4] * 3.0,
        "n_fft": n_fft,
        "stationary": False,
    }


def apply_spectral(waveform, sample_rate, action):
    out = noisereduce.reduce_noise(
        y=waveform,
        sr=sample_rate,
        **to_settings(action, sample_rate),
        use_torch=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return np.nan_to_num(out, nan=0.0)


@torch.no_grad()
def apply_gpu_batch(waveforms, sample_rate, actions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    waveforms = torch.as_tensor(waveforms, dtype=torch.float32, device=device)

    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).clamp(0, 1)

    gpu_actions = torch.stack([
        actions[:, 6],
        actions[:, 7]
    ], dim=1)

    declicked = batch_gpu_declick(waveforms, sample_rate, gpu_actions)

    declicked = declicked.cpu().numpy()
    actions = actions.cpu().numpy()

    return np.stack([
        apply_spectral(waveform, sample_rate, action)
        for waveform, action in zip(declicked, actions)
    ])


def apply_gpu(waveform, sample_rate, action):
    return apply_gpu_batch(
        np.asarray(waveform)[None, :],
        sample_rate,
        np.asarray(action)[None, :]
    )[0]