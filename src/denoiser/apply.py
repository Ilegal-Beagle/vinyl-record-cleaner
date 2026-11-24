import numpy as np
import noisereduce

def to_settings(a: np.ndarray, sr: int) -> dict:
    a = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)

    # n_fft first, since the other limits depend on it: 256, 512, 1024 or 2048
    n_fft = 2 ** int(round(8 + a[5] * 3))
    hop = n_fft // 4                        # noisereduce's default hop

    # lower bounds implied by n_fft / sr (+1 for int rounding in the library)
    min_freq_hz = 2 * sr / n_fft + 1
    min_time_ms = 1000 * hop / sr + 1

    return {
        "prop_decrease":               float(a[0]),
        "n_std_thresh_stationary":     0.5 + a[1] * 3.5,
        "freq_mask_smooth_hz":         min_freq_hz + a[2] * (1000.0 - min_freq_hz),
        "time_mask_smooth_ms":         min_time_ms + a[3] * (200.0 - min_time_ms),
        "thresh_n_mult_nonstationary": 1.0 + a[4] * 3.0,
        "n_fft":                       n_fft,
        "stationary":                  False,
    }

def apply(waveform, sample_rate, action):
    out = noisereduce.reduce_noise(
        y=waveform, sr=sample_rate, **to_settings(action, sample_rate)
    )
    return np.nan_to_num(out, nan=0.0)