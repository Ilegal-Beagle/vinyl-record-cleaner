import numpy as np
import noisereduce

def apply(waveform: np.ndarray, sample_rate: int, settings: dict[str, float]):
    noisereduce.reduce_noise(y=waveform, sr=sample_rate, **settings)

