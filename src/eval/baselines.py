from src.rl.reward import delta_si_sdr
from src.denoiser import apply
import numpy as np

def no_processing(noisy):
    return noisy

def default_denoiser(noisy: np.ndarray, sr: int):
    return apply(noisy, sr)

def best_fixed_denoiser(noisy: np.ndarray, sr: int):
    best_settings = []
    return apply(noisy, sr, best_settings)

def mean_reward(settings, pairs):
    return float(np.mean([
        delta_si_sdr(apply(noisy, settings), noisy, clean) for noisy, clean in pairs
    ]))
