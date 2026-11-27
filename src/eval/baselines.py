import json
import numpy as np
import noisereduce
from src.denoiser.apply import apply_gpu
from src.denoiser.gpu_declick import apply as declick

def no_processing(noisy: np.ndarray, sr: int):
    return noisy

def default_denoiser(noisy: np.ndarray, sr: int):
    # library defaults for noisereduce and GrooveClean detection, untouched by our settings mapping
    declicked = declick(noisy, sr, sensitivity=0.5, strength=1.0)
    return np.nan_to_num(noisereduce.reduce_noise(y=declicked, sr=sr))

_BEST = None
def best_fixed_denoiser(noisy: np.ndarray, sr: int):
    global _BEST
    if _BEST is None:                       # load lazily, only when first used
        _BEST = json.load(open('results/best_fixed.json'))['action']
    return apply_gpu(noisy, sr, _BEST)          # raw 6-vector; apply converts it

BASELINES = {
    'no_processing': no_processing,
    'default_denoiser': default_denoiser,
    'best_fixed_denoiser': best_fixed_denoiser,
}