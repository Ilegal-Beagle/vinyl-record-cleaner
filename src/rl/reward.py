import torch
from torch import Tensor
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr_tm
import numpy as np

def si_sdr(estimate: Tensor, reference: Tensor):
    est = torch.as_tensor(estimate, dtype=torch.float32)
    ref = torch.as_tensor(reference, dtype=torch.float32)
    return si_sdr_tm(est, ref, zero_mean=True)

def delta_si_sdr(restored: Tensor, noisy: Tensor, reference: Tensor):
    return si_sdr(restored, reference) - si_sdr(noisy, reference)

def reward(restored: Tensor, noisy: Tensor, reference: Tensor, clip=(-10.0, 20.0)):
    return float(np.clip(delta_si_sdr(restored, noisy, reference), *clip))