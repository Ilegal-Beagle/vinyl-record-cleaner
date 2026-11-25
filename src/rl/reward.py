import torch
import auraloss
from torch import Tensor
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr_tm

spec_weight = 2
damage_penalty_weight = 1

spectral_loss = auraloss.freq.STFTLoss(
    fft_size=1024,
    hop_size=256,
    win_length=1024
)


def si_sdr(estimate: Tensor, reference: Tensor):
    est = torch.as_tensor(estimate, dtype=torch.float32)
    ref = torch.as_tensor(reference, dtype=torch.float32)
    return si_sdr_tm(est, ref, zero_mean=True)


def delta_si_sdr(restored, noisy, reference):
    return si_sdr(restored, reference) - si_sdr(noisy, reference)


def spectral_distance(audio, reference):
    audio = torch.as_tensor(audio, dtype=torch.float32).reshape(1, 1, -1)

    reference = torch.as_tensor(reference, dtype=torch.float32).reshape(1, 1, -1)

    return spectral_loss(audio, reference).item()

# Batched reward functions

def batch_si_sdr(estimated, reference):
    return si_sdr_tm(estimated, reference, zero_mean=True)


def batch_spectrogram(audio, window):
    spec = torch.stft(
        audio,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        return_complex=True
    )

    return spec.abs().square().clamp_min(1e-8).sqrt()


def batch_spectral_distance(estimated, reference):
    convergence = torch.linalg.vector_norm(reference - estimated, dim=(-2, -1)) / torch.linalg.vector_norm(reference, dim=(-2, -1)).clamp_min(1e-8)

    log_distance = (estimated.log() - reference.log()).abs().mean(dim=(-2, -1))

    return convergence + log_distance


@torch.no_grad()
def batch_reward(restored, noisy, clean, device):
    restored = torch.as_tensor(restored, dtype=torch.float32, device=device)
    
    noisy = torch.as_tensor(noisy, dtype=torch.float32, device=device)

    clean = torch.as_tensor(clean, dtype=torch.float32, device=device)

    sdr = batch_si_sdr(restored, clean) - batch_si_sdr(noisy, clean)

    window = torch.hann_window(1024, device=device)

    restored_spec = batch_spectrogram(restored, window)
    noisy_spec = batch_spectrogram(noisy, window)
    clean_spec = batch_spectrogram(clean, window)

    before = batch_spectral_distance(noisy_spec, clean_spec)
    after = batch_spectral_distance(restored_spec, clean_spec)

    spectral = before - after

    damage_penalty = torch.relu(-sdr)

    reward = sdr + spec_weight * spectral - damage_penalty_weight * damage_penalty

    return torch.clamp(reward, -10.0, 20.0)