from src.data.spectrogram import create_spectrogram
from src.data.noise import add_noise
from src.rl.policy import Policy
from src.rl.reward import batch_reward
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import datasets
import torch
import yaml
import csv

conf = None

try:
    file = open('configs.yaml')
    conf = yaml.safe_load(file)
except Exception:
    print('configs.yaml not found')
    exit()

SR = conf['sample_rate']
SEG_LEN = SR * conf['segment_seconds']
BATCH_SIZE = 32
EPOCHS = 30
SILENCE_RMS = 1e-3

LOG_PATH = "results/train_log.csv"
MODEL_PATH = "results/policy.pt"

FIELDS = [
    "epoch", "step", "mean_reward", "reward_std", "loss",
    "n_kept", "n_total", "frac_worse_than_noop", "mean_std",
    "grad_norm"
] + [f"action_{i}" for i in range(8)]


def fix_length(x, n=SEG_LEN):
    if len(x) >= n:
        start = np.random.randint(0, len(x) - n + 1)
        return x[start:start + n]

    return np.pad(x, (0, n - len(x)))


def collate(batch):
    cleans, noisies = [], []

    for item in batch:
        audio = item["audio"]

        # Supports multiple versions of datasets
        if hasattr(audio, "get_all_samples"):
            audio = audio.get_all_samples().data.mean(dim=0).numpy()
        else:
            audio = audio["array"]

        clean = fix_length(np.asarray(audio, dtype=np.float32))
        noisy, _ = add_noise(clean, SR)

        assert noisy.shape == clean.shape, (noisy.shape, clean.shape)

        cleans.append(clean)
        noisies.append(noisy)

    return {
        "clean": torch.from_numpy(np.stack(cleans)),
        "noisy": torch.from_numpy(np.stack(noisies)),
    }


def train():
    from src.denoiser.apply import apply_gpu_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)

    Path("results").mkdir(exist_ok=True)

    log_file = open(LOG_PATH, "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=FIELDS)
    writer.writeheader()

    training_ds = datasets.load_from_disk('data/raw/train')

    training_ds = training_ds.cast_column('audio', datasets.Audio(sampling_rate=SR))

    dataloader = DataLoader(
        training_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    policy = Policy(settings_shape=8).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    step = 0
    policy.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:

            clean = batch['clean'].numpy()
            noisy = batch['noisy'].numpy()

            # Drop silent segments
            rms = np.sqrt((clean ** 2).mean(axis=1))
            keep = rms > SILENCE_RMS

            if keep.sum() < 2:
                continue

            clean, noisy = clean[keep], noisy[keep]

            specs = np.stack([create_spectrogram(s, SR)[0] for s in noisy])

            specs = torch.as_tensor(specs, dtype=torch.float32, device=device)

            actions, log_probs = policy.sample(specs)

            restored = apply_gpu_batch(noisy, SR, actions.detach())

            rewards = batch_reward(restored, noisy, clean, device)

            ok = torch.isfinite(rewards)

            if ok.sum() < 2:
                continue

            rewards = rewards[ok]
            log_probs = log_probs[ok]

            adv = rewards - rewards.mean()
            loss = -(log_probs * adv).mean()

            opt.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

            opt.step()

            step += 1

            row = {
                "epoch": epoch,
                "step": step,
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "loss": loss.item(),
                "n_kept": int(ok.sum()),
                "n_total": len(ok),
                "frac_worse_than_noop": (rewards < 0).float().mean().item(),
                "mean_std": policy.log_std.exp().mean().item(),
                "grad_norm": float(grad_norm),
                **{
                    f"action_{i}": actions[:, i].mean().item()
                    for i in range(8)
                },
            }

            writer.writerow(row)
            log_file.flush()

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Step {step} | "
                f"Reward: {rewards.mean().item():.4f} | ",
                flush=True
            )

        torch.save(policy.state_dict(), MODEL_PATH)

    log_file.close()
    torch.save(policy.state_dict(), MODEL_PATH)