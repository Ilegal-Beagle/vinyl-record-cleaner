from src.data.spectrogram import create_spectrogram
from src.data.noise import add_noise
from src.rl.policy import Policy
from src.rl.reward import batch_reward
from src.denoiser.apply import apply
from torch.utils.data import DataLoader
import numpy as np
import datasets
import torch
import yaml

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
SILENCE_RMS = 1e-3 # used to filter out silent segments

import csv

LOG_PATH = "results/train_log.csv"
FIELDS = ["epoch", "step", "mean_reward", "reward_std", "loss",
          "n_kept", "n_total", "frac_worse_than_noop", "mean_std",
          "grad_norm"] + [f"action_{i}" for i in range(6)]

def fix_length(x, n=SEG_LEN):
    if len(x) >= n:
        start = np.random.randint(0, len(x) - n + 1)
        return x[start:start + n]
    return np.pad(x, (0, n - len(x)))

def collate(batch):
    cleans, noisies = [], []
    for item in batch:
        audio = item["audio"]

        # supports multiple versions of datasets
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

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

    policy = Policy(settings_shape=6).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    step = 0
    policy.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            clean, noisy = batch['clean'].numpy(), batch['noisy'].numpy()

            # drop silent segments
            rms = np.sqrt((clean ** 2).mean(axis=1))
            keep = rms > SILENCE_RMS
            if keep.sum() < 2:
                continue
            clean, noisy = clean[keep], noisy[keep]

            # create the spectrograms and send them throught the model
            specs = np.stack([create_spectrogram(s, SR)[0] for s in noisy])
            specs = torch.as_tensor(specs, dtype=torch.float32, device=device)

            actions, log_probs = policy.sample(specs)   # actions: (32, 6), log_probs: (32,)
            # log_prob of the action actually applied (recompute, with grad)
            dist = policy.distribution(specs)
            log_probs = dist.log_prob(actions).sum(-1)   # (B,)

            restored = np.stack([
                apply(n, SR, a.numpy())
                for n, a in zip(noisy, actions.detach().cpu())
            ])

            rewards = batch_reward(restored, noisy, clean, device)

            ok = torch.isfinite(rewards)
            if ok.sum() < 2:
                continue

            rewards, log_probs = rewards[ok], log_probs[ok]

            adv = rewards - rewards.mean()
            loss = -(log_probs * adv).mean()

            opt.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

            opt.step()

            step += 1

            row = {
                "epoch": epoch, "step": step,
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "loss": loss.item(),
                "n_kept": int(ok.sum()), "n_total": len(ok),
                "frac_worse_than_noop": (rewards < 0).float().mean().item(),
                "mean_std": policy.log_std.exp().mean().item(),
                "grad_norm": float(grad_norm),
                **{f"action_{i}": actions[:, i].mean().item() for i in range(6)},
            }
            writer.writerow(row)
            log_file.flush()      # so a crash doesn't lose the log

            print(f"Epoch {epoch + 1}/{EPOCHS} | Step {step} | Reward: {rewards.mean().item():.4f}", flush=True)
            
    log_file.close()
    torch.save(policy.state_dict(), 'results/policy.pt')
