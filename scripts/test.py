# src/eval/quick_eval.py
import numpy as np, pandas as pd, torch, yaml, datasets
from tqdm import tqdm
from src.data.noise import add_noise
from src.data.spectrogram import create_spectrogram
from src.denoiser.apply import apply
from src.rl.policy import Policy
from src.rl.reward import si_sdr
from src.eval.baselines import BASELINES

cfg = yaml.safe_load(open("configs.yaml"))
SR, SEG = cfg["sample_rate"], cfg["sample_rate"] * cfg["segment_seconds"]
N_TEST = 200

ds = datasets.load_from_disk("data/raw/test").cast_column("audio", datasets.Audio(sampling_rate=SR))
policy = Policy(6)
policy.load_state_dict(torch.load("results/policy.pt", map_location="cpu"))
policy.eval()

rows = []
for i in tqdm(range(min(200, len(ds)))):                 # start small
    wave = np.asarray(ds[i]["audio"]["array"], dtype=np.float32)
    rng = np.random.default_rng(i)
    np.random.seed(i)                                   # same noise every run
    start = rng.integers(0, max(1, len(wave) - SEG + 1))
    clean = np.pad(wave[start:start + SEG], (0, max(0, SEG - len(wave[start:start + SEG]))))
    if np.sqrt(np.mean(clean ** 2)) < 1e-3:
        continue
    noisy, _ = add_noise(clean, SR)

    with torch.no_grad():
        action = policy.act(create_spectrogram(noisy, SR)[0])[0].numpy()

    outputs = {name: fn(noisy, SR) for name, fn in BASELINES.items()}
    outputs["policy"] = apply(noisy, SR, action)

    base = float(si_sdr(noisy, clean))
    for method, out in outputs.items():
        s = float(si_sdr(out, clean))
        rows.append({"clip": i, "method": method, "si_sdr": s, "delta_si_sdr": s - base})

df = pd.DataFrame(rows)
df.to_csv("results/test_results.csv", index=False)
print(df.groupby("method")["delta_si_sdr"].agg(["mean", "median", "std", "count"]).round(2))