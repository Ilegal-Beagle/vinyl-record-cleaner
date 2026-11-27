import numpy as np, pandas as pd, torch, yaml, datasets
from tqdm import tqdm
from src.data.noise import add_noise
from src.data.spectrogram import create_spectrogram
from src.denoiser.apply import apply_gpu
from src.rl.policy import Policy
from src.rl.reward import si_sdr, spectral_distance
from src.eval.baselines import BASELINES

cfg = yaml.safe_load(open("configs.yaml"))

SR = cfg["sample_rate"]
SEG = SR * cfg["segment_seconds"]

START = 0
N_TEST = 200

ds = datasets.load_from_disk("data/raw/test").cast_column(
    "audio", datasets.Audio(sampling_rate=SR)
)

policy = Policy(8)

policy.load_state_dict(torch.load("results/policy.pt", map_location="cpu"))

policy.eval()

rows = []
policy_actions = []

for i in tqdm(range(START, min(START + N_TEST, len(ds)))):

    wave = np.asarray(ds[i]["audio"]["array"], dtype=np.float32)

    rng = np.random.default_rng(i)
    np.random.seed(i)

    start = rng.integers(0, max(1, len(wave) - SEG + 1))

    clean = wave[start:start + SEG]
    clean = np.pad(clean, (0, max(0, SEG - len(clean))))

    if np.sqrt(np.mean(clean ** 2)) < 1e-3:
        continue

    noisy, _ = add_noise(clean, SR)

    # Predict restoration settings for the noisy segment
    with torch.no_grad():
        spec = create_spectrogram(noisy, SR)[0]
        action = policy.act(spec)[0].numpy()

    policy_actions.append(action)

    # Evaluate each baseline on the same noisy segment
    outputs = {
        name: fn(noisy, SR)
        for name, fn in BASELINES.items()
    }

    # Restore the audio using the policy's predicted settings
    outputs["policy"] = apply_gpu(noisy, SR, action)

    clean_tensor = torch.as_tensor(clean, dtype=torch.float32).unsqueeze(0)

    noisy_tensor = torch.as_tensor(noisy, dtype=torch.float32).unsqueeze(0)

    base = float(si_sdr(noisy, clean))

    base_spec = float(spectral_distance(noisy_tensor, clean_tensor))

    # Calculate improvement relative to the original noisy audio
    for method, out in outputs.items():

        s = float(si_sdr(out, clean))

        out_tensor = torch.as_tensor(out, dtype=torch.float32).unsqueeze(0)

        spec = float(spectral_distance(out_tensor, clean_tensor))

        rows.append({
            "clip": i,
            "method": method,
            "si_sdr": s,
            "delta_si_sdr": s - base,
            "spectral_distance": spec,
            "delta_spec": base_spec - spec
        })


df = pd.DataFrame(rows)

df.to_csv("results/test_results.csv", index=False)

for metric in ["delta_si_sdr", "delta_spec"]:

    grouped = df.groupby("method")[metric]

    print(f"\n{metric} Improvement:")
    print(grouped.agg(["mean", "median", "std", "count"]).round(4))

    print(f"\n{metric} Success Rate (%):")
    print(grouped.apply(lambda x: (x > 0).mean() * 100).round(2))


action_df = pd.DataFrame(policy_actions, columns=[f"action_{i}" for i in range(8)])

action_df.to_csv("results/policy_actions.csv", index=False)

print("\nLearned Policy Actions:")
print(action_df.describe().round(4).to_string())