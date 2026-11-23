# src/eval/evaluate.py
from src.rl.train import fix_length
import librosa
import zlib
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.data.noise import add_noise            # (clean, snr_db, seed) -> noisy
from src.data.spectrogram import create_spectrogram
from src.denoiser.apply import apply
from src.rl.policy import Policy
from src.rl.reward import si_sdr, delta_si_sdr
from src.eval.baselines import BASELINES


def stable_seed(song_id, snr_db, base_seed):
    # not Python's hash(): that changes between runs
    return zlib.crc32(f"{song_id}|{snr_db}|{base_seed}".encode())


def load_policy(ckpt_path, k):
    policy = Policy(k)
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()
    return policy


def run_evaluation(cfg_path="configs/default.yaml",
                   ckpt="results/policy.pt",
                   out_csv="results/test_results.csv"):
                   
    cfg = yaml.safe_load(open(cfg_path))
    sr, seg_sec = cfg["sample_rate"], cfg["segment_seconds"]
    test_ids = Path("data/splits/test.txt").read_text().split()
    policy = load_policy(ckpt, cfg["n_settings"])

    rows = []
    for song_id in tqdm(test_ids):
        clean_wave = librosa.load(song_id, sr)

        for snr_db in cfg["eval_snr_levels"]:                 # e.g. [0, 5, 10]
            seed = stable_seed(song_id, snr_db, cfg["seed"])
            noisy_wave = add_noise(clean_wave, snr_db, seed)

            clean_segs = fix_length(clean_wave, sr, seg_sec)     # (N, samples)
            noisy_segs = fix_length(noisy_wave, sr, seg_sec)

            # policy decides once per segment, all at once, no randomness
            with torch.no_grad():
                actions = policy.act(create_spectrogram(noisy_segs, sample_rate=sr)).numpy()   # (N, K)

            # go through the noise/clean pairs
            for i, (noisy, clean) in enumerate(zip(noisy_segs, clean_segs)):
                
                # skip silent segments
                if np.sqrt(np.mean(clean**2)) < cfg["silence_rms"]:
                    continue

                # apply baselines and ouputted settings
                outputs = {name: fn(noisy) for name, fn in BASELINES.items()}
                outputs["policy"] = apply(noisy, actions[i])

                base = float(si_sdr(noisy, clean))
                for method, restored in outputs.items():
                    score = float(si_sdr(restored, clean))
                    rows.append({
                        "song_id": song_id, "segment": i, "snr_db": snr_db,
                        "method": method,
                        "si_sdr": score,
                        "delta_si_sdr": score - base,          # unclipped, on purpose
                    })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df