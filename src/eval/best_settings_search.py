# src/eval/fixed_search.py
import json
import numpy as np
from pathlib import Path
from src.denoiser.apply import apply
from src.rl.reward import delta_si_sdr

K = 6  # number of settings

# Fixed (noisy, clean) pairs so every candidate is scored on identical data
def make_pairs(clean_segments, sr, add_noise, n_pairs, seed=0, silence_rms=1e-3):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)                      # add_noise uses global np.random
    idx = rng.permutation(len(clean_segments))
    pairs = []

    for i in idx:
        clean = clean_segments[i]
    
        # skip silent
        if np.sqrt(np.mean(clean ** 2)) < silence_rms:
            continue
        
        noisy, _ = add_noise(clean, sr)
        pairs.append((noisy, clean))
        if len(pairs) == n_pairs:
            break

    return pairs

# calculate the scores resulting from the selected settings
def score(action, pairs, sr):
    vals = []
    for noisy, clean in pairs:
        out = apply(noisy, sr, action)
        v = float(delta_si_sdr(out, noisy, clean))

        # make sure score is valid number
        if np.isfinite(v):
            vals.append(v)
    
    return float(np.mean(vals)) if vals else -np.inf

# Use random settings and evaluate their performance, return (settings, score)
# in descending order by score
def random_search(pairs, sr, n_candidates=50, seed=0):
    rng = np.random.default_rng(seed)
    settings = rng.random((n_candidates, K))
    settings[0] = 0.5
    scores = np.array([score(c, pairs, sr) for c in settings])
    order = np.argsort(scores)[::-1]
    return settings[order], scores[order]

# use a hill-climbing method to find the settings with the best score
def refine(best, pairs, sr, rounds=3, n_local=20, sigma=0.15, seed=1):
    rng = np.random.default_rng(seed)
    best, best_score = best.copy(), score(best, pairs, sr)
    for r in range(rounds):
        local = np.clip(best + rng.normal(0, sigma, (n_local, K)), 0, 1)
        for c in local:
            s = score(c, pairs, sr)
            if s > best_score:
                best, best_score = c, s
        sigma *= 0.5
    return best, best_score

# run this to acutally get the results and output them to a json for later use
def run(clean_segments, sr, add_noise, out="results/best_fixed.json"):
    # cheap, many candidates on few pairs
    small = make_pairs(clean_segments, sr, add_noise, n_pairs=40, seed=0)
    cands, _ = random_search(small, sr, n_candidates=100)

    # score the top 10 on a bigger set to avoid picking a lucky fluke
    large = make_pairs(clean_segments, sr, add_noise, n_pairs=200, seed=1)
    top = [(c, score(c, large, sr)) for c in cands[:10]]
    best, _ = max(top, key=lambda t: t[1])

    # local refinement
    best, best_score = refine(best, large, sr)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"action": best.tolist(), "train_mean_dsisdr": best_score}, open(out, "w"))
    return best, best_score

if __name__ == '__main__':
    import yaml
    import datasets
    from torch.utils.data import DataLoader
    from src.rl.train import collate
    from src.data.noise import add_noise

    cfg = yaml.safe_load(open('configs.yaml'))
    SR = cfg['sample_rate']

    training_ds = datasets.load_from_disk('data/raw/train')
    training_ds = training_ds.cast_column('audio', datasets.Audio(sampling_rate=SR))

    clean = []
    for i in range(10):
        clean.append(training_ds[i]['audio']['array'])

    run(clean, SR, add_noise)


