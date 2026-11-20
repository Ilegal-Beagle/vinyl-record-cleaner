# import sys
# sys.path.append('../src')
from src.rl.reward import si_sdr, delta_si_sdr
import numpy as np

def test_reward():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(22050)

    # 1. adding noise at 5 dB SNR should give SI-SDR close to 5
    noise = rng.standard_normal(22050)
    noise *= np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10**(5/10)))
    noisy = clean + noise
    print(si_sdr(noisy, clean))                  # expect about 5

    # 2. scaling the estimate must not change the score
    assert abs(si_sdr(0.5 * noisy, clean) - si_sdr(noisy, clean)) < 1e-3

    # 3. no processing gives delta of 0
    assert abs(delta_si_sdr(noisy, noisy, clean)) < 1e-6

    # 4. a perfect restoration gives a large positive delta
    assert delta_si_sdr(clean, noisy, clean) > 20

if __name__ == "__main__":
    test_reward()
    print('test passed!')