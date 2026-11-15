# vinyl-record-cleaner

## Project Strucrture

- `src`: Where the acutal code lives for the project.
    - `data`: data handling logic goes here.
        - `download.py` downloads from FMA into `data/raw`
        - `noise.py` takes in a waveform, adds noise to it, and returns a new waveform with noise added.
        - `split.py` splits the data into training sets, validation sets, and test sets. This will need to taken in a list of song IDs from FMA and create three files (`train.txt`, `val.txt`, `test.txt`) that contain the song IDs for each set.
        - `spectrogram.py` that takes in a waveform and returns a spectrogram of that waveform.

    - `rl`: the actual NN/RL model lives here.
        - `policy.py`: takes in a batch of spectrograms, outputs a vector of settings for a denoiser to apply
        - `reward.py`: takes in noisy waveform, restored, waveform, and clean waveform, ouputs a score for reinforcement.
        - `train.py`: takes in a files of song IDs, returns a saved model checkpoint, a training log CSV, and the validation run reward.

    - `denoiser/apply.py`: Takes in a waveform and settings vector, returns a denoised version of the original waveform. 
    
    - `eval`: evaluation logic goes here.
        - `baselines.py`: defines the comparison of:
            1. no processing
            2. default noise filter from `noisereduce`
            3. a combination of settings that work best for denoising on average (not completely necessary but would be nice to compare it with). This would entail seaching through different values for the denoiser until you find something that works best on average.
        - `evaluate.py`: Actaully executes the comparison from`baseline.py`
        - `stats.py`: takes in results CSV and creates a summary of the model

- `data`: stores the audio files from datasets (.gitignored).
- `notebooks`: Python notebooks go here to show the results of the model.
- `results`: where the CSVs of the resutls of the model will go.
- `configs.yaml`: default configuration for sample rates of audio, STFT settings, an other things that will need to be used in multiple places