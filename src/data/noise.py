import numpy as np
import soundfile as sf
import sys

def add_noise(file_path):

    data = sf.read(file_path)

    audio, sample_rate = data

    duration = len(audio) / sample_rate

    noisy = audio.copy()

    audio_rms = np.sqrt(np.mean(audio ** 2))
    
    # 20% chance we feed clean audio back to the model
    # The reason for this is to train the model to handle audio that is already clean and prevent it from making unnecessary corrections.
    # We also randomly add hissing, clicks, bursts of static, and a continuous crackle to give the model a good variation of noise types and combinations.
    # SNR is used to scale the hiss volume relative to the volume of the audio as it may vary.
    
    if np.random.random() > 0.2:

        # HISS ------------------
        
        if np.random.random() > 0.2:

            snr = np.random.uniform(5, 30)
        
            hiss_strength = audio_rms / (10 ** (snr / 20))
        
            hiss = np.random.normal(0, hiss_strength, audio.shape)
            noisy += hiss

        # CLICKS ------------------
        
        if np.random.random() < 0.2:
            num_clicks = 0
        else:
            num_clicks = np.random.randint(1, 31)

        # Add clicks by picking a random position and amplitude for each click to be added.
        for i in range(num_clicks):
            position = np.random.randint(0, len(audio))
    
            click_duration = np.random.uniform(0.002, 0.01)
            click_samples = max(1, int(click_duration * sample_rate))
    
            snr = np.random.uniform(-15, 5)

            amplitude = audio_rms / (10 ** (snr / 20))
            amplitude *= np.random.choice([-1, 1])
    
            click = amplitude * np.exp(-np.linspace(0, 5, click_samples))
    
            end = min(position + click_samples, len(audio))
            click = click[:end - position]
    
            if audio.ndim == 2:
                noisy[position:end] += click[:, np.newaxis]
            else:
                noisy[position:end] += click

        # BURSTS ------------------
        
        if np.random.random() < 0.2:
            num_bursts = 0
        else:
            num_bursts = np.random.randint(0, 6)

        # Add bursts by picking a random position and amplitude for each burst to be added.
        for i in range(num_bursts):
            burst_duration = 0.1
            burst_samples = int(burst_duration * sample_rate)
    
            position = np.random.randint(0, len(audio) - burst_samples + 1)
    
            snr = np.random.uniform(0, 20)

            amplitude = audio_rms / (10 ** (snr / 20))
    
            burst_shape = (burst_samples,) + audio.shape[1:]
            static = np.random.normal(0, amplitude, burst_shape)
    
            noisy[position:position + burst_samples] += static

        # CRACKLE ------------------

        # Since we may add thousands of crackles, generate their positions and amplitudes at once to improve efficiency over a large dataset
        if np.random.random() > 0.2:
        
            # Add a constant vinyl-like crackle
            crackle_rate = np.random.randint(100, 501)
            
            num_crackles = int(duration * crackle_rate)
            
            positions = np.random.randint(0, len(audio), num_crackles)
            
            snr = np.random.uniform(0, 20, num_crackles)
            
            amplitudes = audio_rms / (10 ** (snr / 20))
            amplitudes *= np.random.choice([-1, 1], num_crackles)
            
            if audio.ndim == 2:
                for channel in range(audio.shape[1]):
                    np.add.at(noisy[:, channel], positions, amplitudes)
            else:
                np.add.at(noisy, positions, amplitudes)

    return noisy, sample_rate

# Used for testing only
# Upload a .wav file to the directory and run 'python3 noise.py filename.wav' to produce a noisy version of the file
if __name__ == "__main__":

    file_path = sys.argv[1]

    noisy, sample_rate = add_noise(file_path)

    import time

    filename = f"noisy_{int(time.time())}.wav"
    sf.write(filename, noisy, sample_rate, subtype="FLOAT")

    print("Noisy audio saved as ", filename)