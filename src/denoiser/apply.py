import noisereduce

def apply(waveform, sample_rate):
    noisereduce.reduce_noise(y=waveform, sr=sample_rate)

