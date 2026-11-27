from datasets import load_dataset

ds = load_dataset("benjamin-paine/free-music-archive-small")

ds.save_to_disk("data/full")