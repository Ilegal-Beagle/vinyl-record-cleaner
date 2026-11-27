# split.py
import datasets

ds = datasets.load_from_disk("data/full")

ds_split = ds["train"].train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=42
)

ds_split.save_to_disk("data/raw")

print("Training samples:", len(ds_split["train"]))
print("Testing samples:", len(ds_split["test"]))