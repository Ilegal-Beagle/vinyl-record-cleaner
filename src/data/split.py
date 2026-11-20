# split.py
import datasets

ds = datasets.load_from_disk('data')

ds_split = ds['train'].train_test_split(test_size=.2, shuffle=True)

ds_split.save_to_disk('../../data/')