import os
import pickle

import numpy as np

vocab_size = 4

splits = ["train", "val"]

for split in splits:
    dataset_size = 1024 if split == "train" else 128
    obs = np.random.randn(
        dataset_size, 201, 256
    )  # (dataset_size, T_o, C): Note that T_o here is the sequence length of the obs
    action = np.random.randint(
        0, vocab_size, (dataset_size, 200), dtype= np.uint16
    )  # (dataset_size, T_a)   : Note that T_a here is the sequence length of the actions

    obs.tofile(os.path.join(os.path.dirname(__file__), f"{split}_obs.bin"))
    action.tofile(os.path.join(os.path.dirname(__file__), f"{split}_action.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": {i: i for i in range(4)},
    "stoi": {i: i for i in range(4)},
}

with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
