from torch.utils.data import Dataset
import numpy as np
import torch

class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self
    
    
class OnDemandDataset(Dataset):
    def __init__(self, file_list, cache_size=10000):
        self.file_list = file_list
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # If data is not in cache
        if idx not in self.cache:
            obs_temp = np.load(self.file_list[idx])
            obs_temp = torch.from_numpy(obs_temp)
            # If cache is full, remove the oldest item
            if len(self.cache) >= self.cache_size:
                oldest_key = list(self.cache.keys())[0]
                del self.cache[oldest_key]
            self.cache[idx] = obs_temp

        return self.cache[idx], torch.zeros(self.cache[idx].size(0), dtype=bool)
