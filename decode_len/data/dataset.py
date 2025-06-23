import numpy as np
import json, argparse, torch
from torch.utils.data import Dataset, Subset

class PromptLengthDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, max_label):
        with open(path) as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_label = max_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inputs = self.tokenizer(item["prompt"], 
                                truncation=True, 
                                padding="max_length", 
                                max_length=self.max_len, 
                                return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        clipped_len = min(item["length"], self.max_label - 1)
        inputs["labels"] = torch.tensor(clipped_len, dtype=torch.long)
        return inputs