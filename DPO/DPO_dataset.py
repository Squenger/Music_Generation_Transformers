import torch
from torch.utils.data import Dataset, DataLoader

class DPOMusicDataset(Dataset):
    def __init__(self, dpo_data_path, max_length=320):
        """
        loads the pre-computed dataset.
        - max_length: prompt_length + generation_length
        """
        print("loading DPO dataset...")
        self.data = torch.load(dpo_data_path)
        self.max_length = max_length
        print(f"loaded {len(self.data)} pairs")

    def __len__(self):
        return len(self.data)

    def _pad_sequence(self, sequence):
        """pads or truncates the sequence to the fixed max_length."""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            # 0 is the padding token ID
            padding = [0] * (self.max_length - len(sequence))
            return sequence + padding

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # extract the pre-generated sequences
        prompt = item["prompt"]
        winner = item["winner"]
        loser = item["loser"]
        
        # ensure they are exactly the same length for PyTorch batching.
        winner_padded = self._pad_sequence(winner)
        loser_padded = self._pad_sequence(loser)
        prompt_padded = self._pad_sequence(prompt)
        
        # convert to tensors
        tensor_winner = torch.tensor(winner_padded, dtype=torch.long)
        tensor_loser = torch.tensor(loser_padded, dtype=torch.long)
        tensor_prompt = torch.tensor(prompt_padded, dtype=torch.long) # Mostly for debugging
        
        return tensor_prompt, tensor_winner, tensor_loser

