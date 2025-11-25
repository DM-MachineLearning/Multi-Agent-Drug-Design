import torch
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    """
    A memory-efficient dataset handler that loads text files line-by-line 
    and tokenizes on-the-fly to prevent RAM exhaustion.
    """
    def __init__(self, file_path: Path, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = self._load_lines(file_path)

    def _load_lines(self, path: Path) -> List[str]:
        # Reads lines into a list. For massive datasets (>10GB), 
        # consider using the 'datasets' library or file seeking.
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        line = self.lines[idx]
        
        # Tokenize with padding to max_length
        encodings = self.tokenizer(
            line,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Create labels: same as input_ids, but mask padding tokens (-100)
        # so the loss function ignores them.
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }