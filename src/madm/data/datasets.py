from torch.utils.data import Dataset
import torch
from .featurization import smiles_to_ecfp

class MoleculePropertyDataset(Dataset):
    """
    Dataset for classifier training.

    Each item is expected to be a dict-like with keys:
    - 'smiles': SMILES string
    - 'qed', 'sa', 'admet', 'dock': floats
    - 'label': 0 (inactive) or 1 (active)
    """

    def __init__(self, rows, radius: int = 2, n_bits: int = 2048):
        self.rows = list(rows)
        self.radius = radius
        self.n_bits = n_bits

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        smiles = row["smiles"]
        fp = smiles_to_ecfp(smiles, self.radius, self.n_bits)
        props = torch.tensor(
            [row["qed"], row["sa"], row["admet"], row["dock"]],
            dtype=torch.float32,
        )
        y = torch.tensor(row["label"], dtype=torch.long)
        return fp, props, y
