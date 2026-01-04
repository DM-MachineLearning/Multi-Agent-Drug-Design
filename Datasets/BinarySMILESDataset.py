import numpy as np
from torch.utils.data import Dataset

class BinarySMILESDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-tokenized SMILES from a memory-mapped NumPy file.

    This dataset loads tokenized SMILES sequences from a .npy file using memory mapping for efficiency,
    avoiding loading the entire dataset into RAM. It is significantly faster than text-based loading.

    Attributes:
        data (np.ndarray): Memory-mapped NumPy array of tokenized sequences.
        pad_token_id (int): Token ID for padding tokens.

    Args:
        npy_path (str): Path to the .npy file containing pre-tokenized SMILES sequences.
        pad_token_id (int, optional): Token ID used for padding. Default is 0.
    """
    def __init__(self, npy_path, pad_token_id=0):
        """
        Initializes the dataset by loading the NumPy file in memory-mapped mode.

        Args:
            npy_path (str): Path to the .npy file.
            pad_token_id (int, optional): Token ID for padding. Default is 0.
        """
        self.data = np.load(npy_path, mmap_mode='r')
        self.pad_token_id = pad_token_id

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a tokenized sequence at the given index and prepares labels.

        Converts the NumPy array to a PyTorch tensor and masks padding tokens in labels.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'input_ids' and 'labels'.
        """
        input_ids = torch.from_numpy(self.data[idx].astype(np.int64))

        labels = input_ids.clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }
