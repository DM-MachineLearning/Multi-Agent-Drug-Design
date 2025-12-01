import torch
import torch.nn as nn
import numpy as np

class DockingRegressor(nn.Module):
    """ECFP -> predicted docking score (regression).

    The network is a configurable MLP with `num_layers` hidden layers of size
    `hidden_dim` by default. With the default `num_layers=2`, this reproduces
    the original architecture:

        input_dim -> hidden_dim -> hidden_dim -> 1
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        num_layers: int = 2,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Additional hidden layers (if any)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, fp_tensor: torch.Tensor) -> float:
        """Predict docking score for a single fingerprint tensor."""
        self.eval()
        val = self(fp_tensor.unsqueeze(0))
        return float(val.item())
    
    @torch.no_grad()
    def predict_batch(self, fp_tensor_batch: torch.Tensor) -> np.ndarray:
        """Predict docking scores for a batch of fingerprint tensors.
        
        Args:
            fp_tensor_batch: Tensor of shape (batch_size, input_dim)
            
        Returns:
            numpy array of shape (batch_size,) with predicted scores
        """
        self.eval()
        predictions = self(fp_tensor_batch)
        return predictions.cpu().numpy()
    
    def save_state(self, path: str):
        """Save model state dict to file."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_state(
        cls,
        path: str,
        map_location: str = 'cpu',
        input_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        num_layers: int = 2,
    ):
        """Load model state dict from file.
        
        Args:
            path: Path to saved state dict
            map_location: Device to load on ('cpu' or 'cuda')
            input_dim: Input dimension (must match saved model)
            hidden_dim: Hidden dimension (must match saved model)
            dropout: Dropout rate (must match saved model)
            num_layers: Number of hidden layers (must match saved model)
            
        Returns:
            DockingRegressor instance with loaded weights
        """
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
        )
        model.load_state_dict(torch.load(path, map_location=map_location))
        return model
