import torch
import torch.nn as nn

class DockingRegressor(nn.Module):
    """ECFP -> predicted docking score (regression)."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict(self, fp_tensor: torch.Tensor) -> float:
        """Predict docking score for a single fingerprint tensor."""
        self.eval()
        val = self(fp_tensor.unsqueeze(0))
        return float(val.item())
