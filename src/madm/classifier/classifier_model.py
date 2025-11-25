import torch
import torch.nn as nn

class ActivityClassifier(nn.Module):
    """
    ECFP + 4 properties -> Active / Inactive logits.
    """

    def __init__(self, fp_dim: int = 2048, prop_dim: int = 4, hidden_dim: int = 512):
        super().__init__()
        input_dim = fp_dim + prop_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, fp, props):
        x = torch.cat([fp, props], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def active_prob(self, fp, props):
        logits = self(fp, props)
        return torch.softmax(logits, dim=-1)[..., 1]
