import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# CONFIG
DATA_PATH = "data/admet_multitask_latent.pt"
SAVE_PATH = "Models/ADMETClassifier/checkpoints/admet_latent_mlp.pt"
BATCH_SIZE = 32
EPOCHS = 1000

# --- 1. THE MODEL ---
class MultiTaskLatentPredictor(nn.Module):
    def __init__(self, input_dim=128, num_tasks=11):
        super().__init__()
        # Shared Trunk (Learns general chemical features)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Independent Heads (One per property)
        self.heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_tasks)
        ])

    def forward(self, z):
        feat = self.trunk(z)
        # Run each head and stack results
        results = [head(feat) for head in self.heads]
        return torch.cat(results, dim=1) # Shape: [Batch, 11]

# --- 2. CUSTOM LOSS FOR MISSING DATA ---
def masked_bce_loss(logits, targets):
    """
    Computes BCE loss only for valid targets (not NaN).
    """
    # Create mask: True where target is NOT NaN
    mask = ~torch.isnan(targets)
    
    # We only care about valid entries
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    
    if valid_targets.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    return nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_targets)

# --- 3. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Load Data
    data = torch.load(DATA_PATH)
    X_train = data['z'].to(device)
    y_train = data['y'].to(device)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiTaskLatentPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_z, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_z)
            
            # Use our custom masked loss
            loss = masked_bce_loss(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training Complete. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()