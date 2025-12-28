import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
DATA_PATH = "Models/ActivityClassifier/latent_dataset.pt"   # The file you just made
SAVE_MODEL_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
LATENT_DIM = 128   # Must match your VAE's latent dimension
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.001

# --- 1. DEFINE THE MLP MODEL ---
class LatentPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super(LatentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
            # Sigmoid removed here, moved to loss function/inference
        )

    def forward(self, z):
        return self.net(z)

# --- 2. TRAINING FUNCTION ---
def train():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # Load Data
    try:
        data = torch.load(DATA_PATH)
        X_train = data['z'].to(device) # Shape: [N, 128]
        y_train = data['y'].to(device) # Shape: [N, 1]
        print(f"Loaded {len(X_train)} samples.")
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Did you run the previous script?")
        return

    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = LatentPredictor(input_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor([258 / 926]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCELoss() # Binary Cross Entropy (Standard for 0/1 classification)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_z, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(batch_z)
            
            # Loss Calculation
            loss = criterion(predictions, batch_y)
            
            # Backward Pass (Gradient Descent)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Simple logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Training Complete! Model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train()