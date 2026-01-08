import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. Same Architecture, Just Cleaner ---
class MultiHeadADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11):
        super().__init__()
        # Wider and Deeper to force memorization if necessary
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), # Switch to ReLU for sharper gradients
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, z):
        features = self.shared_encoder(z)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)

# --- 2. Dataset with Normalization ---
class LatentDataset(Dataset):
    def __init__(self, processed_path):
        data_pack = torch.load(processed_path)
        self.samples = data_pack['data']
        self.tasks = data_pack['tasks']
        
        # Convert all Zs to a single tensor for calculation
        all_zs = np.vstack([item['z'] for item in self.samples])
        self.mean = torch.tensor(all_zs.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(all_zs.std(axis=0) + 1e-6, dtype=torch.float32)
        
        print(f"✅ Loaded {len(self.samples)} samples.")
        print(f"📊 Normalizing Inputs: Mean={self.mean.mean():.4f}, Std={self.std.mean():.4f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        z = torch.tensor(item['z'], dtype=torch.float32)
        
        # NORMALIZE Z ON THE FLY
        z_norm = (z - self.mean) / (self.std + 1e-8)        
        
        return (
            z_norm, 
            torch.tensor(item['y'], dtype=torch.float32),
            torch.tensor(item['task_idx'], dtype=torch.long)
        )

# --- 3. Simple Training Loop ---
def train_debug():
    BATCH_SIZE = 128 # Smaller batch = more updates
    EPOCHS = 30
    LR = 1e-4 # Slower, safer learning rate
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = LatentDataset("admet_latent_train_bidirec.pt") # USE YOUR NEW DATASET
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiHeadADMET(latent_dim=128, num_tasks=len(dataset.tasks)).to(DEVICE)
    # dataset.mean = dataset.mean.to(DEVICE)
    # dataset.std = dataset.std.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss() # Raw Loss, no weights

    print(f"🚀 DEBUG Training Started...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for z_batch, label_batch, task_ids in loader:
            z_batch = z_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            task_ids = task_ids.to(DEVICE)
            
            optimizer.zero_grad()
            all_preds = model(z_batch)
            
            # Select relevant head
            target_preds = all_preds[torch.arange(z_batch.size(0)), task_ids]
            
            loss = criterion(target_preds, label_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate Accuracy
            probs = torch.sigmoid(target_preds)
            preds = (probs > 0.5).float()
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)
        
        avg_loss = total_loss / len(loader)
        acc = correct / total
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

    torch.save({"model_state": model.state_dict(), "task_names": dataset.tasks}, "admet_predictor_debug.pt")

if __name__ == "__main__":
    train_debug()