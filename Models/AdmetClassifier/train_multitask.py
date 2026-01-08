import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiHeadADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11):
        super().__init__()
        # Wider shared body for better chemical feature extraction
        self.shared_encoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), 
            nn.BatchNorm1d(1024),
            nn.SiLU(), 
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, z):
        features = self.shared_encoder(z)
        outputs = [head(features) for head in self.heads]
        return torch.cat(outputs, dim=1)


# --- 2. Dataset Loader ---
class LatentDataset(Dataset):
    def __init__(self, processed_path):
        data_pack = torch.load(processed_path)
        self.samples = data_pack['data']
        self.tasks = data_pack['tasks']
        self.latent_dim = data_pack['latent_dim']
        print(f"✅ Loaded {len(self.samples)} samples.")
        print(f"📋 Task Order: {self.tasks}") # IMPORTANT: Remember this order!

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item['z'], dtype=torch.float32), 
            torch.tensor(item['y'], dtype=torch.float32),
            torch.tensor(item['task_idx'], dtype=torch.long)
        )
    
def train():
    BATCH_SIZE = 1024
    EPOCHS = 1000
    LR = 1e-3 # Slightly higher starting LR with the wider model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = LatentDataset("admet_latent_train_bidirec.pt")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiHeadADMET(latent_dim=dataset.latent_dim, num_tasks=len(dataset.tasks)).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2) # Stronger weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    # IMPROVEMENT 1: pos_weight=3.0 handles the 1:3 ratio common in ADMET data
    # This makes 'Active' hits much more important to the model
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(DEVICE))

    print(f"🚀 Training Optimized 11-Head Predictor...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for z_batch, label_batch, task_ids in loader:
            z_batch, label_batch, task_ids = z_batch.to(DEVICE), label_batch.to(DEVICE), task_ids.to(DEVICE)
            
            optimizer.zero_grad()
            all_preds = model(z_batch)
            target_preds = all_preds[torch.arange(z_batch.size(0)), task_ids]
            
            # IMPROVEMENT 2: Label Smoothing (Manual)
            # Helps with the 'noisy' nature of biological assay data
            smooth_labels = label_batch * 0.9 + 0.05 
            
            loss = criterion(target_preds, smooth_labels)
            loss.backward()
            
            # IMPROVEMENT 3: Gradient Clipping
            # Keeps the 11 task gradients from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    torch.save({"model_state": model.state_dict(), "task_names": dataset.tasks}, "admet_predictor_bidirec_1000epochs.pt")
    print("✅ Optimized Model Saved!")

if __name__ == "__main__":
    train()