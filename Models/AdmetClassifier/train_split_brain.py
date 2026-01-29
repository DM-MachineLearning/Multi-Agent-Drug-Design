import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import copy

# --- Grouped Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class GroupedADMET(nn.Module):
    def __init__(self, latent_dim=128, all_tasks=None):
        super().__init__()
        
        # 1. Define Groups based on your task list
        # Indices based on your provided list: 
        # ['BBBP', 'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'Caco2', 'HLM', 'P-gp', 'RLM', 'hERG']
        self.cyp_indices = [1, 2, 3, 4, 5] 
        self.prop_indices = [0, 6, 7, 8, 9, 10]
        
        # 2. ENCODER A: The "Enzyme Specialist" (Deep & Narrow for structure)
        self.cyp_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2), # Extra depth for complex CYPs
            nn.Linear(512, 256),
            nn.SiLU()
        )
        
        # 3. ENCODER B: The "Property Specialist" (Wide & Shallow for physical props)
        self.prop_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2),
            nn.Linear(512, 256),
            nn.SiLU()
        )
        
        # 4. Heads (Specific to each task)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(len(all_tasks))
        ])
        
    def forward(self, z):
        # Path A: Process CYPs
        cyp_feats = self.cyp_encoder(z)
        
        # Path B: Process Properties
        prop_feats = self.prop_encoder(z)
        
        outputs = []
        for i in range(len(self.heads)):
            if i in self.cyp_indices:
                # CYP heads read from CYP encoder
                outputs.append(self.heads[i](cyp_feats))
            else:
                # Prop heads read from Prop encoder
                outputs.append(self.heads[i](prop_feats))
                
        return torch.cat(outputs, dim=1)

# --- Dataset (Unchanged) ---
class LatentDataset(Dataset):
    def __init__(self, processed_path):
        data_pack = torch.load(processed_path, weights_only=False)
        self.samples = data_pack['data']
        self.tasks = data_pack['tasks']
        self.latent_dim = data_pack['latent_dim']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item['z'], dtype=torch.float32), 
            torch.tensor(item['y'], dtype=torch.float32),
            torch.tensor(item['task_idx'], dtype=torch.long)
        )

# --- Helper: Weights Calculation (Keep this!) ---
def calculate_pos_weights(dataset):
    print("⚖️  Calculating class weights...")
    num_tasks = len(dataset.tasks)
    pos_counts = torch.zeros(num_tasks)
    neg_counts = torch.zeros(num_tasks)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    for _, labels, task_ids in loader:
        for t in range(num_tasks):
            mask = (task_ids == t)
            if mask.sum() > 0:
                task_labels = labels[mask]
                pos_counts[t] += (task_labels == 1).sum()
                neg_counts[t] += (task_labels == 0).sum()
    pos_weights = neg_counts / (pos_counts + 1e-6)
    return pos_weights

# --- Training with MIXUP ---
def train():
    BATCH_SIZE = 1024
    MAX_EPOCHS = 300
    PATIENCE = 20
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    full_dataset = LatentDataset("admet_latent_train_bidirec.pt")
    task_weights = calculate_pos_weights(full_dataset).to(DEVICE)
    
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Split-Brain Model
    model = GroupedADMET(latent_dim=full_dataset.latent_dim, all_tasks=full_dataset.tasks).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = nn.BCEWithLogitsLoss(reduction='none') 

    print(f"🚀 Training Split-Brain Model with Mixup...")

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        
        for z, label, task_ids in train_loader:
            z, label, task_ids = z.to(DEVICE), label.to(DEVICE), task_ids.to(DEVICE)
            
            optimizer.zero_grad()
            
            # === MIXUP AUGMENTATION ===
            # Creates "in-between" molecules to smooth decision boundaries
            if np.random.random() < 0.5: # Apply mixup 50% of the time
                lam = np.random.beta(0.5, 0.5)
                index = torch.randperm(z.size(0)).to(DEVICE)
                
                mixed_z = lam * z + (1 - lam) * z[index]
                # We also mix the labels!
                mixed_label = lam * label + (1 - lam) * label[index]
                # Note: We can only mix samples from the SAME task comfortably here.
                # Simplification: We only mix z, but keep original labels if tasks differ, 
                # which is risky. 
                # Better Approach for Multi-Task Mixup:
                # Only apply mixup to the latent space z, but enforce consistency? 
                # Actually, simpler Noise Injection is safer for Multi-Task if tasks differ per row.
                # Let's revert to AGGRESSIVE NOISE instead of Mixup to avoid label mismatch errors.
                
                noise = torch.randn_like(z) * 0.1 # Stronger noise
                z_in = z + noise
                target_y = label
            else:
                z_in = z
                target_y = label
            
            all_preds = model(z_in)
            target_preds = all_preds[torch.arange(z.size(0)), task_ids]
            
            # Weighted Loss Calculation
            sample_weights = task_weights[task_ids]
            loss_per_sample = criterion(target_preds, target_y)
            weighted_loss = loss_per_sample * (target_y * sample_weights + (1 - target_y))
            loss = weighted_loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation (Standard)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z, label, task_ids in val_loader:
                z, label, task_ids = z.to(DEVICE), label.to(DEVICE), task_ids.to(DEVICE)
                all_preds = model(z)
                target_preds = all_preds[torch.arange(z.size(0)), task_ids]
                
                sample_weights = task_weights[task_ids]
                loss_per_sample = criterion(target_preds, label)
                weighted_loss = loss_per_sample * (label * sample_weights + (1 - label))
                val_loss += weighted_loss.mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    torch.save({"model_state": model.state_dict(), "task_names": full_dataset.tasks}, "admet_predictor_split.pt")
    print("✅ Split-Brain Model Saved!")

if __name__ == "__main__":
    train()