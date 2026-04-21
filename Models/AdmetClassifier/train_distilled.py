import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import pandas as pd
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- 1. MMoE Architecture (Retained for high capacity) ---
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, output_dim)),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(output_dim, output_dim)),
            nn.SiLU()
        )
    def forward(self, x): return self.net(x)

class MMoEADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11, num_experts=8, expert_dim=512):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(latent_dim, expert_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, num_experts), nn.Softmax(dim=-1)) for _ in range(num_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 256), nn.SiLU(), nn.Linear(256, 1)) for _ in range(num_tasks)
        ])
        
    def forward(self, z):
        expert_outputs = torch.stack([expert(z) for expert in self.experts])
        final_outputs = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](z).unsqueeze(2)
            combined_experts = torch.sum(expert_outputs.permute(1, 0, 2) * gate_weights, dim=1)
            final_outputs.append(self.heads[i](combined_experts))
        return torch.cat(final_outputs, dim=1)

# --- 2. Balanced Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        return loss.mean()

# --- 3. Dataset & Sampler ---
class DistillDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['z'], dtype=torch.float32),
            torch.tensor(item['y_true'], dtype=torch.float32),
            torch.tensor(item['y_soft'], dtype=torch.float32),
            torch.tensor(item['task_idx'], dtype=torch.long)
        )

def get_balanced_sampler(data_list):
    print("⚖️  Calculating class weights for balanced sampling...")
    # Group samples by (task_idx, y_true)
    counts = {}
    for item in data_list:
        key = (item['task_idx'], int(item['y_true']))
        counts[key] = counts.get(key, 0) + 1
    
    # Weight per sample = 1 / count of its class in its task
    weights = []
    for item in data_list:
        key = (item['task_idx'], int(item['y_true']))
        weights.append(1.0 / counts[key])
    
    return WeightedRandomSampler(weights, len(weights))

# --- 4. Training ---
def train_balanced():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load("admet_distill_data.pt", weights_only=False)
    data_list = cache['data']
    tasks = cache['tasks']

    # Use Weighted Sampler only on Training set
    dataset = DistillDataset(data_list)
    sampler = get_balanced_sampler(data_list)
    
    # For this final attempt, we use the whole dataset with the balanced sampler 
    # and a small held-out validation for monitoring
    train_loader = DataLoader(dataset, batch_size=1024, sampler=sampler)
    
    model = MMoEADMET(latent_dim=128, num_tasks=len(tasks), num_experts=8).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    focal_loss = FocalLoss()
    mse_loss = nn.MSELoss()
    
    print(f"🚀 FINAL STAND: Training Balanced MMoE ADMET...")
    
    for epoch in range(500):
        model.train()
        total_loss = 0
        for z, y_true, y_soft, t_idx in train_loader:
            z, y_true, y_soft, t_idx = z.to(DEVICE), y_true.to(DEVICE), y_soft.to(DEVICE), t_idx.to(DEVICE)
            
            optimizer.zero_grad()
            all_preds = model(z)
            target_preds = all_preds[torch.arange(z.size(0)), t_idx]
            
            # Loss = Focal (Hard Samples) + Distill (Structural Knowledge)
            loss = 0.6 * focal_loss(target_preds, y_true) + 0.4 * mse_loss(torch.sigmoid(target_preds), y_soft)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1:03} | Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save({"model_state": model.state_dict(), "task_names": tasks}, "admet_predictor_balanced.pt")
    print("💎 Balanced Model Saved to admet_predictor_balanced.pt")

if __name__ == "__main__":
    train_balanced()
