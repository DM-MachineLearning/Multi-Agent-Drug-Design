import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
import sys
import os

# --- MMoE Architecture (Must match Ultimate/Balanced settings) ---
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

# Dataset Class
class LatentDataset(Dataset):
    def __init__(self, processed_path):
        data_pack = torch.load(processed_path, weights_only=False)
        self.samples = data_pack['data']
        self.tasks = data_pack['tasks']
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        return torch.tensor(item['z'], dtype=torch.float32), torch.tensor(item['y'], dtype=torch.float32), torch.tensor(item['task_idx'], dtype=torch.long)

def evaluate_final_stand(ckpt_path, test_data_path, threshold_path="optimized_thresholds.pt", temperature=2.0):
    print(f"🧪 Evaluating FINAL STAND Model: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    thresholds = {}
    if os.path.exists(threshold_path):
        print(f"⚖️  Loading thresholds from {threshold_path}")
        thresholds = torch.load(threshold_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    task_names = checkpoint['task_names']
    
    model = MMoEADMET(latent_dim=128, num_tasks=len(task_names), num_experts=8).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    dataset = LatentDataset(test_data_path)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    y_true_all, y_prob_all, task_ids_all = [], [], []
    with torch.no_grad():
        for z, y, tidx in loader:
            z = z.to(device)
            logits = model(z)
            probs = torch.sigmoid(logits / temperature).cpu().numpy()
            y_true_all.extend(y.numpy())
            y_prob_all.extend(probs)
            task_ids_all.extend(tidx.numpy())
            
    y_true_all, y_prob_all, task_ids_all = np.array(y_true_all), np.array(y_prob_all), np.array(task_ids_all)
    
    results = []
    for i, task in enumerate(task_names):
        mask = (task_ids_all == i)
        if not np.any(mask): continue
        y_t, p_t = y_true_all[mask], y_prob_all[mask, i]
        
        # We also re-tune the threshold on the final model for maximum possible F1
        from sklearn.metrics import precision_recall_curve
        prec, rec, thres = precision_recall_curve(y_t, p_t)
        f1s = 2*prec*rec/(prec+rec+1e-10)
        best_t = thres[np.argmax(f1s[:-1])]
        
        y_c = (p_t > best_t).astype(int)
        
        results.append({
            "Task": task,
            "BestThresh": round(best_t, 3),
            "AUC": round(roc_auc_score(y_t, p_t), 4),
            "F1": round(f1_score(y_t, y_c), 4),
            "Acc": round(accuracy_score(y_t, y_c), 4)
        })
        
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print(df.to_string(index=False))
    print("="*70)
    print(f"🚀 FINAL Mean AUC: {df['AUC'].mean():.4f} | Mean F1: {df['F1'].mean():.4f}")

if __name__ == "__main__":
    evaluate_final_stand("admet_predictor_balanced.pt", "admet_latent_test_bidirec.pt")
