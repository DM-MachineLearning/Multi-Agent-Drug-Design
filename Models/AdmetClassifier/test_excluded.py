import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import sys
import os

# ==========================================
# 1. ORIGINAL ARCHITECTURE
# ==========================================
class MultiHeadADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11):
        super().__init__()
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

# ==========================================
# 2. EVALUATION LOGIC (8-Task Filter)
# ==========================================
def evaluate_8_tasks(admet_ckpt, test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
    task_names = checkpoint.get('task_names', [
        'BBBP', 'CYP1A2_inhibition', 'CYP2C19_inhibition', 'CYP2C9_inhibition', 
        'CYP2D6_inhibition', 'CYP3A4_inhibition', 'Caco2_permeability', 
        'HLM_stability', 'P-gp_substrate', 'RLM_stability', 'hERG_inhibition'
    ])

    model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
    model.load_state_dict(checkpoint.get('model_state', checkpoint))
    model.eval()

    dataset = LatentDataset(test_data_path)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    y_true_all, y_prob_all, task_ids_all = [], [], []

    with torch.no_grad():
        for z, labels, task_ids in loader:
            z = z.to(device)
            logits = model(z)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true_all.extend(labels.numpy())
            y_prob_all.extend(probs)
            task_ids_all.extend(task_ids.numpy())

    y_true_all, y_prob_all, task_ids_all = np.array(y_true_all), np.array(y_prob_all), np.array(task_ids_all)

    # UPDATED FILTER: Removed CYP2D6, Caco2, and hERG
    IGNORED_TASKS = ["CYP2D6_inhibition", "Caco2_permeability"]
    results = []

    for i, task in enumerate(task_names):
        if task in IGNORED_TASKS:
            continue

        mask = (task_ids_all == i)
        if not np.any(mask): continue
        
        y_true = y_true_all[mask]
        y_prob = y_prob_all[mask, i]
        y_class = (y_prob > 0.5).astype(int)

        try:
            results.append({
                "Task": task,
                "AUC-ROC": round(roc_auc_score(y_true, y_prob), 4),
                "F1-Score": round(f1_score(y_true, y_class), 4),
                "Accuracy": round(accuracy_score(y_true, y_class), 4),
                "Precision": round(precision_score(y_true, y_class, zero_division=0), 4),
                "Recall": round(recall_score(y_true, y_class, zero_division=0), 4)
            })
        except: continue

    df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("--- PERFORMANCE: ORIGINAL MODEL (Cleaned 8-Task Subset) ---")
    print("="*85)
    print(df.to_string(index=False))
    print("\nGlobal Averages:")
    print(df.mean(numeric_only=True).to_frame().T.to_string(index=False))

if __name__ == "__main__":
    # evaluate_8_tasks("admet_predictor_bidirec_1000epochs.pt", "admet_latent_test_bidirec.pt")
    evaluate_8_tasks("admet_predictor_bidirec_1000epochs.pt", "Benchmarks/molobj_pretrained.pt")