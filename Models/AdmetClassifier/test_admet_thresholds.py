import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import sys
import os

# ==========================================
# 1. ARCHITECTURE DEFINITION (Matches Training)
# ==========================================
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
        self.cyp_indices = [1, 2, 3, 4, 5] 
        self.prop_indices = [0, 6, 7, 8, 9, 10]
        
        self.cyp_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2), 
            nn.Linear(512, 256),
            nn.SiLU()
        )
        
        self.prop_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2),
            nn.Linear(512, 256),
            nn.SiLU()
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1)
            ) for _ in range(len(all_tasks))
        ])
        
    def forward(self, z):
        cyp_feats = self.cyp_encoder(z)
        prop_feats = self.prop_encoder(z)
        outputs = []
        for i in range(len(self.heads)):
            if i in self.cyp_indices:
                outputs.append(self.heads[i](cyp_feats))
            else:
                outputs.append(self.heads[i](prop_feats))
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
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_on_test(admet_ckpt, test_data_path, threshold_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Thresholds
    if os.path.exists(threshold_path):
        print(f"✅ Loading optimized thresholds from {threshold_path}")
        thresholds = torch.load(threshold_path)
    else:
        print("⚠️  Warning: Threshold file not found! Using 0.5 default.")
        thresholds = {}

    # Load Model
    print(f"📥 Loading split-brain model from {admet_ckpt}...")
    checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
    task_names = checkpoint['task_names']
    
    model = GroupedADMET(latent_dim=128, all_tasks=task_names).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Load Dataset
    dataset = LatentDataset(test_data_path)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    y_true_all, y_prob_all, task_ids_all = [], [], []

    print(f"🧪 Evaluating {len(dataset)} samples across {len(task_names)} tasks...")

    with torch.no_grad():
        for z, labels, task_ids in loader:
            z = z.to(device)
            logits = model(z)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            y_true_all.extend(labels.numpy())
            y_prob_all.extend(probs)
            task_ids_all.extend(task_ids.numpy())

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    task_ids_all = np.array(task_ids_all)

    results = []
    for i, task in enumerate(task_names):
        mask = (task_ids_all == i)
        if not np.any(mask): continue
        
        y_true = y_true_all[mask]
        y_prob = y_prob_all[mask, i]
        
        # Use task-specific threshold
        cutoff = thresholds.get(task, 0.5)
        y_class = (y_prob > cutoff).astype(int)

        try:
            auc = roc_auc_score(y_true, y_prob)
            f1 = f1_score(y_true, y_class)
            acc = accuracy_score(y_true, y_class)
            prec = precision_score(y_true, y_class, zero_division=0)
            rec = recall_score(y_true, y_class, zero_division=0)
            
            results.append({
                "Task": task,
                "Thresh": round(cutoff, 3),
                "AUC-ROC": round(auc, 4),
                "F1-Score": round(f1, 4),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4)
            })
        except ValueError:
            continue

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("--- FINAL ADMET TEST REPORT (SPLIT-BRAIN + OPTIMIZED THRESHOLDS) ---")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n--- Global Averages ---")
    print(df.mean(numeric_only=True).to_frame().T.to_string(index=False))

if __name__ == "__main__":
    MODEL = "admet_predictor_split.pt"
    TEST_SET = "admet_latent_train_bidirec.pt"
    THRESHOLDS = "optimized_thresholds.pt"
    
    evaluate_on_test(MODEL, TEST_SET, THRESHOLDS)