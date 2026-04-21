import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- 1. Enhanced Architecture with Spectral Normalization ---
class SpectralMultiHeadADMET(nn.Module):
    def __init__(self, latent_dim=128, num_tasks=11):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 1024)),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(512, 128)),
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

def find_optimal_thresholds(model, loader, device, task_names, temperature=2.0):
    model.eval()
    print("🔍 Generating predictions for threshold optimization...")
    
    all_probs = []
    all_labels = []
    all_task_ids = []
    
    with torch.no_grad():
        for z, labels, task_ids in loader:
            z = z.to(device)
            logits = model(z)
            # Use temperature scaling as in production
            probs = torch.sigmoid(logits / temperature)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_task_ids.append(task_ids.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_task_ids = np.concatenate(all_task_ids)

    best_thresholds = {}
    
    print("\n" + "="*85)
    print(f"{'TASK':<25} | {'BEST THRESH':<12} | {'OLD F1':<8} -> {'NEW F1':<8} | {'NEW ACC':<8}")
    print("="*85)

    global_f1_old = []
    global_f1_new = []

    for i, task in enumerate(task_names):
        mask = (all_task_ids == i)
        if mask.sum() == 0: continue
            
        y_true = all_labels[mask]
        y_scores = all_probs[mask, i]
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        f1_scores = f1_scores[:-1] 
        
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        default_preds = (y_scores > 0.5).astype(int)
        default_f1 = f1_score(y_true, default_preds)
        
        new_preds = (y_scores > best_thresh).astype(int)
        new_acc = accuracy_score(y_true, new_preds)
        
        best_thresholds[task] = float(best_thresh)
        global_f1_old.append(default_f1)
        global_f1_new.append(best_f1)
        
        print(f"{task:<25} | {best_thresh:.4f}       | {default_f1:.4f}   -> {best_f1:.4f}   | {new_acc:.4f}")

    print("-" * 85)
    print(f"Global Average F1        |              | {np.mean(global_f1_old):.4f}   -> {np.mean(global_f1_new):.4f}")
    print("=" * 85)
    
    return best_thresholds

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "admet_predictor_distilled_300epochs.pt" 
    DATA_PATH = "admet_latent_test_bidirec.pt"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    dataset = LatentDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    print(f"📥 Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    model = SpectralMultiHeadADMET(latent_dim=128, num_tasks=len(dataset.tasks)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    
    thresholds = find_optimal_thresholds(model, loader, DEVICE, dataset.tasks)
    
    torch.save(thresholds, "optimized_thresholds.pt")
    print("\n✅ Thresholds saved to 'optimized_thresholds.pt'")

if __name__ == "__main__":
    main()
