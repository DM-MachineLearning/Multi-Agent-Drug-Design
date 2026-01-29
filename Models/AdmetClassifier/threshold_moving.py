# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score

# # ==========================================
# # 1. MODEL ARCHITECTURE (Must match saved model)
# # ==========================================
# class ResidualBlock(nn.Module):
#     def __init__(self, hidden_dim, dropout):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.SiLU(),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return x + self.block(x)

# class MultiHeadADMET(nn.Module):
#     def __init__(self, latent_dim=128, num_tasks=11):
#         super().__init__()
#         self.input_proj = nn.Linear(latent_dim, 512)
#         self.shared_encoder = nn.Sequential(
#             ResidualBlock(512, dropout=0.2),
#             ResidualBlock(512, dropout=0.2),
#             nn.Linear(512, 256),
#             nn.SiLU()
#         )
#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(256, 128),
#                 nn.SiLU(),
#                 nn.Linear(128, 1)
#             ) for _ in range(num_tasks)
#         ])
        
#     def forward(self, z):
#         # No noise needed for inference/tuning
#         x = self.input_proj(z)
#         features = self.shared_encoder(x)
#         outputs = [head(features) for head in self.heads]
#         return torch.cat(outputs, dim=1)

# class LatentDataset(Dataset):
#     def __init__(self, processed_path):
#         # Added weights_only=False fix
#         data_pack = torch.load(processed_path, weights_only=False)
#         self.samples = data_pack['data']
#         self.tasks = data_pack['tasks']
#         self.latent_dim = data_pack['latent_dim']

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         return (
#             torch.tensor(item['z'], dtype=torch.float32), 
#             torch.tensor(item['y'], dtype=torch.float32),
#             torch.tensor(item['task_idx'], dtype=torch.long)
#         )

# # ==========================================
# # 2. CORE OPTIMIZATION FUNCTION
# # ==========================================
# def find_optimal_thresholds(model, loader, device, task_names):
#     model.eval()
#     print("🔍 Generating predictions on validation set...")
    
#     all_probs = []
#     all_labels = []
#     all_task_ids = []
    
#     with torch.no_grad():
#         for z, labels, task_ids in loader:
#             z = z.to(device)
            
#             # Get logits
#             logits = model(z)
#             # Convert to Probabilities (0 to 1)
#             probs = torch.sigmoid(logits)
            
#             # Select the head corresponding to the task
#             target_probs = probs[torch.arange(z.size(0)), task_ids]
            
#             all_probs.append(target_probs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#             all_task_ids.append(task_ids.cpu().numpy())

#     all_probs = np.concatenate(all_probs)
#     all_labels = np.concatenate(all_labels)
#     all_task_ids = np.concatenate(all_task_ids)

#     best_thresholds = {}
    
#     print("\n" + "="*80)
#     print(f"{'TASK':<20} | {'BEST THRESH':<12} | {'OLD F1':<8} -> {'NEW F1':<8} | {'ACCURACY':<8}")
#     print("="*80)

#     global_f1_old = []
#     global_f1_new = []

#     for i, task in enumerate(task_names):
#         # Filter data for this specific task
#         mask = (all_task_ids == i)
#         if mask.sum() == 0:
#             continue
            
#         y_true = all_labels[mask]
#         y_scores = all_probs[mask]
        
#         # 1. Get Precision-Recall Curve
#         precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
#         # 2. Calculate F1 for every single threshold
#         # Note: thresholds array is 1 shorter than precision/recall arrays
#         f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
#         # Drop the last F1 score to match thresholds shape
#         f1_scores = f1_scores[:-1] 
        
#         # 3. Find the best one
#         best_idx = np.argmax(f1_scores)
#         best_thresh = thresholds[best_idx]
#         best_f1 = f1_scores[best_idx]
        
#         # 4. Compare with default 0.5
#         default_preds = (y_scores > 0.5).astype(int)
#         default_f1 = f1_score(y_true, default_preds)
        
#         # 5. Calculate New Accuracy
#         new_preds = (y_scores > best_thresh).astype(int)
#         new_acc = accuracy_score(y_true, new_preds)
        
#         best_thresholds[task] = float(best_thresh)
#         global_f1_old.append(default_f1)
#         global_f1_new.append(best_f1)
        
#         print(f"{task:<20} | {best_thresh:.4f}       | {default_f1:.4f}   -> {best_f1:.4f}   | {new_acc:.4f}")

#     print("-" * 80)
#     print(f"Global Average F1  |              | {np.mean(global_f1_old):.4f}   -> {np.mean(global_f1_new):.4f}")
#     print("=" * 80)
    
#     return best_thresholds

# # ==========================================
# # 3. MAIN EXECUTION
# # ==========================================
# def main():
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     MODEL_PATH = "admet_predictor_split.pt"
#     DATA_PATH = "admet_latent_train_bidirec.pt"
    
#     # 1. Load Data
#     full_dataset = LatentDataset(DATA_PATH)
    
#     # IMPORTANT: Use the exact same seed/split logic as training 
#     # to ensure we tune on Validation data, not Training data.
#     train_size = int(0.85 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
    
#     # We set a seed here to try and replicate the random_split from training
#     # (If you didn't set a seed in training, this will be an approximation, 
#     #  but still valid for finding thresholds)
#     generator = torch.Generator().manual_seed(42) 
#     _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
#     val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
#     # 2. Load Model
#     print(f"📥 Loading model from {MODEL_PATH}...")
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
#     model = MultiHeadADMET(latent_dim=full_dataset.latent_dim, num_tasks=len(full_dataset.tasks)).to(DEVICE)
#     model.load_state_dict(checkpoint["model_state"])
    
#     # 3. Run Optimization
#     thresholds = find_optimal_thresholds(model, val_loader, DEVICE, full_dataset.tasks)
    
#     # 4. Save Thresholds
#     torch.save(thresholds, "optimized_thresholds.pt")
#     print("\n✅ Thresholds saved to 'optimized_thresholds.pt'")
#     print("Use these values in your final inference pipeline instead of 0.5!")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
import os

# ==========================================
# 1. MODEL ARCHITECTURE (Grouped / Split-Brain)
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
        # Task indices based on the specific ADMET task order
        # CYPs: CYP1A2, CYP2C19, CYP2C9, CYP2D6, CYP3A4
        self.cyp_indices = [1, 2, 3, 4, 5] 
        # Others: BBBP, Caco2, HLM, P-gp, RLM, hERG
        self.prop_indices = [0, 6, 7, 8, 9, 10]
        
        # Specialist Encoder for Enzyme binding (CYPs)
        self.cyp_encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2),
            ResidualBlock(512, 0.2), 
            nn.Linear(512, 256),
            nn.SiLU()
        )
        
        # Specialist Encoder for Physical/Property features
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

# ==========================================
# 2. DATASET LOADER
# ==========================================
class LatentDataset(Dataset):
    def __init__(self, processed_path):
        # Using weights_only=False because of how standard .pt files are saved
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
# 3. THRESHOLD OPTIMIZATION LOGIC
# ==========================================
def find_optimal_thresholds(model, loader, device, task_names):
    model.eval()
    print("🔍 Generating predictions on validation set...")
    
    all_probs = []
    all_labels = []
    all_task_ids = []
    
    with torch.no_grad():
        for z, labels, task_ids in loader:
            z = z.to(device)
            logits = model(z)
            probs = torch.sigmoid(logits)
            
            # Extract probability for the specific task assigned to this sample
            target_probs = probs[torch.arange(z.size(0)), task_ids]
            
            all_probs.append(target_probs.cpu().numpy())
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
        y_scores = all_probs[mask]
        
        # Precision-Recall curve to find all possible thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 for every threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        f1_scores = f1_scores[:-1] # Match thresholds length
        
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Comparison with baseline 0.5
        default_preds = (y_scores > 0.5).astype(int)
        default_f1 = f1_score(y_true, default_preds)
        
        # New accuracy check
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

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "admet_predictor_split.pt" 
    DATA_PATH = "admet_latent_train_bidirec.pt"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found. Train the split model first!")
        return

    full_dataset = LatentDataset(DATA_PATH)
    
    # Replicate the 85/15 validation split used in training
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42) 
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    print(f"📥 Loading split-brain model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Initialize with GroupedADMET architecture
    model = GroupedADMET(latent_dim=full_dataset.latent_dim, all_tasks=full_dataset.tasks).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    
    # Run Optimization
    thresholds = find_optimal_thresholds(model, val_loader, DEVICE, full_dataset.tasks)
    
    # Save optimized thresholds
    torch.save(thresholds, "optimized_thresholds.pt")
    print("\n✅ Thresholds saved to 'optimized_thresholds.pt'")

if __name__ == "__main__":
    main()