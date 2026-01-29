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
    EPOCHS = 100
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

    torch.save({"model_state": model.state_dict(), "task_names": dataset.tasks}, "NewAdmetModels/test1.pt")
    print("✅ Optimized Model Saved!")

if __name__ == "__main__":
    train()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import copy

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss) # prevents nans when probability is 0
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         return F_loss

# # --- 1. Balanced Architecture (Residuals + Moderate Capacity) ---
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
#         return x + self.block(x) # Skip Connection

# class MultiHeadADMET(nn.Module):
#     def __init__(self, latent_dim=128, num_tasks=11):
#         super().__init__()
        
#         # Increased capacity slightly, but using Residuals for stability
#         self.input_proj = nn.Linear(latent_dim, 512)
        
#         self.shared_encoder = nn.Sequential(
#             ResidualBlock(512, dropout=0.2), # Lower dropout (was 0.5)
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
#         # Reduced Noise Injection (0.1 -> 0.05)
#         if self.training:
#             noise = torch.randn_like(z) * 0.05 
#             z = z + noise
            
#         x = self.input_proj(z)
#         features = self.shared_encoder(x)
#         outputs = [head(features) for head in self.heads]
#         return torch.cat(outputs, dim=1)

# # --- 2. Dataset Loader (Unchanged) ---
# class LatentDataset(Dataset):
#     def __init__(self, processed_path):
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

# # --- 3. AUTO-CALCULATE WEIGHTS ---
# def calculate_pos_weights(dataset):
#     print("⚖️  Calculating optimal class weights...")
#     num_tasks = len(dataset.tasks)
    
#     # Initialize counters
#     pos_counts = torch.zeros(num_tasks)
#     neg_counts = torch.zeros(num_tasks)
    
#     # Iterate once to count (fast enough for 78k samples)
#     loader = DataLoader(dataset, batch_size=4096, shuffle=False)
#     for _, labels, task_ids in loader:
#         # labels: [B], task_ids: [B]
#         for t in range(num_tasks):
#             mask = (task_ids == t)
#             if mask.sum() > 0:
#                 task_labels = labels[mask]
#                 pos_counts[t] += (task_labels == 1).sum()
#                 neg_counts[t] += (task_labels == 0).sum()
    
#     # Calculate weights: pos_weight = negative / positive
#     # If 100 negatives and 10 positives, weight should be 10.
#     pos_weights = neg_counts / (pos_counts + 1e-6) # Avoid div by zero
    
#     print("📊 Calculated Weights per Task:")
#     for t, task in enumerate(dataset.tasks):
#         print(f"  - {task:<20}: {pos_weights[t]:.4f} (Pos: {int(pos_counts[t])}, Neg: {int(neg_counts[t])})")
        
#     return pos_weights

# # --- 4. Training ---
# def train():
#     BATCH_SIZE = 1024
#     MAX_EPOCHS = 300 
#     PATIENCE = 20
#     LR = 1e-3 
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     full_dataset = LatentDataset("admet_latent_train_bidirec.pt")
    
#     # --- GET DYNAMIC WEIGHTS ---
#     # This vector has shape [11], one specific weight for each task
#     task_weights = calculate_pos_weights(full_dataset).to(DEVICE)
    
#     train_size = int(0.85 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     model = MultiHeadADMET(latent_dim=full_dataset.latent_dim, num_tasks=len(full_dataset.tasks)).to(DEVICE)
#     optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
#     # Standard BCE - we handle weights manually in the loop because
#     # pos_weight argument in constructor expects fixed batch shape usually, 
#     # but our batch shape changes because we flatten tasks.
#     # We will apply weights inside the loop for precision.
#     # criterion = nn.BCEWithLogitsLoss(reduction='none') 
#     criterion = FocalLoss(alpha=0.6, gamma=2.0).to(DEVICE) # Alpha > 0.5 gives slightly more weight to positives

#     print(f"🚀 Training | Train Size: {train_size} | Val Size: {val_size}")

#     best_val_loss = float('inf')
#     best_model_wts = copy.deepcopy(model.state_dict())
#     patience_counter = 0

#     for epoch in range(MAX_EPOCHS):
#         model.train()
#         train_loss = 0
        
#         for z, label, task_ids in train_loader:
#             z, label, task_ids = z.to(DEVICE), label.to(DEVICE), task_ids.to(DEVICE)
            
#             optimizer.zero_grad()
#             all_preds = model(z)
#             target_preds = all_preds[torch.arange(z.size(0)), task_ids]
            
#             # 1. Apply Dynamic Weights
#             # Select the weight corresponding to the task of each sample
#             sample_weights = task_weights[task_ids]
            
#             # 2. Compute Loss per sample
#             loss_per_sample = criterion(target_preds, label)
            
#             # 3. Weight positive samples
#             # BCEWithLogitsLoss formula: weight * y * log(sigma) + ...
#             # We manually multiply the loss of positive samples by our calculated weight
#             weighted_loss = loss_per_sample * (label * sample_weights + (1 - label))
            
#             loss = weighted_loss.mean()
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
#             optimizer.step()
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)

#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for z, label, task_ids in val_loader:
#                 z, label, task_ids = z.to(DEVICE), label.to(DEVICE), task_ids.to(DEVICE)
#                 all_preds = model(z)
#                 target_preds = all_preds[torch.arange(z.size(0)), task_ids]
                
#                 # Use same weighting logic for validation so metric is comparable
#                 sample_weights = task_weights[task_ids]
#                 loss_per_sample = criterion(target_preds, label)
#                 weighted_loss = loss_per_sample * (label * sample_weights + (1 - label))
                
#                 val_loss += weighted_loss.mean().item()
        
#         avg_val_loss = val_loss / len(val_loader)
        
#         scheduler.step(avg_val_loss)
#         current_lr = optimizer.param_groups[0]['lr']

#         if (epoch+1) % 5 == 0:
#             print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= PATIENCE:
#                 print(f"🛑 Early stopping at epoch {epoch+1}")
#                 break

#     model.load_state_dict(best_model_wts)
#     torch.save({"model_state": model.state_dict(), "task_names": full_dataset.tasks}, "admet_predictor_weights_focal_loss.pt")
#     print("✅ Balanced Model Saved!")

# if __name__ == "__main__":
#     train()

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# from pathlib import Path
# from tqdm import tqdm
# from transformers import PreTrainedTokenizerFast

# # --- 1. MODEL ARCHITECTURE (SMILES Encoder) ---
# class SMILESMultiTaskClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_tasks=11):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         # Bidirectional GRU to extract chemical context from text
#         self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        
#         self.shared_body = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 512),
#             nn.BatchNorm1d(512),
#             nn.SiLU(),
#             nn.Dropout(0.3)
#         )
        
#         # One head for each ADMET task found in your folders
#         self.heads = nn.ModuleList([
#             nn.Sequential(nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 1)) 
#             for _ in range(num_tasks)
#         ])

#     def forward(self, x, lengths):
#         x = self.embedding(x)
#         # Pack sequence to ignore padding and fix the drift issue
#         packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, h_n = self.gru(packed_x)
        
#         # Concatenate forward and backward hidden states from the last layer
#         features = torch.cat((h_n[-2], h_n[-1]), dim=1)
#         shared = self.shared_body(features)
        
#         return torch.cat([head(shared) for head in self.heads], dim=1)

# # --- 2. DATASET LOADER (Folder-based) ---
# class FolderADMETDataset(Dataset):
#     def __init__(self, data_root, tokenizer, max_len=128):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.data = []
        
#         root = Path(data_root)
#         self.tasks = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')])
#         self.task_to_idx = {name: i for i, name in enumerate(self.tasks)}

#         print(f"📂 Loading folders from {data_root}...")
#         for task in self.tasks:
#             task_dir = root / task
#             csv_files = list(task_dir.glob("*_train_set.csv")) # Change to _test_set.csv for evaluation
#             if not csv_files: continue
            
#             df = pd.read_csv(csv_files[0])
#             label_col = 'bioclass' if 'bioclass' in df.columns else df.columns[-1]
            
#             for _, row in df.iterrows():
#                 self.data.append({
#                     'smiles': str(row['SMILES']),
#                     'label': float(row[label_col]),
#                     'task_idx': self.task_to_idx[task]
#                 })
#         print(f"✅ Total samples: {len(self.data)} across {len(self.tasks)} tasks.")

#     def __len__(self): return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         tokens = self.tokenizer.encode(item['smiles'], add_special_tokens=True)
#         if len(tokens) > self.max_len: tokens = tokens[:self.max_len]
#         return torch.tensor(tokens), torch.tensor(item['label']), item['task_idx']

# def collate_fn(batch):
#     # Sort by length for packing
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     tokens, labels, task_ids = zip(*batch)
#     lengths = torch.tensor([len(x) for x in tokens])
#     tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
#     return tokens_padded, lengths, torch.tensor(labels), torch.tensor(task_ids)

# # --- 3. TRAINING SCRIPT ---
# def train():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     VOCAB_PATH = "./vocab.json"
#     DATA_ROOT = "./Models/AdmetClassifier/Auto_ML_dataset"
    
#     tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_PATH)
#     if tokenizer.pad_token_id is None: tokenizer.pad_token_id = 0
    
#     dataset = FolderADMETDataset(DATA_ROOT, tokenizer)
#     loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
#     model = SMILESMultiTaskClassifier(len(tokenizer), num_tasks=len(dataset.tasks)).to(DEVICE)
#     optimizer = optim.AdamW(model.parameters(), lr=5e-4)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(DEVICE))

#     print("🚀 Starting SMILES-based Multi-Task Training...")
#     for epoch in range(20):
#         model.train()
#         epoch_loss = 0
#         for x, lengths, y, t_idx in tqdm(loader, desc=f"Epoch {epoch+1}"):
#             x, y, t_idx = x.to(DEVICE), y.to(DEVICE), t_idx.to(DEVICE)
            
#             optimizer.zero_grad()
#             preds = model(x, lengths)
            
#             # Select the correct head for the task
#             task_preds = preds[torch.arange(x.size(0)), t_idx]
#             loss = criterion(task_preds, y)
            
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
            
#         print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")

#     torch.save({"state": model.state_dict(), "tasks": dataset.tasks}, "smiles_admet_model.pt")

# if __name__ == "__main__":
#     train()