import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import sys
import os

# Ensure same directory structure access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset

def evaluate_metrics(admet_ckpt, processed_val_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    checkpoint = torch.load(admet_ckpt, map_location=device)
    task_names = checkpoint['task_names']
    model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 2. Load Validation Data
    # Note: You should create a 'admet_latent_val.pt' using your valid_sets
    dataset = LatentDataset(processed_val_data)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # Containers for results
    y_true_all = []
    y_pred_all = []
    task_ids_all = []

    print(f"🧪 Evaluating {len(dataset)} samples across {len(task_names)} tasks...")

    with torch.no_grad():
        for z_batch, label_batch, task_ids in loader:
            z_batch = z_batch.to(device)
            
            # Get logits and convert to probabilities
            logits = model(z_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            y_true_all.extend(label_batch.numpy())
            y_pred_all.extend(probs)
            task_ids_all.extend(task_ids.numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    task_ids_all = np.array(task_ids_all)

    # 3. Calculate Metrics per Task
    results = []
    for i, task in enumerate(task_names):
        # Mask for current task
        mask = (task_ids_all == i)
        if not np.any(mask): continue
        
        y_true = y_true_all[mask]
        # Pick the column for this specific head
        y_prob = y_pred_all[mask, i]
        y_class = (y_prob > 0.5).astype(int)

        # Calculate metrics
        try:
            auc = roc_auc_score(y_true, y_prob)
            acc = accuracy_score(y_true, y_class)
            f1 = f1_score(y_true, y_class)
            prec = precision_score(y_true, y_class, zero_division=0)
            rec = recall_score(y_true, y_class, zero_division=0)
            
            results.append({
                "Task": task,
                "AUC-ROC": round(auc, 4),
                "F1-Score": round(f1, 4),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4)
            })
        except ValueError:
            # Handles cases where only one class is present in the batch
            continue

    # 4. Display results
    df_results = pd.DataFrame(results)
    print("\n--- ADMET MODEL PERFORMANCE REPORT ---")
    print(df_results.to_string(index=False))
    
    # Calculate Global Average
    print("\n--- Global Averages ---")
    print(df_results.mean(numeric_only=True).to_frame().T.to_string(index=False))

if __name__ == "__main__":
    # You need to run create_multitask_dataset.py on your VALIDATION files 
    # and save it as admet_latent_val.pt first!
    evaluate_metrics("admet_predictor_bidirec_1000epochs.pt", "admet_latent_test_bidirec.pt")