import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Ensure same directory structure access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset

# --- CONFIGURATION ---
FILTERS = {
    "hERG_inhibition":    {"target": "low",  "threshold": 0.3},
    "CYP3A4_inhibition":  {"target": "low",  "threshold": 0.4},
    "BBBP":               {"target": "low",  "threshold": 0.3},
    "HLM_stability":      {"target": "high", "threshold": 0.6},
    "RLM_stability":      {"target": "high", "threshold": 0.6},
    "P-gp_substrate":     {"target": "low",  "threshold": 0.4},
    "CYP1A2_inhibition":  {"target": "low",  "threshold": 0.4},
    "CYP2C9_inhibition":  {"target": "low",  "threshold": 0.4},
    "CYP2C19_inhibition": {"target": "low",  "threshold": 0.4},
}

def run_goal_inference(admet_ckpt, processed_pt_file, output_csv=None, min_passed=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
    task_names = checkpoint['task_names']
    model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 2. Load Data
    dataset = LatentDataset(processed_pt_file)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for z_batch, _, _ in loader:
            logits = model(z_batch.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs_array = np.vstack(all_probs)
    df_probs = pd.DataFrame(probs_array, columns=task_names)
    
    # 3. Apply Custom Filter Logic
    summary = []
    total_mols = len(df_probs)
    
    # This array will store how many filters each molecule passed
    pass_counts_per_mol = np.zeros(total_mols, dtype=int)
    overall_pass_mask = np.ones(total_mols, dtype=bool)

    for task in task_names:
        if task in FILTERS:
            target = FILTERS[task]['target']
            thresh = FILTERS[task]['threshold']
            
            if target == "high":
                success_mask = df_probs[task] >= thresh
            else:
                success_mask = df_probs[task] <= thresh
            
            # Increment the pass count for molecules that succeeded in this task
            pass_counts_per_mol += success_mask.values.astype(int)
            overall_pass_mask &= success_mask.values
            
            summary.append({
                "Task": task,
                "Goal": f"{target} (<{thresh})" if target == "low" else f"{target} (>{thresh})",
                "Success Count": success_mask.sum(),
                "Success Rate (%)": round((success_mask.sum() / total_mols) * 100, 2)
            })

    # 4. Final Report
    df_summary = pd.DataFrame(summary)
    perfect_mols = overall_pass_mask.sum()
    
    # NEW: Calculate molecules passing at least N filters
    at_least_n_mols = (pass_counts_per_mol >= min_passed).sum()

    print("\n" + "="*65)
    print(f"🎯 DESIGN GOAL REPORT: {Path(processed_pt_file).name}")
    print("="*65)
    print(df_summary.to_string(index=False))
    print("-"*65)
    print(f"💎 Molecules passing ALL filters:    {perfect_mols} ({round(perfect_mols/total_mols*100, 4)}%)")
    print(f"✨ Molecules passing >= {min_passed} filters: {at_least_n_mols} ({round(at_least_n_mols/total_mols*100, 4)}%)")
    print("="*65)

    if output_csv:
        df_probs['pass_count'] = pass_counts_per_mol
        df_probs['meets_all_criteria'] = overall_pass_mask
        df_probs[f'meets_at_least_{min_passed}'] = (pass_counts_per_mol >= min_passed)
        df_probs.to_csv(output_csv, index=False)

if __name__ == "__main__":
    CKPT = "admet_predictor_bidirec_1000epochs.pt"
    # benchmarks = ["Benchmarks/exploration_updateMeanVar_CLEAN.pt"]
    # benchmarks = ["Benchmarks/molobj_pretrained.pt"]
    benchmarks = [
        # "Benchmarks/rl_graph_pretrained_num.pt",
        # "Benchmarks/moflow_pretrained.pt",
        # "Benchmarks/chemvae_4000_pretrained.pt",
        # "Benchmarks/molobj_pretrained.pt",
        "Benchmarks/exploration_updateMeanVar_50update_CLEAN.pt",
        # "Benchmarks/exploration_updateMeanVar_CLEAN.pt"
    ]
    
    for bf in benchmarks:
        if os.path.exists(bf):
            # Added parameter 'min_passed=4'
            run_goal_inference(CKPT, bf, output_csv=bf.replace(".pt", "_scores.csv"), min_passed=4)