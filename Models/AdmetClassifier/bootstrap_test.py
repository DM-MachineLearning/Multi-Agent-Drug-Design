import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import sys
import os
from pathlib import Path

# Ensure same directory structure access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset
from utils.ScoringEngine import ScoringEngine

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

def run_bootstrapped_goal_inference(admet_ckpt, processed_pt_file, num_samples=1000, trials=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Models
    ACTIVITY_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
    engine = ScoringEngine(activity_classifier_path=ACTIVITY_PATH, admet_model_path=admet_ckpt)
    engine.admet_classifier_model.model.eval()
    engine.activity_classifier_model.model.eval()

    # 2. Load Dataset (but don't run inference yet)
    full_dataset = LatentDataset(processed_pt_file)
    total_available = len(full_dataset)
    
    if total_available < num_samples:
        num_samples = total_available

    trial_stats = []

    for trial in range(trials):
        print(f"🧪 Trial {trial+1}/{trials} for {Path(processed_pt_file).name}...")
        
        # 3. Random Sampling of Indices
        indices = np.random.choice(total_available, num_samples, replace=False)
        subset = Subset(full_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False)

        trial_scores = []
        with torch.no_grad():
            for z_batch, _, _ in loader:
                z_batch = z_batch.to(device)
                for i in range(z_batch.size(0)):
                    # Inference ONLY on the selected 10k samples
                    trial_scores.append(engine.get_all_scores(z_batch[i]))

        df_trial = pd.DataFrame(trial_scores)

        # 4. Success Mask Logic for this trial
        def get_mask(df, task_name):
            target = FILTERS[task_name]['target']
            thresh = FILTERS[task_name]['threshold']
            return (df[task_name] >= thresh) if target == "high" else (df[task_name] <= thresh)

        mask_herg = get_mask(df_trial, "hERG_inhibition")
        mask_cyp3a4 = get_mask(df_trial, "CYP3A4_inhibition")
        mask_both = mask_herg & mask_cyp3a4

        soft_tasks = [t for t in FILTERS.keys() if t not in ["hERG_inhibition", "CYP3A4_inhibition"]]
        soft_pass_counts = np.zeros(len(df_trial))
        for t in soft_tasks:
            soft_pass_counts += get_mask(df_trial, t).values.astype(int)
        
        mask_goal = mask_both & (soft_pass_counts >= 3)

        trial_stats.append({
            "hERG_pct": mask_herg.sum() / len(df_trial) * 100,
            "CYP3A4_pct": mask_cyp3a4.sum() / len(df_trial) * 100,
            "Both_pct": mask_both.sum() / len(df_trial) * 100,
            "Goal_pct": mask_goal.sum() / len(df_trial) * 100
        })

    # 5. Aggregate Across Trials
    df_results = pd.DataFrame(trial_stats)
    means = df_results.mean()
    stds = df_results.std()

    res_row = {"benchmark": Path(processed_pt_file).name}
    for col in means.index:
        res_row[col] = f"{means[col]:.2f} ± {stds[col]:.2f}"
    
    return res_row

if __name__ == "__main__":
    CKPT = "admet_predictor_bidirec_1000epochs.pt"
    benchmarks = [
        # "Benchmarks/rl_graph_pretrained_num.pt",
        # "Benchmarks/moflow_pretrained.pt",
        # "Benchmarks/chemvae_4000_pretrained.pt",
        # "Benchmarks/molobj_pretrained.pt",
        # "Benchmarks_new/rnn_smiles.pt"
        # "Benchmarks_new/molgan_15k_smiles.pt"
        # "Benchmarks_new/bimodal.pt"
        # "Benchmarks_new/organ.pt"
        "Benchmarks_new/generated1.pt"
    ]
    
    all_final_results = []
    for bf in benchmarks:
        if os.path.exists(bf):
            res = run_bootstrapped_goal_inference(CKPT, bf, num_samples=500, trials=5)
            all_final_results.append(res)
    
    if all_final_results:
        summary_df = pd.DataFrame(all_final_results)
        print("\n" + "="*100)
        print(f"{'BOOTSTRAPPED GOAL SUMMARY (Mean % ± Std Dev)':^100}")
        print("="*100)
        print(summary_df.to_string(index=False))
        print("="*100)