# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# import sys
# import os
# from pathlib import Path

# # Ensure same directory structure access
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset

# # --- CONFIGURATION ---
# FILTERS = {
#     "hERG_inhibition":    {"target": "low",  "threshold": 0.3},
#     "CYP3A4_inhibition":  {"target": "low",  "threshold": 0.4},
#     "BBBP":               {"target": "low",  "threshold": 0.3},
#     "HLM_stability":      {"target": "high", "threshold": 0.6},
#     "RLM_stability":      {"target": "high", "threshold": 0.6},
#     "P-gp_substrate":     {"target": "low",  "threshold": 0.4},
#     "CYP1A2_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C9_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C19_inhibition": {"target": "low",  "threshold": 0.4},
# }

# def run_goal_inference(admet_ckpt, processed_pt_file, output_csv=None, min_passed=4):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Model
#     checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
#     task_names = checkpoint['task_names']
#     model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
#     model.load_state_dict(checkpoint['model_state'])
#     model.eval()

#     # 2. Load Data
#     dataset = LatentDataset(processed_pt_file)
#     loader = DataLoader(dataset, batch_size=512, shuffle=False)

#     all_probs = []
#     with torch.no_grad():
#         for z_batch, _, _ in loader:
#             logits = model(z_batch.to(device))
#             all_probs.append(torch.sigmoid(logits).cpu().numpy())

#     probs_array = np.vstack(all_probs)
#     df_probs = pd.DataFrame(probs_array, columns=task_names)
    
#     # 3. Apply Custom Filter Logic
#     summary = []
#     total_mols = len(df_probs)
    
#     # This array will store how many filters each molecule passed
#     pass_counts_per_mol = np.zeros(total_mols, dtype=int)
#     overall_pass_mask = np.ones(total_mols, dtype=bool)

#     for task in task_names:
#         if task in FILTERS:
#             target = FILTERS[task]['target']
#             thresh = FILTERS[task]['threshold']
            
#             if target == "high":
#                 success_mask = df_probs[task] >= thresh
#             else:
#                 success_mask = df_probs[task] <= thresh
            
#             # Increment the pass count for molecules that succeeded in this task
#             pass_counts_per_mol += success_mask.values.astype(int)
#             overall_pass_mask &= success_mask.values
            
#             summary.append({
#                 "Task": task,
#                 "Goal": f"{target} (<{thresh})" if target == "low" else f"{target} (>{thresh})",
#                 "Success Count": success_mask.sum(),
#                 "Success Rate (%)": round((success_mask.sum() / total_mols) * 100, 2)
#             })

#     # 4. Final Report
#     df_summary = pd.DataFrame(summary)
#     perfect_mols = overall_pass_mask.sum()
    
#     # NEW: Calculate molecules passing at least N filters
#     at_least_n_mols = (pass_counts_per_mol >= min_passed).sum()

#     print("\n" + "="*65)
#     print(f"🎯 DESIGN GOAL REPORT: {Path(processed_pt_file).name}")
#     print("="*65)
#     print(df_summary.to_string(index=False))
#     print("-"*65)
#     print(f"💎 Molecules passing ALL filters:    {perfect_mols} ({round(perfect_mols/total_mols*100, 4)}%)")
#     print(f"✨ Molecules passing >= {min_passed} filters: {at_least_n_mols} ({round(at_least_n_mols/total_mols*100, 4)}%)")
#     print("="*65)

#     if output_csv:
#         df_probs['pass_count'] = pass_counts_per_mol
#         df_probs['meets_all_criteria'] = overall_pass_mask
#         df_probs[f'meets_at_least_{min_passed}'] = (pass_counts_per_mol >= min_passed)
#         df_probs.to_csv(output_csv, index=False)

# if __name__ == "__main__":
#     CKPT = "admet_predictor_bidirec_1000epochs.pt"
#     # CKPT = "admet_predictor.pt"
#     # CKPT = "NewAdmetModels/test1.pt"
#     # benchmarks = ["Benchmarks/exploration_updateMeanVar_CLEAN.pt"]
#     # benchmarks = ["Benchmarks/molobj_pretrained.pt"]
#     benchmarks = [
#         # "Benchmarks/rl_graph_pretrained_num.pt",
#         "Benchmarks/moflow_pretrained.pt",
#         # "Benchmarks/chemvae_4000_pretrained.pt",
#         # "Benchmarks/molobj_pretrained.pt",
#         # "Benchmarks/exploration_updateMeanVar_50update_CLEAN.pt",
#         # "Benchmarks/exploration_updateMeanVar_CLEAN.pt"
#         # "Benchmarks_new/run1.pt"
#     ]
    
#     for bf in benchmarks:
#         if os.path.exists(bf):
#             # Added parameter 'min_passed=4'
#             run_goal_inference(CKPT, bf, output_csv=bf.replace(".pt", "_scores.csv"), min_passed=4)

# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# import sys
# import os
# from pathlib import Path

# # Ensure same directory structure access
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset

# # --- CONFIGURATION ---
# FILTERS = {
#     "hERG_inhibition":    {"target": "low",  "threshold": 0.3},
#     "CYP3A4_inhibition":  {"target": "low",  "threshold": 0.4},
#     "BBBP":               {"target": "low",  "threshold": 0.3},
#     "HLM_stability":      {"target": "high", "threshold": 0.6},
#     "RLM_stability":      {"target": "high", "threshold": 0.6},
#     "P-gp_substrate":     {"target": "low",  "threshold": 0.4},
#     "CYP1A2_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C9_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C19_inhibition": {"target": "low",  "threshold": 0.4},
# }

# def run_goal_inference(admet_ckpt, processed_pt_file, output_csv=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Model
#     checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
#     task_names = checkpoint['task_names']
#     model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
#     model.load_state_dict(checkpoint['model_state'])
#     model.eval()

#     # 2. Load Data
#     dataset = LatentDataset(processed_pt_file)
#     loader = DataLoader(dataset, batch_size=512, shuffle=False)

#     all_probs = []
#     with torch.no_grad():
#         for z_batch, _, _ in loader:
#             logits = model(z_batch.to(device))
#             all_probs.append(torch.sigmoid(logits).cpu().numpy())

#     probs_array = np.vstack(all_probs)
#     df_probs = pd.DataFrame(probs_array, columns=task_names)
#     total_mols = len(df_probs)

#     # 3. Success Mask Logic Helper
#     def get_mask(task_name):
#         if task_name not in FILTERS: return np.ones(total_mols, dtype=bool)
#         target = FILTERS[task_name]['target']
#         thresh = FILTERS[task_name]['threshold']
#         return (df_probs[task_name] >= thresh) if target == "high" else (df_probs[task_name] <= thresh)

#     # 4. Calculate Specific Metrics
#     mask_herg = get_mask("hERG_inhibition")
#     mask_cyp3a4 = get_mask("CYP3A4_inhibition")
#     mask_both = mask_herg & mask_cyp3a4

#     # Calculate Soft Filters (The other 7)
#     soft_tasks = [t for t in FILTERS.keys() if t not in ["hERG_inhibition", "CYP3A4_inhibition"]]
#     soft_pass_counts = np.zeros(total_mols)
#     for t in soft_tasks:
#         soft_pass_counts += get_mask(t).values.astype(int)
    
#     mask_both_and_3_soft = mask_both & (soft_pass_counts >= 3)

#     # 5. Print Detailed Report
#     print("\n" + "="*65)
#     print(f"🎯 CUSTOM GOAL REPORT: {Path(processed_pt_file).name}")
#     print("="*65)
#     print(f"1. Passed hERG:                {mask_herg.sum():>6} ({round(mask_herg.sum()/total_mols*100, 2)}%)")
#     print(f"2. Passed CYP3A4:              {mask_cyp3a4.sum():>6} ({round(mask_cyp3a4.sum()/total_mols*100, 2)}%)")
#     print(f"3. Passed BOTH (1 & 2):        {mask_both.sum():>6} ({round(mask_both.sum()/total_mols*100, 2)}%)")
#     print("-" * 65)
#     print(f"⭐ Passed BOTH + >=3 Soft:     {mask_both_and_3_soft.sum():>6} ({round(mask_both_and_3_soft.sum()/total_mols*100, 4)}%)")
#     print("="*65)

#     if output_csv:
#         df_probs['passed_herg_cyp'] = mask_both
#         df_probs['soft_pass_count'] = soft_pass_counts
#         df_probs['passed_custom_goal'] = mask_both_and_3_soft
#         df_probs.to_csv(output_csv, index=False)

# if __name__ == "__main__":
#     CKPT = "admet_predictor_bidirec_1000epochs.pt"
#     benchmarks = [
#         # "Benchmarks/rl_graph_pretrained_num.pt",
#         # "Benchmarks/moflow_pretrained.pt",
#         # "Benchmarks/chemvae_4000_pretrained.pt",
#         # "Benchmarks/molobj_pretrained.pt",
#         # "Benchmarks/exploration_updateMeanVar_50update_CLEAN.pt",
#         "Benchmarks/exploration_updateMeanVar_CLEAN.pt"
#         # "Benchmarks_new/run1.pt"
#     ]
    
#     for bf in benchmarks:
#         if os.path.exists(bf):
#             run_goal_inference(CKPT, bf, output_csv=bf.replace(".pt", "_scores.csv"))

# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# import sys
# import os
# from pathlib import Path

# # Ensure same directory structure access
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset
# from utils.ScoringEngine import ScoringEngine

# # --- CONFIGURATION ---
# FILTERS = {
#     "hERG_inhibition":    {"target": "low",  "threshold": 0.3},
#     "CYP3A4_inhibition":  {"target": "low",  "threshold": 0.4},
#     "BBBP":               {"target": "low",  "threshold": 0.3},
#     "HLM_stability":      {"target": "high", "threshold": 0.6},
#     "RLM_stability":      {"target": "high", "threshold": 0.6},
#     "P-gp_substrate":     {"target": "low",  "threshold": 0.4},
#     "CYP1A2_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C9_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C19_inhibition": {"target": "low",  "threshold": 0.4},
# }

# def run_goal_inference(admet_ckpt, processed_pt_file, output_csv=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Model
#     checkpoint = torch.load(admet_ckpt, map_location=device, weights_only=False)
#     task_names = checkpoint['task_names']
#     model = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names)).to(device)
#     model.load_state_dict(checkpoint['model_state'])
#     model.eval()

#     # 2. Load Data
#     dataset = LatentDataset(processed_pt_file)
#     loader = DataLoader(dataset, batch_size=512, shuffle=False)

#     all_probs = []
#     with torch.no_grad():
#         for z_batch, _, _ in loader:
#             logits = model(z_batch.to(device))
#             all_probs.append(torch.sigmoid(logits).cpu().numpy())

#     probs_array = np.vstack(all_probs)
#     df_probs = pd.DataFrame(probs_array, columns=task_names)
#     total_mols = len(df_probs)

#     # 3. Success Mask Logic Helper
#     def get_mask(task_name):
#         if task_name not in FILTERS: return np.ones(total_mols, dtype=bool)
#         target = FILTERS[task_name]['target']
#         thresh = FILTERS[task_name]['threshold']
#         return (df_probs[task_name] >= thresh) if target == "high" else (df_probs[task_name] <= thresh)

#     # 4. Calculate Specific Metrics
#     mask_herg = get_mask("hERG_inhibition")
#     mask_cyp3a4 = get_mask("CYP3A4_inhibition")
#     mask_both = mask_herg & mask_cyp3a4

#     soft_tasks = [t for t in FILTERS.keys() if t not in ["hERG_inhibition", "CYP3A4_inhibition"]]
#     soft_pass_counts = np.zeros(total_mols)
#     for t in soft_tasks:
#         soft_pass_counts += get_mask(t).values.astype(int)
    
#     mask_both_and_3_soft = mask_both & (soft_pass_counts >= 3)

#     # Store results for averaging
#     results = {
#         "benchmark": Path(processed_pt_file).name,
#         "total": total_mols,
#         "hERG_pass": mask_herg.sum(),
#         "CYP3A4_pass": mask_cyp3a4.sum(),
#         "Both_pass": mask_both.sum(),
#         "Both_3Soft_pass": mask_both_and_3_soft.sum(),
#         "hERG_pct": mask_herg.sum()/total_mols * 100,
#         "CYP3A4_pct": mask_cyp3a4.sum()/total_mols * 100,
#         "Both_pct": mask_both.sum()/total_mols * 100,
#         "Goal_pct": mask_both_and_3_soft.sum()/total_mols * 100
#     }

#     print(f"Done: {results['benchmark']} | Goal: {results['Goal_pct']:.2f}%")

#     if output_csv:
#         df_probs.to_csv(output_csv, index=False)
    
#     return results

# if __name__ == "__main__":
#     CKPT = "admet_predictor_bidirec_1000epochs.pt"
#     benchmarks = [
#         # "Benchmarks/rl_graph_pretrained_num.pt",
#         "Benchmarks/moflow_pretrained.pt",
#         # "Benchmarks/chemvae_4000_pretrained.pt",
#         # "Benchmarks/molobj_pretrained.pt",
#     ]
    
#     all_results = []
    
#     for bf in benchmarks:
#         if os.path.exists(bf):
#             res = run_goal_inference(CKPT, bf, output_csv=bf.replace(".pt", "_scores.csv"))
#             all_results.append(res)
    
#     # --- FINAL AGGREGATED REPORT ---
#     if all_results:
#         summary_df = pd.DataFrame(all_results)
        
#         print("\n" + "="*85)
#         print(f"{'BENCHMARK SUMMARY':^85}")
#         print("="*85)
#         print(summary_df[["benchmark", "hERG_pct", "CYP3A4_pct", "Both_pct", "Goal_pct"]].to_string(index=False, float_format="%.2f"))
#         print("-" * 85)
        
#         # Calculate Averages
#         avg_h = summary_df["hERG_pct"].mean()
#         avg_c = summary_df["CYP3A4_pct"].mean()
#         avg_b = summary_df["Both_pct"].mean()
#         avg_g = summary_df["Goal_pct"].mean()
        
#         print(f"{'OVERALL AVERAGE':<20} | hERG: {avg_h:.2f}% | CYP: {avg_c:.2f}% | Both: {avg_b:.2f}% | GOAL: {avg_g:.4f}%")
#         print("="*85)

# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# import sys
# import os
# from pathlib import Path

# # Ensure same directory structure access
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Models.AdmetClassifier.train_multitask import MultiHeadADMET, LatentDataset
# from utils.ScoringEngine import ScoringEngine

# # --- CONFIGURATION ---
# FILTERS = {
#     "hERG_inhibition":    {"target": "low",  "threshold": 0.3},
#     "CYP3A4_inhibition":  {"target": "low",  "threshold": 0.4},
#     "BBBP":               {"target": "low",  "threshold": 0.3},
#     "HLM_stability":      {"target": "high", "threshold": 0.6},
#     "RLM_stability":      {"target": "high", "threshold": 0.6},
#     "P-gp_substrate":     {"target": "low",  "threshold": 0.4},
#     "CYP1A2_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C9_inhibition":  {"target": "low",  "threshold": 0.4},
#     "CYP2C19_inhibition": {"target": "low",  "threshold": 0.4},
# }

# def run_goal_inference(admet_ckpt, processed_pt_file, output_csv=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Scoring Engine
#     # Ensure this path is correct for your local setup
#     # ACTIVITY_PATH = "path/to/your/activity_classifier.pt" 
#     ACTIVITY_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
#     engine = ScoringEngine(activity_classifier_path=ACTIVITY_PATH, admet_model_path=admet_ckpt)
    
#     # 2. Load Data
#     dataset = LatentDataset(processed_pt_file)
#     loader = DataLoader(dataset, batch_size=512, shuffle=False)

#     all_score_dicts = []
    
#     # Set models to eval mode via the engine wrappers if they aren't already
#     engine.admet_classifier_model.model.eval()
#     engine.activity_classifier_model.model.eval()

#     with torch.no_grad():
#         for z_batch, _, _ in loader:
#             z_batch = z_batch.to(device)
            
#             # Since classify_admet inside the engine expects a single vector 
#             # and calls .item(), we must iterate through the batch here.
#             for i in range(z_batch.size(0)):
#                 z_individual = z_batch[i] # Shape [latent_dim]
                
#                 # Now this returns the dict of scalars the engine is built for
#                 scores = engine.get_all_scores(z_individual)
#                 all_score_dicts.append(scores)

#     df_probs = pd.DataFrame(all_score_dicts)
#     total_mols = len(df_probs)

#     # 3. Success Mask Logic Helper
#     def get_mask(task_name):
#         if task_name not in FILTERS or task_name not in df_probs.columns: 
#             return np.ones(total_mols, dtype=bool)
#         target = FILTERS[task_name]['target']
#         thresh = FILTERS[task_name]['threshold']
#         return (df_probs[task_name] >= thresh) if target == "high" else (df_probs[task_name] <= thresh)

#     # 4. Calculate Specific Metrics
#     mask_herg = get_mask("hERG_inhibition")
#     mask_cyp3a4 = get_mask("CYP3A4_inhibition")
#     mask_both = mask_herg & mask_cyp3a4

#     soft_tasks = [t for t in FILTERS.keys() if t not in ["hERG_inhibition", "CYP3A4_inhibition"]]
#     soft_pass_counts = np.zeros(total_mols)
#     for t in soft_tasks:
#         soft_pass_counts += get_mask(t).values.astype(int)
    
#     mask_both_and_3_soft = mask_both & (soft_pass_counts >= 3)

#     # Store results for averaging
#     results = {
#         "benchmark": Path(processed_pt_file).name,
#         "total": total_mols,
#         "hERG_pass": mask_herg.sum(),
#         "CYP3A4_pass": mask_cyp3a4.sum(),
#         "Both_pass": mask_both.sum(),
#         "Both_3Soft_pass": mask_both_and_3_soft.sum(),
#         "hERG_pct": mask_herg.sum()/total_mols * 100,
#         "CYP3A4_pct": mask_cyp3a4.sum()/total_mols * 100,
#         "Both_pct": mask_both.sum()/total_mols * 100,
#         "Goal_pct": mask_both_and_3_soft.sum()/total_mols * 100
#     }

#     print(f"Done: {results['benchmark']} | Goal: {results['Goal_pct']:.2f}%")

#     if output_csv:
#         df_probs.to_csv(output_csv, index=False)
    
#     return results

# if __name__ == "__main__":
#     CKPT = "admet_predictor_bidirec_1000epochs.pt"
#     benchmarks = [
#         # "Benchmarks/rl_graph_pretrained_num.pt",
#         "Benchmarks/moflow_pretrained.pt",
#         "Benchmarks/chemvae_4000_pretrained.pt",
#         "Benchmarks/molobj_pretrained.pt",
#         "Benchmarks_new/rnn_smiles.pt"
#         # "Benchmarks/exploration_updateMeanVar_50update_CLEAN.pt",
#         # "Benchmarks/exploration_updateMeanVar_CLEAN.pt"
#         # "Benchmarks_new/run1.pt"
#     ]
    
#     all_results = []
    
#     for bf in benchmarks:
#         if os.path.exists(bf):
#             res = run_goal_inference(CKPT, bf, output_csv=bf.replace(".pt", "_scores.csv"))
#             all_results.append(res)
    
#     # --- FINAL AGGREGATED REPORT ---
#     if all_results:
#         summary_df = pd.DataFrame(all_results)
        
#         print("\n" + "="*85)
#         print(f"{'BENCHMARK SUMMARY':^85}")
#         print("="*85)
#         print(summary_df[["benchmark", "hERG_pct", "CYP3A4_pct", "Both_pct", "Goal_pct"]].to_string(index=False, float_format="%.2f"))
#         print("-" * 85)
        
#         # Calculate Averages
#         avg_h = summary_df["hERG_pct"].mean()
#         avg_c = summary_df["CYP3A4_pct"].mean()
#         avg_b = summary_df["Both_pct"].mean()
#         avg_g = summary_df["Goal_pct"].mean()
        
#         print(f"{'OVERALL AVERAGE':<20} | hERG: {avg_h:.2f}% | CYP: {avg_c:.2f}% | Both: {avg_b:.2f}% | GOAL: {avg_g:.4f}%")
#         print("="*85)

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

def run_bootstrapped_goal_inference(admet_ckpt, processed_pt_file, num_samples=10000, trials=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Models
    ACTIVITY_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
    engine = ScoringEngine(activity_classifier_path=ACTIVITY_PATH, admet_model_path=admet_ckpt)
    engine.admet_classifier_model.model.eval()
    engine.activity_classifier_model.model.eval()

    # 2. Get All Scores First (Inference is slow, do it once)
    dataset = LatentDataset(processed_pt_file)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    print(f"🧬 Extracting scores for {Path(processed_pt_file).name}...")
    all_score_dicts = []
    with torch.no_grad():
        for z_batch, _, _ in loader:
            z_batch = z_batch.to(device)
            for i in range(z_batch.size(0)):
                all_score_dicts.append(engine.get_all_scores(z_batch[i]))

    df_probs = pd.DataFrame(all_score_dicts)
    total_available = len(df_probs)
    
    if total_available < num_samples:
        num_samples = total_available

    # 3. Bootstrapping Loop
    trial_stats = []
    print(f"🧪 Running {trials} trials (N={num_samples})...")

    for _ in range(trials):
        # Sample the dataframe
        df_sample = df_probs.sample(n=num_samples, replace=False)
        
        # Helper to get pass masks
        def get_mask(df, task_name):
            target = FILTERS[task_name]['target']
            thresh = FILTERS[task_name]['threshold']
            return (df[task_name] >= thresh) if target == "high" else (df[task_name] <= thresh)

        # Logic for this trial
        mask_herg = get_mask(df_sample, "hERG_inhibition")
        mask_cyp3a4 = get_mask(df_sample, "CYP3A4_inhibition")
        mask_both = mask_herg & mask_cyp3a4

        soft_tasks = [t for t in FILTERS.keys() if t not in ["hERG_inhibition", "CYP3A4_inhibition"]]
        soft_pass_counts = np.zeros(num_samples)
        for t in soft_tasks:
            soft_pass_counts += get_mask(df_sample, t).values.astype(int)
        
        mask_goal = mask_both & (soft_pass_counts >= 3)

        trial_stats.append({
            "hERG_pct": mask_herg.sum() / num_samples * 100,
            "CYP3A4_pct": mask_cyp3a4.sum() / num_samples * 100,
            "Both_pct": mask_both.sum() / num_samples * 100,
            "Goal_pct": mask_goal.sum() / num_samples * 100
        })

    # 4. Aggregate Stats
    df_trials = pd.DataFrame(trial_stats)
    means = df_trials.mean()
    stds = df_trials.std()

    res = {"benchmark": Path(processed_pt_file).name}
    for col in means.index:
        res[col] = f"{means[col]:.2f} ± {stds[col]:.2f}"
    
    return res

if __name__ == "__main__":
    CKPT = "admet_predictor_bidirec_1000epochs.pt"
    benchmarks = [
        "Benchmarks/moflow_pretrained.pt",
        "Benchmarks/chemvae_4000_pretrained.pt",
        "Benchmarks/molobj_pretrained.pt",
        "Benchmarks_new/rnn_smiles.pt"
    ]
    
    all_results = []
    for bf in benchmarks:
        if os.path.exists(bf):
            res = run_bootstrapped_goal_inference(CKPT, bf, num_samples=5000, trials=10)
            all_results.append(res)
    
    # --- FINAL REPORT ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "="*95)
        print(f"{'BOOTSTRAPPED BENCHMARK SUMMARY (Mean % ± Std Dev)':^95}")
        print("="*95)
        print(summary_df.to_string(index=False))
        print("="*95)