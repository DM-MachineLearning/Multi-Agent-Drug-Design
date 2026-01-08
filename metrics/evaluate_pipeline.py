import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, QED, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import os

def run_full_evaluation(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(filepath)
    # Ensure we only use rows with actual SMILES
    valid_smiles = df[df['smiles'].str.len() > 5]['smiles'].tolist()
    mols = [Chem.MolFromSmiles(s) for s in valid_smiles if Chem.MolFromSmiles(s)]
    
    if not mols:
        print("❌ No valid molecules found to evaluate.")
        return

    print(f"\n{'='*60}")
    print(f"       DRUG DISCOVERY PIPELINE EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total Successful Leads Analyzed: {len(mols)}")

    # --- METRIC 1: DRUG-LIKENESS (QED) ---
    qed_scores = [QED.qed(m) for m in mols]
    avg_qed = np.mean(qed_scores)
    
    # --- METRIC 2: MOLECULAR WEIGHT & LOGP ---
    mws = [Descriptors.MolWt(m) for m in mols]
    logps = [Descriptors.MolLogP(m) for m in mols]

    # --- METRIC 3: SCAFFOLD DIVERSITY ---
    scaffs = [MurckoScaffold.GetScaffoldForMol(m) for m in mols]
    scaff_smis = [Chem.MolToSmiles(s) for s in scaffs if s]
    unique_scaffolds = len(set(scaff_smis))
    div_ratio = unique_scaffolds / len(mols)

    # --- METRIC 4: INTERNAL SIMILARITY (Tanimoto) ---
    # Sampling 100 pairs if the set is too large to save time
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    sims = []
    sample_size = min(len(fps), 200) 
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    avg_sim = np.mean(sims) if sims else 0

    # --- OUTPUT REPORT ---
    print(f"\n[1] CHEMICAL QUALITY")
    print(f"    - Avg Druglikeness (QED): {avg_qed:.3f} (Ideal: > 0.600)")
    print(f"    - Avg Mol Weight:         {np.mean(mws):.1f} Da")
    print(f"    - Avg LogP:               {np.mean(logps):.2f}")
    
    print(f"\n[2] DIVERSITY & NOVELTY")
    print(f"    - Scaffold Diversity:     {div_ratio:.2f} ({unique_scaffolds} unique cores)")
    print(f"    - Internal Similarity:    {avg_sim:.3f} (0.0=Diverse, 1.0=Redundant)")
    
    # ASCII Histogram for QED
    print(f"\n[3] QED DISTRIBUTION")
    hist, bins = np.histogram(qed_scores, bins=5, range=(0, 1))
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / (max(hist) if max(hist) > 0 else 1) * 30)
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({hist[i]})")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Update with your actual CSV path
    run_full_evaluation("outputs/exploration_updateMeanVar.csv")
    # run_full_evaluation("outputs/run1.csv")