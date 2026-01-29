import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import os
import re

# --- CONFIG ---
# GENERATED_FILE = "outputs/successful_molecules_CLEAN.csv"
# GENERATED_FILE = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
GENERATED_FILE = "outputs/successful_molecules_scaffold.csv"
TRAINING_DATA_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (2).xlsx"
ACTIVITY_THRESHOLD = 6.0  # Only compare against molecules with pIC50 >= 6.0

def plot_tanimoto_actives_only():
    if not os.path.exists(GENERATED_FILE):
        print(f"❌ Error: {GENERATED_FILE} not found.")
        return

    # 1. LOAD DATA
    print("📂 Loading datasets...")
    df_gen = pd.read_csv(GENERATED_FILE)
    df_gen = df_gen[df_gen['smiles'] != 'Latent_Vector_Only']
    gen_smiles = df_gen['smiles'].tolist()

    print(f"   - Generated Candidates: {len(gen_smiles)}")

    # 2. LOAD & FILTER TRAINING DATA
    print(f"📂 Reading Training Data: {TRAINING_DATA_PATH}...")
    try:
        df_train = pd.read_excel(TRAINING_DATA_PATH)
        
        # Auto-detect columns
        # Look for 'smile' in column names
        smi_col = next((c for c in df_train.columns if 'smile' in c.lower()), None)
        # Look for 'ic50', 'pchembl', 'value' for activity
        act_col = next((c for c in df_train.columns if any(x in c.lower() for x in ['ic50', 'pchembl', 'value', 'standard_value'])), None)

        if not smi_col or not act_col:
            print(f"❌ Error: Could not detect SMILES or Activity columns.")
            print(f"   Columns found: {df_train.columns.tolist()}")
            return

        # FILTER: Keep only Actives
        print(f"   - Filtering for Actives ({act_col} >= {ACTIVITY_THRESHOLD})...")
        
        # Ensure numeric
        df_train[act_col] = pd.to_numeric(df_train[act_col], errors='coerce')
        df_actives = df_train[df_train[act_col] >= ACTIVITY_THRESHOLD].dropna(subset=[smi_col])
        
        train_smiles = df_actives[smi_col].tolist()
        print(f"   - Found {len(train_smiles)} Active Molecules (out of {len(df_train)} total).")

        if len(train_smiles) == 0:
            print("❌ No actives found! Check your threshold or column names.")
            return

    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return

    # 3. COMPUTE FINGERPRINTS
    print("⚗️ Computing fingerprints...")
    
    # Pre-compute training FPs (The Actives)
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols if m]

    # # Compute Generated FPs (Sampled for speed if needed)
    # # For a paper, try to do ALL if possible. 25k x 4k is ~100M comparisons, might take 1-2 mins.
    # # We will sample 5000 for quick plotting, or remove this block for full fidelity.
    # if len(gen_smiles) > 5000:
    #     print("   (Sampling random 5,000 generated molecules for plotting speed...)")
    #     np.random.seed(42)
    #     gen_smiles = np.random.choice(gen_smiles, 5000, replace=False)

    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles]
    gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in gen_mols if m]

    # 4. CALCULATE MAX SIMILARITY TO NEAREST ACTIVE
    print("🔍 Calculating distance to nearest Active...")
    max_sims = []
    
    for g_fp in gen_fps:
        # Compare 1 generated molecule vs ALL 4700 actives
        sims = DataStructs.BulkTanimotoSimilarity(g_fp, train_fps)
        max_sims.append(max(sims))

    # 5. PLOT
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    plt.figure(figsize=(8, 6))
    
    ax = sns.histplot(
        max_sims, 
        bins=40, 
        kde=True, 
        stat="density",
        color="#8e44ad", 
        edgecolor="white",
        line_kws={'linewidth': 3}
    )

    mean_sim = np.mean(max_sims)
    
    # Annotations
    plt.axvline(mean_sim, color='black', linestyle='--', linewidth=2, label=f'Mean ({mean_sim:.2f})')
    plt.axvspan(0.0, 0.4, color='green', alpha=0.1, label='Novelty Zone (<0.4)')
    plt.axvspan(0.8, 1.0, color='red', alpha=0.1, label='Memorization Zone (>0.8)')

    plt.xlabel("Max Similarity to Known Actives (pIC50 > 6)", fontsize=14, fontweight='bold')
    plt.ylabel("Density", fontsize=14, fontweight='bold')
    plt.title(f"Novelty vs. Known Actives (N={len(train_fps)})", fontsize=16, pad=20)
    plt.xlim(0, 1.0)
    
    plt.legend(loc='upper right')
    sns.despine(trim=True)
    plt.tight_layout()

    output_file = 'outputs/tanimoto_actives_only.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Corrected Plot saved to {output_file}")
    print(f"📊 New Mean Similarity: {mean_sim:.3f}")

if __name__ == "__main__":
    plot_tanimoto_actives_only()