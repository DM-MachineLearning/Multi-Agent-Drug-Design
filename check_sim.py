import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os

# --- CONFIG ---
# GENERATED_FILE = "outputs/successful_molecules_CLEAN.csv"  # Your generated leads
# GENERATED_FILE = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
GENERATED_FILE = "outputs/exploration_updateMeanVar_CLEAN.csv"
TRAINING_DATA_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (2).xlsx"
ACTIVITY_THRESHOLD = 6.0  # Definition of a "Good" molecule (pIC50)

def calculate_similarity_to_actives():
    # 1. Load Generated Molecules
    if not os.path.exists(GENERATED_FILE):
        print(f"❌ Generated file not found: {GENERATED_FILE}")
        return

    print(f"📂 Loading generated leads from {GENERATED_FILE}...")
    try:
        # Check if it has a header 'smiles' or is just a raw list
        df_gen = pd.read_csv(GENERATED_FILE)
        if 'smiles' in df_gen.columns:
            gen_smiles = df_gen['smiles'].tolist()
        else:
            # Fallback for headerless files
            gen_smiles = df_gen.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading generated file: {e}")
        return

    # 2. Load Reference Dataset (AKT1 Training Data)
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Training data not found: {TRAINING_DATA_PATH}")
        return

    print(f"📂 Loading training data from {TRAINING_DATA_PATH}...")
    try:
        df_train = pd.read_excel(TRAINING_DATA_PATH)
        
        # Identify columns (Adjust these names if your Excel is different!)
        # Common names: 'Smiles', 'SMILES', 'Canonical_Smiles'
        # Common names: 'pIC50', 'Value', 'Standard Value'
        smiles_col = next((c for c in df_train.columns if 'smile' in c.lower()), None)
        activity_col = next((c for c in df_train.columns if 'ic50' in c.lower() or 'value' in c.lower()), None)
        
        if not smiles_col or not activity_col:
            print(f"❌ Could not auto-detect columns. Found: {df_train.columns.tolist()}")
            return

        # Filter for "Good" molecules
        actives_df = df_train[df_train[activity_col] >= ACTIVITY_THRESHOLD]
        ref_smiles = actives_df[smiles_col].dropna().tolist()
        
        print(f"   - Found {len(ref_smiles)} active molecules (pIC50 >= {ACTIVITY_THRESHOLD}) in training set.")

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 3. Compute Fingerprints
    print("⚗️ Calculating chemical fingerprints...")
    
    # Helper to safe-gen fingerprint
    def get_fp(smi):
        m = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

    # Generated FPs
    gen_fps = [get_fp(s) for s in gen_smiles]
    gen_fps = [x for x in gen_fps if x is not None] # Remove invalid

    # Reference FPs (Actives)
    ref_fps = [get_fp(s) for s in ref_smiles]
    ref_fps = [x for x in ref_fps if x is not None]
    
    if not gen_fps or not ref_fps:
        print("❌ No valid fingerprints could be generated.")
        return

    # 4. Compare Generated vs. Reference
    # We want to know: For each generated molecule, how close is its *nearest neighbor* in the training set?
    
    print("🔍 Comparing Generated vs. Training Actives (this may take a moment)...")
    
    max_similarities = []
    
    # Loop through generated molecules
    for i, g_fp in enumerate(gen_fps):
        # Calculate bulk similarity to ALL actives at once
        sims = DataStructs.BulkTanimotoSimilarity(g_fp, ref_fps)
        max_sim = max(sims)  # The single closest match
        max_similarities.append(max_sim)

    # 5. Report Statistics
    avg_max_sim = np.mean(max_similarities)
    
    print("\n" + "="*50)
    print("📊 SIMILARITY ANALYSIS REPORT")
    print("="*50)
    print(f"Generated Molecules Checked: {len(gen_fps)}")
    print(f"Training Actives Used:       {len(ref_fps)}")
    print("-" * 50)
    print(f"Average Max Similarity:      {avg_max_sim:.3f}")
    print(f"   (0.0 = Totally New, 1.0 = Memorized)")
    
    # Categorize
    novel = sum(1 for s in max_similarities if s < 0.4)
    derivatives = sum(1 for s in max_similarities if 0.4 <= s < 0.8)
    clones = sum(1 for s in max_similarities if s >= 0.8)
    
    print("\nBreakdown of Generated Leads:")
    print(f"   🌱 Novel Structures (<0.4 sim):     {novel} ({novel/len(gen_fps):.1%})")
    print(f"   🌿 Derivatives (0.4-0.8 sim):       {derivatives} ({derivatives/len(gen_fps):.1%})")
    print(f"   👯 Clones/Highly Similar (>0.8 sim):{clones} ({clones/len(gen_fps):.1%})")
    print("="*50)

if __name__ == "__main__":
    calculate_similarity_to_actives()
