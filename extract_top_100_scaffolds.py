import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import os

# CONFIG
TRAINING_DATA_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (2).xlsx"
ACTIVITY_THRESHOLD = 7.0  # Slightly looser threshold to ensure we get 100 diverse ones
OUTPUT_FILE = "outputs/top_100_scaffolds.txt"

def extract_legion():
    if not os.path.exists(TRAINING_DATA_PATH):
        print("❌ Data file not found.")
        return

    print(f"📂 Reading {TRAINING_DATA_PATH}...")
    df = pd.read_excel(TRAINING_DATA_PATH)

    # Detect columns
    smi_col = next((c for c in df.columns if 'smile' in c.lower()), None)
    act_col = next((c for c in df.columns if any(x in c.lower() for x in ['ic50', 'pchembl', 'value'])), None)

    # Filter for Actives
    df[act_col] = pd.to_numeric(df[act_col], errors='coerce')
    df_active = df[df[act_col] >= ACTIVITY_THRESHOLD].dropna(subset=[smi_col])
    
    print(f"   - Found {len(df_active)} Potent Inhibitors (pIC50 >= {ACTIVITY_THRESHOLD})")

    scaffolds = []
    print("⚗️  Extracting Bemis-Murcko Scaffolds...")
    
    for s in df_active[smi_col]:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                # Get the core skeleton
                core = MurckoScaffold.GetScaffoldForMol(mol)
                core_smi = Chem.MolToSmiles(core)
                if core_smi and len(core_smi) > 5: # Ignore tiny fragments like benzene
                    scaffolds.append(core_smi)
            except:
                continue

    # Count frequencies
    counts = Counter(scaffolds)
    
    # Get Top 100
    top_100 = counts.most_common(100)
    
    print("\n" + "="*50)
    print(f"🏛️  THE LEGION OF 100")
    print("="*50)
    
    with open(OUTPUT_FILE, "w") as f:
        for i, (scaff, count) in enumerate(top_100):
            f.write(f"{scaff}\n")
            if i < 5:
                print(f"#{i+1}: {scaff} ({count} hits)")
    
    print(f"\n✅ Saved Top 100 Scaffolds to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_legion()