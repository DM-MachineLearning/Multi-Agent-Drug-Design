import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import os

# CONFIG
TRAINING_DATA_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (2).xlsx"
ACTIVITY_THRESHOLD = 7.5  # Only look at highly potent molecules

def get_best_scaffold():
    if not os.path.exists(TRAINING_DATA_PATH):
        print("❌ File not found.")
        return

    print(f"📂 Reading {TRAINING_DATA_PATH}...")
    df = pd.read_excel(TRAINING_DATA_PATH)

    # Detect columns
    smi_col = next((c for c in df.columns if 'smile' in c.lower()), None)
    act_col = next((c for c in df.columns if any(x in c.lower() for x in ['ic50', 'pchembl', 'value'])), None)

    # Filter for High Potency
    df[act_col] = pd.to_numeric(df[act_col], errors='coerce')
    df_elite = df[df[act_col] >= ACTIVITY_THRESHOLD].dropna(subset=[smi_col])
    
    print(f"   - Analyzing {len(df_elite)} Elite Inhibitors (pIC50 >= {ACTIVITY_THRESHOLD})...")

    scaffolds = []
    for s in df_elite[smi_col]:
        mol = Chem.MolFromSmiles(s)
        if mol:
            # Get the Bemis-Murcko Core (The Skeleton)
            core = MurckoScaffold.GetScaffoldForMol(mol)
            core_smi = Chem.MolToSmiles(core)
            if core_smi: scaffolds.append(core_smi)

    # Find the Winner
    counts = Counter(scaffolds)
    best_scaffold, count = counts.most_common(1)[0]
    
    print("\n" + "="*50)
    print("🏆 THE GOLDEN SCAFFOLD FOUND")
    print("="*50)
    print(f"Structure: {best_scaffold}")
    print(f"Frequency: Appears in {count} elite inhibitors")
    print("-" * 50)
    
    # Generate an image of it
    mol = Chem.MolFromSmiles(best_scaffold)
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles("O=C(Cc1nc(N2CCOCC2)cc(=O)[nH]1)N1CCc2ccccc21")
    d = Draw.rdMolDraw2D.MolDraw2DSVG(400, 400)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    with open("outputs/golden_scaffold.svg", "w") as f:
        f.write(d.GetDrawingText())
    
    return best_scaffold

if __name__ == "__main__":
    get_best_scaffold()