import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import os

# --- CONFIG ---
# GENERATED_FILE = "outputs/successful_molecules_CLEAN.csv"
# GENERATED_FILE = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
# GENERATED_FILE = "outputs/successful_molecules_scaffold"
GENERATED_FILE = "outputs/successful_molecules_scaffold.csv"
TRAINING_DATA_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (2).xlsx"
# OUTPUT_HTML = "outputs/scaffold_hopping_with_ic50.html"
OUTPUT_HTML = "outputs/successful_molecules_scaffold.html"
ACTIVITY_THRESHOLD = 6.0  # Only look at actives

def visualize_hops_with_scores():
    if not os.path.exists(GENERATED_FILE):
        print(f"❌ Error: {GENERATED_FILE} not found.")
        return

    # 1. LOAD GENERATED DATA
    print(f"📂 Loading Generated Leads...")
    df_gen = pd.read_csv(GENERATED_FILE)
    df_gen = df_gen[df_gen['smiles'] != 'Latent_Vector_Only']
    gen_smiles = df_gen['smiles'].tolist()

    # 2. LOAD & FILTER TRAINING DATA
    print(f"📂 Loading Training Data...")
    try:
        df_train = pd.read_excel(TRAINING_DATA_PATH)
        
        # Auto-detect columns
        smi_col = next((c for c in df_train.columns if 'smile' in c.lower()), None)
        # Look for the activity column
        act_col = next((c for c in df_train.columns if any(x in c.lower() for x in ['ic50', 'pchembl', 'value', 'standard_value'])), None)

        if not smi_col or not act_col:
            print("❌ Error: Could not detect SMILES or Activity columns.")
            return

        # Ensure numeric
        df_train[act_col] = pd.to_numeric(df_train[act_col], errors='coerce')
        
        # FILTER: Keep only Actives
        df_actives = df_train[df_train[act_col] >= ACTIVITY_THRESHOLD].dropna(subset=[smi_col])
        
        # We need to store both SMILES and their pIC50 values
        train_data = []
        for _, row in df_actives.iterrows():
            train_data.append({
                'smiles': row[smi_col],
                'pic50': row[act_col]
            })
            
        print(f"   - Found {len(train_data)} Active Molecules (pIC50 >= {ACTIVITY_THRESHOLD}).")

    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return

    # 3. COMPUTE TRAINING FINGERPRINTS
    print("⚗️ Pre-calculating fingerprints...")
    train_mols = [Chem.MolFromSmiles(d['smiles']) for d in train_data]
    
    # Store tuples of (Fingerprint, Index) to map back to pIC50 later
    train_fps = []
    valid_indices = []
    
    for i, m in enumerate(train_mols):
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            train_fps.append(fp)
            valid_indices.append(i)

    if not train_fps:
        print("❌ No valid training molecules.")
        return

    # 4. HUNT FOR SCAFFOLD HOPS
    print("🔍 Hunting for High-Potency Scaffold Hops...")
    
    results = []
    check_limit = min(len(gen_smiles), 25000)
    
    for i in range(check_limit):
        gen_smi = gen_smiles[i]
        gen_mol = Chem.MolFromSmiles(gen_smi)
        if not gen_mol: continue
        
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
        
        # Find Nearest Neighbor
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fps)
        max_sim = max(sims)
        
        # Filter for "Interesting" Hops (0.45 - 0.85)
        if 0.1 < max_sim < 0.85:
            best_fp_idx = sims.index(max_sim)          # Index in train_fps list
            real_data_idx = valid_indices[best_fp_idx] # Index in train_data list
            
            neighbor_data = train_data[real_data_idx]
            
            results.append({
                'gen_smi': gen_smi,
                'train_smi': neighbor_data['smiles'],
                'train_pic50': neighbor_data['pic50'], # Capture the pIC50!
                'similarity': max_sim
            })

    # Sort by Similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_pairs = results[:10]

    # 5. GENERATE HTML
    print(f"🎨 Generating Gallery for Top {len(top_pairs)} pairs...")
    
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f4f6f7; color: #2c3e50; }
            h1 { text-align: center; margin-top: 30px; }
            .container { display: flex; flex-direction: column; align-items: center; gap: 20px; padding-bottom: 50px; }
            .pair-card { 
                display: flex; align-items: center; gap: 30px; background: white; 
                padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); width: 900px; 
            }
            .mol-box { text-align: center; flex: 1; position: relative; }
            .mol-box h3 { font-size: 16px; margin-bottom: 10px; color: #34495e; }
            .info-box { 
                display: flex; flex-direction: column; justify-content: center; 
                align-items: center; width: 140px; font-weight: bold; 
            }
            .score-circle { 
                width: 70px; height: 70px; border-radius: 50%; background: #ecf0f1; 
                display: flex; align-items: center; justify-content: center;
                font-size: 18px; color: #2980b9; border: 3px solid #2980b9;
            }
            .pic50-tag {
                background: #e74c3c; color: white; padding: 4px 12px; border-radius: 20px;
                font-weight: bold; font-size: 14px; margin-top: 5px; display: inline-block;
            }
            .label { font-size: 10px; margin-top: 5px; color: #95a5a6; text-transform: uppercase; }
            svg { border: 1px solid #eee; border-radius: 8px; }
            .smiles { font-family: monospace; font-size: 9px; color: #bdc3c7; margin-top: 5px; max-width: 300px; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <h1>Scaffold Hopping Analysis</h1>
        <p style="text-align:center">Generated Lead (Left) vs. High-Potency Reference (Right)</p>
        <div class="container">
    """

    for idx, pair in enumerate(top_pairs):
        # Draw Generated
        gen_mol = Chem.MolFromSmiles(pair['gen_smi'])
        d1 = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        d1.drawOptions().clearBackground = False
        d1.DrawMolecule(gen_mol)
        d1.FinishDrawing()
        gen_svg = d1.GetDrawingText()

        # Draw Active Neighbor
        train_mol = Chem.MolFromSmiles(pair['train_smi'])
        d2 = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
        d2.drawOptions().clearBackground = False
        d2.DrawMolecule(train_mol)
        d2.FinishDrawing()
        train_svg = d2.GetDrawingText()

        html_content += f"""
        <div class="pair-card">
            <div class="mol-box">
                <h3>GENERATED (AI)</h3>
                {gen_svg}
                <div class="smiles">{pair['gen_smi']}</div>
            </div>
            <div class="info-box">
                <div class="score-circle">{pair['similarity']:.3f}</div>
                <div class="label">Similarity</div>
                <div style="margin-top:20px; font-size:24px;">➡️</div>
            </div>
            <div class="mol-box">
                <h3>KNOWN ACTIVE</h3>
                {train_svg}
                <div class="pic50-tag">pIC50: {pair['train_pic50']:.2f}</div>
                <div class="smiles">{pair['train_smi']}</div>
            </div>
        </div>
        """

    html_content += "</div></body></html>"

    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
    
    print(f"✅ Gallery saved to: {OUTPUT_HTML}")

if __name__ == "__main__":
    visualize_hops_with_scores()