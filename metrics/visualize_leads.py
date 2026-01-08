import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

def visualize_to_html(filepath, output_html="outputs/molecule_gallery.html", max_mols=100):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    # 1. Load data
    df = pd.read_csv(filepath)
    valid_df = df[df['smiles'] != 'Latent_Vector_Only'].head(max_mols)
    
    if valid_df.empty:
        print("No valid SMILES found.")
        return

    # 2. Generate SVG for each molecule
    html_content = """
    <html>
    <head>
        <style>
            .grid { display: flex; flex-wrap: wrap; gap: 20px; font-family: sans-serif; }
            .mol-card { border: 1px solid #ccc; padding: 10px; border-radius: 8px; text-align: center; width: 320px; }
            .mol-card svg { background-color: white; }
            .score-box { font-size: 10px; text-align: left; background: #f9f9f9; padding: 5px; height: 100px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <h1>Drug Discovery Leads - Generation Gallery</h1>
        <div class="grid">
    """

    print(f"🧬 Converting {len(valid_df)} molecules to SVG...")

    for idx, row in valid_df.iterrows():
        smiles = row['smiles']
        scores = row['captions'] # Using your 'captions' column
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            # Generate SVG text directly (Does not require Cairo)
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            html_content += f"""
            <div class="mol-card">
                <h3>Lead #{idx}</h3>
                {svg}
                <div class="score-box"><strong>Scores:</strong><br>{scores}</div>
            </div>
            """

    html_content += "</div></body></html>"

    # 3. Save HTML
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_content)
    
    print(f"✅ Success! Open this file in your browser: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    visualize_to_html("outputs/exploration_updateMeanVar.csv")