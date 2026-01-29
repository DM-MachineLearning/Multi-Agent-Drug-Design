import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
import gzip
import pickle
from rdkit import Chem
from rdkit.Chem import QED

# --- 1. AUTO-DOWNLOADER FOR SA SCORE ---
# SA Score is external to standard RDKit, so we fetch it dynamically.
SA_REPO_URL = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/"

def ensure_sa_dependencies():
    files = ["sascorer.py", "fpscores.pkl.gz"]
    for f in files:
        if not os.path.exists(f):
            print(f"⬇️ Downloading {f} for Synthetic Accessibility calculation...")
            r = requests.get(SA_REPO_URL + f)
            with open(f, 'wb') as file:
                file.write(r.content)
    print("✅ SA Score dependencies ready.")

# Download first, then import
ensure_sa_dependencies()
import sascorer

def plot_sa_qed_landscape(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"📂 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df['smiles'] != 'Latent_Vector_Only']

    qeds = []
    sa_scores = []
    
    print(f"⚗️ Calculating SA Scores for {len(df)} molecules (this takes a minute)...")
    
    for smi in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            
            # 1. Calculate QED
            q = QED.qed(mol)
            
            # 2. Calculate SA Score
            # Range: 1 (Easy) to 10 (Hard)
            sa = sascorer.calculateScore(mol)
            
            qeds.append(q)
            sa_scores.append(sa)
        except:
            continue
            
    if not qeds: return

    # --- 2. ICML VISUALIZATION ---
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")

    # JointGrid: QED (X) vs SA Score (Y)
    g = sns.JointGrid(x=qeds, y=sa_scores, height=8, ratio=4)

    # A. HEXBIN PLOT
    # Note: We invert the colormap ('viridis_r') because for SA Score, 
    # we usually want to highlight the 'Easy' region (Low SA).
    g.plot_joint(plt.hexbin, 
                 gridsize=35, 
                 cmap='Spectral_r',  # Red=Hard, Blue=Easy/Good
                 bins='log', 
                 mincnt=1, 
                 linewidths=0.2, 
                 edgecolors='gray')

    # B. MARGINALS
    sns.kdeplot(x=qeds, ax=g.ax_marg_x, fill=True, color="#2ecc71", bw_adjust=0.5)
    sns.kdeplot(y=sa_scores, ax=g.ax_marg_y, fill=True, color="#e74c3c", bw_adjust=0.5)

    # --- 3. ANNOTATIONS ---
    
    # Ideal Zone: High QED (>0.6) AND Easy to Synthesize (<4.0)
    import matplotlib.patches as patches
    
    # Draw "Sweet Spot" Box (Bottom Right)
    rect = patches.Rectangle((0.6, 1.0), 0.4, 3.0, 
                             linewidth=2.5, edgecolor='#2980b9', facecolor='none', 
                             linestyle='--', zorder=10)
    g.ax_joint.add_patch(rect)
    g.ax_joint.text(0.8, 2.5, "Sweet Spot\n(Drug-like & Easy)", 
                    color='#2980b9', ha='center', va='center', fontsize=12, fontweight='bold')

    # Labels
    g.ax_joint.set_xlabel("Drug-likeness (QED) $\\rightarrow$", fontsize=16, fontweight='bold')
    g.ax_joint.set_ylabel("Synthetic Accessibility (SA Score) $\\leftarrow$", fontsize=16, fontweight='bold')
    
    # Invert Y Axis? 
    # Usually SA score 1 is good (bottom), 10 is bad (top). 
    # We keep standard 1->10 but emphasize the bottom is better.
    g.ax_joint.set_ylim(1, 6) # Zoom in on the realistic range (1-6). Most drugs are < 6.
    g.ax_joint.set_xlim(0, 1.05)

    # Add colorbar
    cb = plt.colorbar(g.ax_joint.collections[0], ax=g.ax_joint, pad=0.02, aspect=30)
    cb.set_label('Molecule Density (Log Scale)', fontsize=12)

    # plt.suptitle("Synthesizability vs. Drug-likeness Landscape", y=1.02, fontsize=18)
    
    plt.savefig('outputs/sa_score_landscape.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("✅ Plot saved to outputs/sa_score_landscape.pdf")

if __name__ == "__main__":
    # plot_sa_qed_landscape("outputs/successful_molecules.csv")
    plot_sa_qed_landscape("outputs/exploration_updateMeanVar_50update.csv")