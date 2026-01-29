# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import os
# import numpy as np

# def generate_paper_artifacts(csv_path):
#     if not os.path.exists(csv_path):
#         print(f"❌ Error: {csv_path} not found.")
#         return

#     print(f"📂 Reading {csv_path}...")
#     df = pd.read_csv(csv_path)
    
#     # Filter out garbage rows
#     df = df[df['smiles'] != 'Latent_Vector_Only']

#     print(f"   - Analyzing {len(df)} successful candidates...")

#     # --- 1. DATA EXTRACTION (Robust Regex) ---
#     potencies = []
#     bbbp_scores = []
#     qed_scores = [] # If available in your logs, otherwise we skip

#     # Regex to capture scientific notation (e.g., 2.54e-28)
#     number_pattern = r"([\d\.]+(?:[eE][-+]?\d+)?)"
    
#     potency_regex = re.compile(r"'potency':\s*(?:tensor\(\[\[)?" + number_pattern)
#     bbbp_regex = re.compile(r"'BBBP':\s*" + number_pattern)
    
#     # Depending on your log format, you might have QED. 
#     # If not, we just rely on the other metrics.

#     for index, row in df.iterrows():
#         score_str = str(row['captions'])
        
#         p_match = potency_regex.search(score_str)
#         b_match = bbbp_regex.search(score_str)
        
#         if p_match and b_match:
#             try:
#                 p_val = float(p_match.group(1))
#                 b_val = float(b_match.group(1))
                
#                 # Sanity Clamp (Probability must be 0-1)
#                 p_val = min(max(p_val, 0.0), 1.0)
#                 b_val = min(max(b_val, 0.0), 1.0)
                
#                 potencies.append(p_val)
#                 bbbp_scores.append(b_val)
#             except:
#                 continue

#     if not potencies:
#         print("❌ No valid scores extracted. Check regex.")
#         return

#     # --- 2. PARETO PLOT (The "Money" Plot) ---
#     plt.figure(figsize=(10, 7))
    
#     # Scatter with color gradient
#     sc = plt.scatter(potencies, bbbp_scores, 
#                      c=potencies, cmap='viridis', 
#                      alpha=0.5, s=15, edgecolors='none')
    
#     # Draw the "Hard Filter" line to show we respected constraints
#     plt.axhline(y=0.3, color='black', linestyle='--', alpha=0.5, label='Hard Filter Threshold (0.3)')
    
#     # Highlight the "Super-Lead" Zone
#     # Potency > 0.9 AND BBBP < 0.1 (The "Perfect" Drug)
#     rect = plt.Rectangle((0.9, 0.0), 0.1, 0.1, 
#                          linewidth=2, edgecolor='red', facecolor='none', 
#                          label='Super-Lead Zone (Elite)')
#     plt.gca().add_patch(rect)

#     plt.colorbar(sc, label='Potency Score')
#     plt.title(f'Multi-Agent Optimization Landscape (N={len(potencies)})', fontsize=14)
#     plt.xlabel('Biological Potency (Target: AKT1)', fontsize=12)
#     plt.ylabel('BBBP Toxicity (Lower is Better)', fontsize=12)
    
#     # Zoom in to relevant area
#     plt.ylim(-0.02, 0.35) # Focus on the safe zone
#     plt.xlim(0.5, 1.02)   # Focus on the potent zone
    
#     plt.legend(loc='upper left')
#     plt.grid(True, linestyle=':', alpha=0.6)
    
#     plt.savefig('outputs/icml_pareto_plot.png', dpi=300)
#     print("✅ Plot saved: outputs/icml_pareto_plot.png")

#     # --- 3. LATEX TABLE STATS ---
#     # Calculate stats for the paper
    
#     high_potency_count = sum(1 for p in potencies if p > 0.9)
#     ultra_safe_count = sum(1 for b in bbbp_scores if b < 0.01)
#     elite_count = sum(1 for p, b in zip(potencies, bbbp_scores) if p > 0.9 and b < 0.05)
    
#     print("\n" + "="*50)
#     print("📝 DATA FOR LATEX TABLE")
#     print("="*50)
#     print(f"Total Valid Candidates & {len(potencies)} \\\\")
#     print(f"Mean Potency & {np.mean(potencies):.4f} $\\pm$ {np.std(potencies):.4f} \\\\")
#     print(f"Mean BBBP Score & {np.mean(bbbp_scores):.4f} (Target $<0.3$) \\\\")
#     print(f"Elite Candidates (Potency$>0.9$, BBBP$<0.05$) & {elite_count} ({elite_count/len(potencies)*100:.1f}\\% of total) \\\\")
#     print("="*50)

# if __name__ == "__main__":
#     # generate_paper_artifacts("outputs/successful_molecules.csv")
#     generate_paper_artifacts("outputs/exploration_updateMeanVar_50update.csv")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
import re
import os

def plot_hexbin_pareto(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    # --- 1. DATA LOADING (Same Robust Logic) ---
    print(f"📂 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df['smiles'] != 'Latent_Vector_Only']
    
    potencies = []
    qeds = []
    
    number_pattern = r"([\d\.]+(?:[eE][-+]?\d+)?)"
    potency_regex = re.compile(r"'potency':\s*(?:tensor\(\[\[)?" + number_pattern)

    print("⚗️ Parsing scores...")
    for index, row in df.iterrows():
        try:
            score_str = str(row['captions'])
            p_match = potency_regex.search(score_str)
            if not p_match: continue
            
            p_val = float(p_match.group(1))
            p_val = min(max(p_val, 0.0), 1.0) # Clamp 0-1
            
            # Recalculate QED for precision
            mol = Chem.MolFromSmiles(row['smiles'])
            if not mol: continue
            q_val = QED.qed(mol)
            
            potencies.append(p_val)
            qeds.append(q_val)
        except:
            continue
            
    if not potencies: return

    # --- 2. THE VISUALIZATION UPGRADE ---
    
    # Setup the Canvas
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    # Create JointGrid
    g = sns.JointGrid(x=qeds, y=potencies, height=8, ratio=4)

    # A. THE HEXBIN PLOT (The Fix for Skew)
    # gridsize=40 gives high resolution
    # bins='log' turns on the Logarithmic Scaling (Crucial for your skewed data!)
    # cmap='inferno_r' goes from Light Yellow (low density) to Dark Black/Purple (high density)
    g.plot_joint(plt.hexbin, 
                 gridsize=40, 
                 cmap='inferno_r', 
                 bins='log', 
                 mincnt=1, 
                 linewidths=0.2,
                 edgecolors='gray')

    # B. MARGINAL PLOTS (Handling the Skew)
    # We use 'kde' but with a low bandwidth adjustment (bw_adjust)
    # This prevents the peak from getting smoothed out too much
    sns.kdeplot(x=qeds, ax=g.ax_marg_x, fill=True, color="#e67e22", bw_adjust=0.5)
    
    # For the Y-axis (Activity), since it's SO skewed, a Histogram is often clearer than KDE
    # We use many bins (50) to show the fine detail near 1.0
    g.ax_marg_y.hist(potencies, bins=50, orientation='horizontal', color="#e67e22", alpha=0.7)

    # --- 3. ANNOTATIONS ---
    
    # Add Colorbar for the Hexbins
    # We have to fetch the current figure to add the floating colorbar
    cb = plt.colorbar(g.ax_joint.collections[0], ax=g.ax_joint, pad=0.02, aspect=30)
    cb.set_label('Molecule Density (Log Scale)', fontsize=12)

    # Elite Zone Box
    import matplotlib.patches as patches
    rect = patches.Rectangle((0.7, 0.9), 0.3, 0.1, 
                             linewidth=2.5, edgecolor='#2ecc71', facecolor='none', 
                             linestyle='--', zorder=10)
    g.ax_joint.add_patch(rect)
    g.ax_joint.text(0.85, 0.88, "Elite Zone\n(Target)", 
                    color='#2ecc71', ha='center', va='top', fontsize=12, fontweight='bold')

    # Labels
    g.ax_joint.set_xlabel("Drug-likeness (QED)", fontsize=16, fontweight='bold')
    g.ax_joint.set_ylabel("Activity Probability", fontsize=16, fontweight='bold')
    
    # Zoom Y-axis if necessary (e.g., if everything is > 0.5)
    # g.ax_joint.set_ylim(0.4, 1.02) # Uncomment to zoom in on top half
    g.ax_joint.set_ylim(0, 1.05)
    g.ax_joint.set_xlim(0, 1.05)

    plt.suptitle("Optimization Density: Convergence to High-Potency Regions", y=1.02, fontsize=18)
    
    plt.savefig('outputs/hexbin_pareto_icml.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('outputs/hexbin_pareto_icml.png', format='png', bbox_inches='tight', dpi=300)
    print("✅ Hexbin Plot saved! Check outputs/hexbin_pareto_icml.png")

if __name__ == "__main__":
    # plot_paper_quality_pareto("outputs/successful_molecules.csv")
    plot_hexbin_pareto("outputs/exploration_updateMeanVar_50update.csv")