# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import os
# import numpy as np

# def plot_hard_filter_pareto(csv_path):
#     if not os.path.exists(csv_path):
#         print(f"❌ Error: {csv_path} not found.")
#         return

#     print(f"📂 Reading {csv_path}...")
#     df = pd.read_csv(csv_path)
#     df = df[df['smiles'] != 'Latent_Vector_Only']

#     # --- REGEX SETUP ---
#     # Captures floats and scientific notation (e.g., 1.2e-5)
#     num_pat = r"([\d\.]+(?:[eE][-+]?\d+)?)"
    
#     # Compile regex for speed
#     reg_potency = re.compile(r"'potency':\s*(?:tensor\(\[\[)?" + num_pat)
#     reg_herg    = re.compile(r"'hERG_inhibition':\s*" + num_pat)
#     reg_cyp     = re.compile(r"'CYP3A4_inhibition':\s*" + num_pat)

#     data = {'potency': [], 'herg': [], 'cyp': []}

#     print(f"   - Parsing scores for {len(df)} candidates...")

#     for _, row in df.iterrows():
#         s = str(row.get('captions', ''))
        
#         m_pot = reg_potency.search(s)
#         m_herg = reg_herg.search(s)
#         m_cyp = reg_cyp.search(s)
        
#         if m_pot and m_herg and m_cyp:
#             try:
#                 p = float(m_pot.group(1))
#                 h = float(m_herg.group(1))
#                 c = float(m_cyp.group(1))
                
#                 # Clamp probabilities 0-1
#                 data['potency'].append(min(max(p, 0), 1))
#                 data['herg'].append(min(max(h, 0), 1))
#                 data['cyp'].append(min(max(c, 0), 1))
#             except:
#                 continue

#     # --- PLOTTING ---
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

#     # --- PLOT 1: hERG (The Heart Safety Test) ---
#     sc1 = ax1.scatter(data['potency'], data['herg'], 
#                       c=data['potency'], cmap='viridis', 
#                       alpha=0.5, s=15, edgecolors='none')
    
#     # Threshold Line
#     ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Hard Limit (0.3)')
    
#     # Elite Zone (Potency > 0.9, hERG < 0.05)
#     rect1 = plt.Rectangle((0.9, 0.0), 0.1, 0.05, linewidth=2, edgecolor='lime', facecolor='none', label='Elite Safety Zone')
#     ax1.add_patch(rect1)

#     ax1.set_title(f"Target 1: hERG Safety (N={len(data['potency'])})", fontsize=14)
#     ax1.set_xlabel("Biological Potency (AKT1)", fontsize=12)
#     ax1.set_ylabel("hERG Inhibition Probability", fontsize=12)
#     ax1.set_ylim(-0.02, 0.35) # Zoom in on the safe zone
#     ax1.set_xlim(0.5, 1.02)
#     ax1.legend(loc='upper left')
#     ax1.grid(True, linestyle=':', alpha=0.6)

#     # --- PLOT 2: CYP3A4 (The Metabolic Stability Test) ---
#     sc2 = ax2.scatter(data['potency'], data['cyp'], 
#                       c=data['potency'], cmap='magma', 
#                       alpha=0.5, s=15, edgecolors='none')
    
#     # Threshold Line
#     ax2.axhline(y=0.4, color='red', linestyle='--', linewidth=2, label='Hard Limit (0.4)')
    
#     # Elite Zone (Potency > 0.9, CYP < 0.05)
#     rect2 = plt.Rectangle((0.9, 0.0), 0.1, 0.05, linewidth=2, edgecolor='cyan', facecolor='none', label='Elite Metabolic Zone')
#     ax2.add_patch(rect2)

#     ax2.set_title(f"Target 2: CYP3A4 Stability (N={len(data['potency'])})", fontsize=14)
#     ax2.set_xlabel("Biological Potency (AKT1)", fontsize=12)
#     ax2.set_ylabel("CYP3A4 Inhibition Probability", fontsize=12)
#     ax2.set_ylim(-0.02, 0.45) # Zoom in
#     ax2.set_xlim(0.5, 1.02)
#     ax2.legend(loc='upper left')
#     ax2.grid(True, linestyle=':', alpha=0.6)

#     # Save
#     plt.tight_layout()
#     plt.savefig('outputs/hard_filters_pareto.png', dpi=300)
#     print("✅ Dual-Pareto Plot saved: outputs/hard_filters_pareto.png")

#     # --- LATEX STATS ---
#     safe_herg = sum(1 for h in data['herg'] if h < 0.05)
#     safe_cyp = sum(1 for c in data['cyp'] if c < 0.05)
#     elite_both = sum(1 for p, h, c in zip(data['potency'], data['herg'], data['cyp']) 
#                      if p > 0.9 and h < 0.05 and c < 0.05)
    
#     print("\n📝 PAPER STATS:")
#     print(f"Candidates with near-zero hERG (<0.05): {safe_herg} ({safe_herg/len(data['herg'])*100:.1f}%)")
#     print(f"Candidates with near-zero CYP3A4 (<0.05): {safe_cyp} ({safe_cyp/len(data['cyp'])*100:.1f}%)")
#     print(f"TRIPLE ELITE (Potent + Safe hERG + Safe CYP): {elite_both} candidates")

# if __name__ == "__main__":
#     # plot_hard_filter_pareto("outputs/successful_molecules.csv")
#     plot_hard_filter_pareto("outputs/exploration_updateMeanVar_50update.csv")

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

def generate_scientific_plots(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"📂 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df['smiles'] != 'Latent_Vector_Only']

    print(f"   - Calculating metrics for {len(df)} candidates...")

    # Data Containers
    potencies = []
    bbbp_scores = []
    qed_scores = []
    
    # Regex for scientific notation
    number_pattern = r"([\d\.]+(?:[eE][-+]?\d+)?)"
    potency_regex = re.compile(r"'potency':\s*(?:tensor\(\[\[)?" + number_pattern)
    bbbp_regex = re.compile(r"'BBBP':\s*" + number_pattern)

    for index, row in df.iterrows():
        try:
            # 1. Get Potency & BBBP from logs
            score_str = str(row['captions'])
            p_match = potency_regex.search(score_str)
            b_match = bbbp_regex.search(score_str)
            
            # 2. Calculate QED "On the Fly" (Crucial for a real trade-off plot)
            mol = Chem.MolFromSmiles(row['smiles'])
            if not mol: continue
            
            qed = QED.qed(mol)

            if p_match and b_match:
                p_val = float(p_match.group(1))
                b_val = float(b_match.group(1))
                
                # Clamp for graph cleanliness
                p_val = min(max(p_val, 0.0), 1.0)
                b_val = min(max(b_val, 1e-35), 1.0) # Avoid log(0) error
                
                potencies.append(p_val)
                bbbp_scores.append(b_val)
                qed_scores.append(qed)
        except:
            continue

    # --- PLOT 1: THE REAL TRADE-OFF (Potency vs. QED) ---
    plt.figure(figsize=(10, 7))
    
    plt.scatter(potencies, qed_scores, 
                c=qed_scores, cmap='RdYlGn', 
                alpha=0.6, s=15, edgecolors='grey', linewidth=0.2)
    
    # Optimal Zone: Potent (>0.9) AND Drug-like (>0.7)
    rect = plt.Rectangle((0.9, 0.7), 0.1, 0.3, 
                         linewidth=2, edgecolor='blue', facecolor='none', 
                         label='High-Priority Candidates')
    plt.gca().add_patch(rect)

    plt.title(f'Multi-Objective Optimization: Potency vs. Drug-likeness', fontsize=14)
    plt.xlabel('Biological Potency P(Active)', fontsize=12)
    plt.ylabel('QED Score (Structural Quality)', fontsize=12)
    plt.colorbar(label='QED Score')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    
    plt.savefig('outputs/pareto_potency_qed.png', dpi=300)
    print("✅ Plot 1 Saved: outputs/pareto_potency_qed.png")

    # --- PLOT 2: THE SAFETY PROOF (Log Scale) ---
    # This proves the agents aren't just outputting "0", but are optimizing deep into the safe zone
    plt.figure(figsize=(10, 7))
    
    plt.scatter(potencies, bbbp_scores, 
                c='dodgerblue', alpha=0.5, s=15, edgecolors='none')
    
    plt.yscale('log') # <--- THE KEY CHANGE
    
    plt.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Hard Safety Limit (0.3)')
    
    plt.title(f'Safety Constraint Satisfaction (Log Scale)', fontsize=14)
    plt.xlabel('Biological Potency P(Active)', fontsize=12)
    plt.ylabel('BBBP Toxicity Probability (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper left')
    
    plt.savefig('outputs/pareto_safety_log.png', dpi=300)
    print("✅ Plot 2 Saved: outputs/pareto_safety_log.png")

if __name__ == "__main__":
    # generate_scientific_plots("outputs/successful_molecules.csv")
    # generate_scientific_plots("outputs/exploration_updateMeanVar_50update.csv")
    generate_scientific_plots("outputs/successful_molecules_scaffold_sample_5_soft.csv")