import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np

def plot_pareto_correct(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    print(f"📂 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[df['smiles'] != 'Latent_Vector_Only']

    potencies = []
    bbbp_scores = []

    # --- THE FIX: Regex that handles scientific notation (e.g., 1.2e-05) ---
    # Matches numbers like 0.99, 1.0, 1.23e-5, 4.5E-10
    number_pattern = r"([\d\.]+(?:[eE][-+]?\d+)?)"
    
    potency_regex = re.compile(r"'potency':\s*(?:tensor\(\[\[)?" + number_pattern)
    bbbp_regex = re.compile(r"'BBBP':\s*" + number_pattern)

    for index, row in df.iterrows():
        score_str = str(row['captions'])
        
        p_match = potency_regex.search(score_str)
        b_match = bbbp_regex.search(score_str)
        
        if p_match and b_match:
            try:
                p_val = float(p_match.group(1))
                b_val = float(b_match.group(1))
                
                # Sanity check: Probabilities shouldn't be > 1.0 (unless logit)
                # If slightly over due to float error, clamp it.
                if p_val > 1.0 and p_val < 1.0001: p_val = 1.0
                if b_val > 1.0 and b_val < 1.0001: b_val = 1.0
                
                potencies.append(p_val)
                bbbp_scores.append(b_val)
            except:
                continue

    # --- PLOT FOR ICML ---
    plt.figure(figsize=(10, 7))
    
    # Plot points
    sc = plt.scatter(potencies, bbbp_scores, 
                     c=potencies, cmap='viridis', 
                     alpha=0.5, s=20, edgecolors='none')
    
    # Target Box: Potency > 0.8 AND BBBP < 0.2
    rect = plt.Rectangle((0.8, 0.0), 0.25, 0.2, 
                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.1, 
                         label='Pareto Optimal Region')
    plt.gca().add_patch(rect)

    plt.colorbar(sc, label='Potency Score')
    plt.title(f'Pareto Frontier (Corrected): {len(potencies)} Candidates', fontsize=14)
    plt.xlabel('Potency (Higher is Better)', fontsize=12)
    plt.ylabel('BBBP Toxicity (Lower is Better)', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xlim(0.4, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')

    plt.savefig('outputs/pareto_fixed.png', dpi=300)
    print("✅ Corrected Plot saved to outputs/pareto_fixed.png")

if __name__ == "__main__":
    # plot_pareto_frontier("outputs/successful_molecules.csv")
    # plot_pareto_frontier("outputs/exploration_updateMeanVar.csv")
    plot_pareto_correct("outputs/exploration_updateMeanVar_50update.csv")