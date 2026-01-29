import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
import os

def plot_publication_quality_qed(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    # --- 1. PREPARE DATA ---
    print(f"📂 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter valid molecules
    df = df[df['smiles'] != 'Latent_Vector_Only']
    
    print("⚗️ Calculating QED for all candidates (this may take a moment)...")
    qed_values = []
    
    for smi in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                qed_values.append(QED.qed(mol))
        except:
            continue
            
    if not qed_values:
        print("❌ No valid molecules found.")
        return

    # --- 2. SETUP "ICML" AESTHETICS ---
    # Set the context to 'paper' or 'talk'. 'paper' uses slightly smaller fonts suitable for double-column.
    sns.set_context("paper", font_scale=1.4) 
    sns.set_style("ticks") # Minimalist style (no grey grid background)
    
    plt.figure(figsize=(8, 5)) # Width=8, Height=5 inches (Standard half-page)

    # --- 3. THE PLOT ---
    # Histplot with Kernel Density Estimate (KDE)
    ax = sns.histplot(
        qed_values, 
        kde=True,                  # The smooth line
        bins=30, 
        color="#2c3e50",           # Professional Dark Blue/Grey
        edgecolor="white",         # Clean separation between bars
        stat="density",            # Normalize so area=1
        linewidth=0.5,
        alpha=0.6                  # Transparency
    )
    
    # Get line from KDE and make it pop
    plt.setp(ax.lines, linewidth=2.5, color="#e74c3c") # Red line for contrast

    # --- 4. ANNOTATIONS & REGIONS ---
    
    # Shade the "High Quality" region (QED > 0.6)
    plt.axvspan(0.6, 1.0, color='green', alpha=0.05) #, label='Drug-like Zone (>0.6)')
    
    # Add vertical line for MEAN
    mean_qed = np.mean(qed_values)
    plt.axvline(mean_qed, color='black', linestyle='--', linewidth=1.5) #, label=f'Mean ({mean_qed:.2f})')

    # Add Text Box with Stats (Top Left)
    stats_text = (
        f"$\mathbf{{N}} = {len(qed_values)}$\n"
        f"$\mu = {mean_qed:.3f}$\n"
        f"$\sigma = {np.std(qed_values):.3f}$"
    )
    plt.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"))

    # --- 5. CLEANUP ---
    plt.xlabel("QED Score (Drug-likeness)", fontsize=14, fontweight='bold')
    plt.ylabel("Density", fontsize=14, fontweight='bold')
    plt.title("Distribution of Drug-likeness in Generated Library", fontsize=16, pad=20)
    
    plt.xlim(0, 1.0)
    sns.despine(offset=10, trim=True) # Removes the top and right borders (Very "Paper" look)
    
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    # --- 6. SAVE AS VECTOR ---
    # Save as PDF for the LaTeX paper (Infinite resolution)
    plt.savefig('outputs/qed_distribution_icml.pdf', format='pdf', dpi=300)
    # Save as PNG for slides
    plt.savefig('outputs/qed_distribution_icml.png', format='png', dpi=300)
    
    print("✅ High-Quality Plot saved to outputs/qed_distribution_icml.pdf")

if __name__ == "__main__":
    # plot_publication_quality_qed("outputs/successful_molecules.csv")
    plot_publication_quality_qed("outputs/exploration_updateMeanVar_50update.csv")