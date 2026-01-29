# import pandas as pd
# import numpy as np
# from rdkit import Chem, DataStructs
# from rdkit.Chem import rdFingerprintGenerator
# import os

# def calculate_clean_metrics(gen_df, train_smiles_list):
#     # Modern Fingerprint Generator
#     mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
#     # 1. CLEANING STEP: Remove placeholders and non-string values
#     print("🧹 Cleaning data and filtering placeholders...")
    
#     # Ensure column is treated as string and drop empty values
#     smi_col = gen_df['smiles'].astype(str).str.strip()
    
#     # Filter out known non-SMILES strings
#     filter_mask = (smi_col != "Latent_Vector_Only") & (smi_col != "nan") & (smi_col != "")
#     clean_raw_smiles = smi_col[filter_mask].tolist()
    
#     # 2. RDKIT VALIDATION
#     valid_mols = []
#     valid_smiles = []
#     for s in clean_raw_smiles:
#         m = Chem.MolFromSmiles(s)
#         if m:
#             valid_mols.append(m)
#             valid_smiles.append(Chem.MolToSmiles(m))
    
#     # Validity is based on the total rows in the CSV (to punish for junk rows)
#     validity = len(valid_mols) / len(gen_df)
    
#     if not valid_mols:
#         return {"Error": "No valid SMILES found after filtering 'Latent_Vector_Only'"}

#     # 3. UNIQUENESS & NOVELTY
#     unique_smiles = list(set(valid_smiles))
#     uniqueness = len(unique_smiles) / len(valid_smiles)
    
#     train_set = set(train_smiles_list)
#     novel_smiles = [s for s in unique_smiles if s not in train_set]
#     novelty = len(novel_smiles) / len(unique_smiles)
    
#     # 4. SIMILARITY & DIVERSITY
#     print(f"🧬 Processing {len(unique_smiles)} unique molecules...")
    
#     # Generate Training Fingerprints (Use a small sample if train_set is huge)
#     train_fps = [mfpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles_list if Chem.MolFromSmiles(s)]
#     gen_fps = [mfpgen.GetFingerprint(m) for m in valid_mols]

#     # Nearest Neighbor Similarity (External)
#     nn_sims = []
#     for g_fp in gen_fps:
#         sims = DataStructs.BulkTanimotoSimilarity(g_fp, train_fps)
#         nn_sims.append(max(sims))
#     avg_nn_sim = np.mean(nn_sims)

#     # Diversity (Internal)
#     # Using 1 - Tanimoto to measure how spread out the agents are
#     int_sims = []
#     # Sample up to 1000 pairs if dataset is massive for speed
#     sample_fps = gen_fps[:1000] 
#     for i in range(len(sample_fps)):
#         if i + 1 < len(sample_fps):
#             s = DataStructs.BulkTanimotoSimilarity(sample_fps[i], sample_fps[i+1:])
#             int_sims.extend(s)
#     diversity = 1 - np.mean(int_sims) if int_sims else 0

#     return {
#         "Validity": validity,
#         "Uniqueness": uniqueness,
#         "Novelty": novelty,
#         "Avg Nearest Neighbor Sim": avg_nn_sim,
#         "Internal Diversity": diversity
#     }

# if __name__ == "__main__":
#     GENERATED_CSV = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
    
#     # LOAD DATA
#     df = pd.read_csv(GENERATED_CSV)
    
#     # IMPORTANT: Replace this with your actual training data SMILES
#     train_smiles = pd.read_csv("data/ChEMBL_Smiles.csv")['Smiles'].tolist()

#     # If ChEMBL is too big, take a representative sample of 100,000 molecules
#     if len(train_smiles) > 100000:
#         import random
#         train_smiles = random.sample(train_smiles, 100000)
#         print("⚠️ Sampled 100k molecules from ChEMBL for faster calculation.")
#     # train_smiles = ["C", "O=C(O)C", "c1ccccc1"] # Placeholder
    
#     results = calculate_clean_metrics(df, train_smiles)
    
#     print("\n" + "="*45)
#     print(f"{'Captions (Metric)':<30} | {'Value':<10}")
#     print("="*45)
#     for k, v in results.items():
#         if isinstance(v, float):
#             print(f"{k:<30} | {v:.4f}")
#         else:
#             print(f"{k:<30} | {v}")
#     print("="*45)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog

def calculate_benchmark_metrics(gen_smiles):
    # Initialize Filter Catalog (PAINS, BRENK, NIH)
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.ALL)
    catalog = FilterCatalog.FilterCatalog(params)
    
    scaffolds = set()
    pass_filters = 0
    valid_smiles = []
    
    print(f"⌛ Benchmarking {len(gen_smiles)} molecules...")
    
    for smi in gen_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 1. Scaffold Extraction
            try:
                scaff = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(scaff))
            except:
                pass
            
            # 2. MedChem Filters (PAINS/BRENK)
            if not catalog.HasMatch(mol):
                pass_filters += 1
            
            valid_smiles.append(smi)

    # Metrics
    scaffold_count = len(scaffolds)
    scaffold_diversity = scaffold_count / len(valid_smiles) if valid_smiles else 0
    filter_pass_rate = pass_filters / len(valid_smiles) if valid_smiles else 0
    
    return {
        "Scaffold Count": scaffold_count,
        "Scaffold Diversity": scaffold_diversity,
        "MedChem Pass Rate (PAINS/NIH)": filter_pass_rate
    }

# --- RUN ON YOUR CSV ---
# df = pd.read_csv("outputs/exploration_updateMeanVar_50update.csv")
df = pd.read_csv("outputs/successful_molecules_scaffold_sample_5_soft.csv")
# Filter out the junk rows like we did before
clean_smiles = df[df['smiles'] != "Latent_Vector_Only"]['smiles'].tolist()

bench_results = calculate_benchmark_metrics(clean_smiles)

print("\n" + "="*45)
print(f"{'Captions (Benchmark)':<30} | {'Value':<10}")
print("="*45)
for k, v in bench_results.items():
    print(f"{k:<30} | {v:.4f}")
print("="*45)