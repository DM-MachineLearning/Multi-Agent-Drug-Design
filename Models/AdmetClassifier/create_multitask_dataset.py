import torch
import pandas as pd
import numpy as np
import os
import glob
from Generators.VAE import VAE 

# --- CONFIG ---
ROOT_DIR = "Models/AdmetClassifier/Auto_ML_dataset/"
SAVE_PATH = "data/admet_multitask_latent.pt"
VAE_PATH = "checkpoints/vae.pt"

# The exact folder names you listed
TASK_NAMES = [
    'BBBP', 'Caco2_permeability', 'CYP1A2_inhibition', 'CYP2C19_inhibition', 
    'CYP2C9_inhibition', 'CYP2D6_inhibition', 'CYP3A4_inhibition', 
    'hERG_inhibition', 'HLM_stability', 'P-gp_substrate', 'RLM_stability'
]

def create_master_dataset():
    # 1. Build a Master Dictionary: { SMILES: [Label1, Label2, ... Label11] }
    # Initialize with NaN (meaning "missing data")
    print("Merging CSV files...")
    smiles_db = {} 

    for task_idx, task in enumerate(TASK_NAMES):
        task_folder = os.path.join(ROOT_DIR, task)
        
        # Find the train CSV (assuming format like *_train_set.csv)
        search_path = os.path.join(task_folder, "*_train_set.csv")
        files = glob.glob(search_path)
        
        if not files:
            print(f"⚠️ Warning: No train file found for {task}")
            continue
            
        # Load Data
        df = pd.read_csv(files[0])
        # Columns are usually SMILES, bioclass. 
        # Ensure we strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        print(f"  - Loaded {len(df)} rows for {task}")

        for _, row in df.iterrows():
            smi = row['SMILES'].strip()
            label = float(row['bioclass']) # 0 or 1
            
            if smi not in smiles_db:
                # New molecule: Create array of NaNs (placeholder for missing tasks)
                smiles_db[smi] = np.full(len(TASK_NAMES), np.nan)
            
            # Set the label for THIS task
            smiles_db[smi][task_idx] = label

    print(f"Total unique molecules found: {len(smiles_db)}")

    # 2. Encode with VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading VAE...")
    vae = VAE(model_path=VAE_PATH)
    vae.load_model()
    vae.model.eval()

    z_list = []
    y_list = []
    
    print("Encoding molecules to Latent Space...")
    with torch.no_grad():
        for i, (smi, labels) in enumerate(smiles_db.items()):
            try:
                # Encode
                tokens = vae.tokenizer.encode(smi, add_special_tokens=True)
                input_tensor = torch.tensor([tokens]).to(device)
                embed = vae.model.embedding(input_tensor)
                _, h_n = vae.model.encoder_gru(embed)
                h_n = h_n.squeeze(0)
                z = vae.model.fc_mu(h_n) # Use Mean (mu)
                
                z_list.append(z.cpu())
                y_list.append(torch.tensor(labels).float())
                
            except Exception:
                continue # Skip broken SMILES
            
            if i % 1000 == 0:
                print(f"Processed {i}...")

    # 3. Save
    X_train = torch.cat(z_list, dim=0)
    y_train = torch.stack(y_list) # Shape: [N, 11]
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({'z': X_train, 'y': y_train, 'tasks': TASK_NAMES}, SAVE_PATH)
    print(f"✅ Saved master dataset to {SAVE_PATH}. Shape: {y_train.shape}")

if __name__ == "__main__":
    create_master_dataset()