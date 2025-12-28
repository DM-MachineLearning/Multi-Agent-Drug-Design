import torch
import pandas as pd
import numpy as np
from Generators.VAE import VAE 

# CONFIG
CSV_PATH = "Models/ActivityClassifier/AKT1 CHEMBL (1).xlsx"   # Your file path
SAVE_PATH = "Models/ActivityClassifier/latent_dataset.pt"
VAE_PATH = "checkpoints/vae.pt"
ACTIVITY_THRESHOLD = 6.0 # pChEMBL > 6 means "Active" (Label 1)

def create_dataset_from_chembl():
    # 1. Load Data
    df = pd.read_excel(CSV_PATH) # ChEMBL is often tab-separated, change to ',' if comma
    
    # Check column names (Clean up spaces just in case)
    df.columns = df.columns.str.strip() 
    
    # Filter for valid data
    # We only want rows that have a SMILES and a pChEMBL value
    df = df.dropna(subset=['Smiles', 'pChEMBL Value'])
    
    print(f"Loaded {len(df)} molecules.")

    # 2. Load VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(model_path=VAE_PATH)
    vae.load_model()
    vae.model.eval()

    z_list = []
    y_list = []

    print("Encoding molecules to Latent Space...")
    
    with torch.no_grad():
        for index, row in df.iterrows():
            smiles = row['Smiles']
            pchembl = float(row['pChEMBL Value'])
            
            try:
                # --- ENCODE SMILES TO Z ---
                # Tokenize
                tokens = vae.tokenizer.encode(smiles, add_special_tokens=True)
                # Convert to Tensor
                input_tensor = torch.tensor([tokens]).to(device)
                
                # Pass through VAE Encoder
                embed = vae.model.embedding(input_tensor)
                _, h_n = vae.model.encoder_gru(embed)
                h_n = h_n.squeeze(0)
                mu = vae.model.fc_mu(h_n) # Use Mean (mu) for training features
                
                # --- CREATE LABEL ---
                # Binary Classification: 1 if Active, 0 if Inactive
                label = 1.0 if pchembl >= ACTIVITY_THRESHOLD else 0.0
                
                z_list.append(mu.cpu())
                y_list.append(torch.tensor([label]))
                
            except Exception as e:
                # Sometimes RDKit/VAE fails on weird molecules, just skip them
                continue
            
            if index % 100 == 0:
                print(f"Processed {index}...")

    # 3. Save Final Dataset
    if len(z_list) > 0:
        X_train = torch.cat(z_list, dim=0)
        y_train = torch.stack(y_list)
        
        torch.save({'z': X_train, 'y': y_train}, SAVE_PATH)
        print(f"Success! Saved dataset to {SAVE_PATH}")
        print(f"Data Shape: {X_train.shape}")
        print(f"Active Molecules: {y_train.sum().item()} / {len(y_train)}")
    else:
        print("Error: No valid data created.")

# Run it
create_dataset_from_chembl()