# import os
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# import sys
# import re

# # Add the root directory to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Generators.VAE import VAE

# # --- 1. DEFINE THE REGEX (Crucial Step) ---
# SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# regex = re.compile(SMI_REGEX_PATTERN)

# def encode_datasets(data_root, vae_path, vocab_path, output_file="admet_latent.pt"):
#     print(f"🚀 Loading VAE from {vae_path}...")
#     vae = VAE(model_path=vae_path)
#     vae.load_model(vocab_base=vocab_path)
#     vae.model.eval()
#     device = vae.device
    
#     root = Path(data_root)
#     tasks = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
#     tasks.sort()
    
#     print(f"📋 Found {len(tasks)} ADMET Tasks: {tasks}")
    
#     all_data = []

#     for task_idx, task in enumerate(tasks):
#         task_dir = root / task
#         csv_files = list(task_dir.glob("*_train_set.csv"))
#         if not csv_files: continue
            
#         csv_path = csv_files[0]
#         print(f"   Processing {task}...")
        
#         df = pd.read_csv(csv_path)
#         label_col = 'bioclass' if 'bioclass' in df.columns else df.columns[-1]
        
#         valid_count = 0
#         with torch.no_grad():
#             for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
#                 smi = str(row['SMILES']).strip() # Ensure it's a string
#                 label = float(row[label_col])
                
#                 try:
#                     # --- FIX: REGEX SPLIT BEFORE TOKENIZATION ---
#                     # 1. Split into chemical tokens [C, C, (, C, ), ...]
#                     token_list = regex.findall(smi)
                    
#                     # 2. Join with spaces so the tokenizer recognizes them
#                     spaced_smi = " ".join(token_list)
                    
#                     # 3. Encode (Now it will work!)
#                     tokens = vae.tokenizer.encode(spaced_smi, add_special_tokens=True)
                    
#                     # Safety check for empty or too long
#                     if len(tokens) == 0 or len(tokens) > 128: 
#                         continue 
                    
#                     token_tensor = torch.tensor([tokens], device=device)
                    
#                     # Call model forward pass
#                     if hasattr(vae.model, "module"):
#                         _, mu, _ = vae.model.module(token_tensor)
#                     else:
#                         _, mu, _ = vae.model(token_tensor)
                    
#                     z_vec = mu.cpu().numpy().flatten()
                    
#                     all_data.append({
#                         "z": z_vec,
#                         "y": label,
#                         "task_idx": task_idx
#                     })
#                     valid_count += 1
#                 except Exception as e:
#                     # Silent skip for weird molecules, but print if crucial
#                     continue
        
#         print(f"   ✅ Encoded {valid_count} molecules for {task}")

#     print(f"💾 Saving processed dataset to {output_file}...")
#     torch.save({
#         "tasks": tasks,
#         "data": all_data,
#         "latent_dim": len(all_data[0]['z']) if all_data else 128
#     }, output_file)
#     print("✨ Done!")

# if __name__ == "__main__":
#     # --- CONFIGURATION ---
#     # Update this path to your best checkpoint!
#     VAE_PATH = "./trained_vae/vae_weights.pt"
#     VOCAB_PATH = "./vocab.json"
#     DATA_ROOT = "./Models/AdmetClassifier/Auto_ML_dataset"
    
#     encode_datasets(DATA_ROOT, VAE_PATH, VOCAB_PATH)

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import re

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Generators.VAE import VAE

# --- CONFIG ---
BATCH_SIZE = 256  # Process 256 molecules at once (adjust based on GPU memory)
MAX_LEN = 1024     # Max length for padding

# Regex for SMILES splitting
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(SMI_REGEX_PATTERN)

def encode_datasets(data_root, vae_path, vocab_path, output_file="admet_latent_test_bidirec.pt"):
    print(f"🚀 Loading VAE from {vae_path}...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not found! This script requires a GPU for speed.")

    # Standard dynamic device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ VAE loading onto: {device}")
    
    vae = VAE(model_path=vae_path)
    vae.load_model(vocab_base=vocab_path)
    vae.model.to(device) # Force move to detected device
    vae.model.eval()
    print(f"✅ VAE loaded on {device}")
    
    # Get Pad Token ID
    pad_id = vae.tokenizer.pad_token_id if vae.tokenizer.pad_token_id is not None else 0

    root = Path(data_root)
    tasks = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    tasks.sort()
    
    print(f"📋 Found {len(tasks)} ADMET Tasks: {tasks}")
    
    all_data = []

    for task_idx, task in enumerate(tasks):
        task_dir = root / task
        csv_files = list(task_dir.glob("*_test_set.csv"))
        if not csv_files: continue
            
        csv_path = csv_files[0]
        print(f"   Processing {task}...")
        
        df = pd.read_csv(csv_path)
        label_col = 'bioclass' if 'bioclass' in df.columns else df.columns[-1]
        
        # Prepare lists for batching
        smiles_list = df['SMILES'].astype(str).tolist()
        labels_list = df[label_col].tolist()
        
        valid_count = 0
        
        # --- BATCH PROCESSING LOOP ---
        for i in tqdm(range(0, len(df), BATCH_SIZE), leave=False):
            batch_smiles = smiles_list[i : i + BATCH_SIZE]
            batch_labels = labels_list[i : i + BATCH_SIZE]
            
            tokenized_batch = []
            valid_indices = [] # Keep track of which ones in the batch are valid
            
            # 1. Tokenize CPU side
            for idx, smi in enumerate(batch_smiles):
                try:
                    smi_clean = smi.strip()
                    tokens = regex.findall(smi_clean)
                    spaced_smi = " ".join(tokens)
                    encoded = vae.tokenizer.encode(spaced_smi, add_special_tokens=True)
                    
                    if 0 < len(encoded) <= MAX_LEN:
                        tokenized_batch.append(encoded)
                        valid_indices.append(idx)
                except:
                    continue
            
            if not tokenized_batch:
                continue

            # 2. Pad Batch
            # Create a tensor of shape [Batch, Max_Len] filled with pad_id
            batch_tensor = torch.full((len(tokenized_batch), MAX_LEN), pad_id, dtype=torch.long)
            
            for j, seq in enumerate(tokenized_batch):
                seq_len = len(seq)
                batch_tensor[j, :seq_len] = torch.tensor(seq)
            
            batch_tensor = batch_tensor.to(device)
            
            # 3. Batch Inference (GPU)
            with torch.no_grad():
                # Forward pass returns (logits, mu, logvar)
                if hasattr(vae.model, "module"):
                    _, mu, _ = vae.model.module(batch_tensor)
                else:
                    _, mu, _ = vae.model(batch_tensor)
                
                # Move to CPU
                z_batch = mu.cpu().numpy()
            
            # 4. Store Results
            for j, original_idx in enumerate(valid_indices):
                all_data.append({
                    "z": z_batch[j],          # The latent vector
                    "y": float(batch_labels[original_idx]), # The label
                    "task_idx": task_idx
                })
                valid_count += 1
                
        print(f"   ✅ Encoded {valid_count} molecules for {task}")

    print(f"💾 Saving processed dataset to {output_file}...")
    torch.save({
        "tasks": tasks,
        "data": all_data,
        "latent_dim": len(all_data[0]['z']) if all_data else 128
    }, output_file)
    print("✨ Done!")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update this path to your best checkpoint!
    VAE_PATH = "./trained_vae/vae_weights_bidirec.pt"
    VOCAB_PATH = "./vocab.json"
    DATA_ROOT = "./Models/AdmetClassifier/Auto_ML_dataset"
    
    encode_datasets(DATA_ROOT, VAE_PATH, VOCAB_PATH)