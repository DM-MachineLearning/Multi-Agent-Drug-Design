# # import pandas as pd
# # import ast
# # import torch
# # import sys
# # import os
# # import re

# # # --- CONFIG ---
# # CSV_PATH = "outputs/exploration_updateMeanVar_50update.csv"
# # ADMET_CKPT = "admet_predictor_bidirec_1000epochs.pt"
# # ACTIVITY_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
# # VAE_PATH = "./trained_vae/vae_weights_bidirec.pt"
# # VOCAB_PATH = "./vocab.json"

# # # ⚠️ CRITICAL: MUST MATCH TRAINING MAX_LEN
# # # Try 100, 128, or 1024. If you aren't sure, 128 is the standard default for many chemical VAEs.
# # MAX_LEN = 128  

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# # from utils.ScoringEngine import ScoringEngine
# # from Generators.VAE import VAE

# # # Regex for SMILES tokenization
# # SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# # regex = re.compile(SMI_REGEX_PATTERN)

# # def verify_scores_with_padding():
# #     # 1. LOAD DATA
# #     if not os.path.exists(CSV_PATH):
# #         print(f"❌ File not found: {CSV_PATH}")
# #         return

# #     df = pd.read_csv(CSV_PATH)
# #     print(f"✅ Loaded {len(df)} rows.")

# #     # 2. SETUP MODELS
# #     print("⚙️ Loading Models...")
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# #     vae = VAE(model_path=VAE_PATH)
# #     vae.load_model(vocab_base=VOCAB_PATH)
# #     vae.model.to(device).eval()
    
# #     engine = ScoringEngine(activity_classifier_path=ACTIVITY_PATH, admet_model_path=ADMET_CKPT)
# #     engine.admet_classifier_model.model.eval()

# #     print(f"\n⚡ CHECKING DRIFT (Padding to MAX_LEN={MAX_LEN})...")
# #     print(f"{'SMILES (Partial)':<20} | {'OLD hERG':<10} | {'NEW hERG':<10} | {'DIFF':<10} | {'STATUS'}")
# #     print("-" * 80)

# #     for i, row in df.head(20).iterrows():
# #         smiles = str(row.get('smiles', row.get('Smiles', '')))
        
# #         # A. EXTRACT OLD SCORE (Simple Regex)
# #         raw_text = str(row.get('captions', row.get('scores', '')))
# #         match = re.search(r"'hERG_inhibition':\s*(?:tensor\s*\(\s*)?([0-9\.]+(?:e[-+]?\d+)?)", raw_text)
# #         old_herg = float(match.group(1)) if match else None
        
# #         if old_herg is None: continue

# #         # B. CALCULATE NEW SCORE (With Strict Pre-processing)
# #         try:
# #             # 1. Regex Split
# #             tokens = regex.findall(smiles)
# #             smi_str = " ".join(tokens)
            
# #             # 2. Encode
# #             input_ids = vae.tokenizer.encode(smi_str, add_special_tokens=True)
            
# #             # 3. CRITICAL FIX: PADDING
# #             # Create a tensor of Full MAX_LEN filled with Pad Tokens
# #             pad_id = vae.tokenizer.pad_token_id or 0
# #             padded_ids = torch.full((1, MAX_LEN), pad_id, dtype=torch.long, device=device)
            
# #             # Fill the beginning with actual data
# #             seq_len = min(len(input_ids), MAX_LEN)
# #             padded_ids[0, :seq_len] = torch.tensor(input_ids[:seq_len], device=device)
            
# #             # 4. Get Z
# #             with torch.no_grad():
# #                 # Pass the padded tensor
# #                 _, mu, _ = vae.model(padded_ids)
# #                 z_new = mu[0]

# #             # 5. Score
# #             new_scores = engine.get_all_scores(z_new)
# #             new_herg = new_scores['hERG_inhibition']
# #             if torch.is_tensor(new_herg): new_herg = new_herg.item()

# #             # Report
# #             diff = new_herg - old_herg
# #             status = "✅ Match" if abs(diff) < 0.1 else "⚠️ Drift"
# #             print(f"{smiles[:15]:<20} | {old_herg:.4f}     | {new_herg:.4f}     | {diff:+.4f}     | {status}")

# #         except Exception as e:
# #             print(f"{smiles[:15]:<20} | ERROR: {e}")

# # if __name__ == "__main__":
# #     verify_scores_with_padding()

# import pandas as pd
# import torch
# import sys
# import os
# import re
# import numpy as np

# # --- CONFIG ---
# CSV_PATH = "outputs/exploration_updateMeanVar_50update.csv"
# VAE_PATH = "./trained_vae/vae_weights_bidirec.pt"
# VOCAB_PATH = "./vocab.json"
# MAX_LEN = 128  # Keep this consistent

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Generators.VAE import VAE

# # Regex Pattern
# SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# regex = re.compile(SMI_REGEX_PATTERN)

# def diagnose_round_trip():
#     # 1. SETUP
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vae = VAE(model_path=VAE_PATH)
#     vae.load_model(vocab_base=VOCAB_PATH)
#     vae.model.to(device).eval()
    
#     # 2. LOAD DATA
#     if not os.path.exists(CSV_PATH): return
#     df = pd.read_csv(CSV_PATH)
#     print(f"Loaded {len(df)} rows. Testing reconstruction on first 10...")

#     print(f"\n{'METHOD':<10} | {'ORIGINAL (Partial)':<20} | {'RECONSTRUCTED (Partial)':<20} | {'STATUS'}")
#     print("-" * 80)

#     for i, row in df.head(10).iterrows():
#         original_smi = str(row.get('smiles', row.get('Smiles', '')))
#         if len(original_smi) < 2: continue
        
#         # --- TEST 1: RAW TOKENIZATION (Standard) ---
#         # Just pass the string. The tokenizer handles the splitting.
#         try:
#             tokens_raw = vae.tokenizer.encode(original_smi, add_special_tokens=True)
#             z_raw = get_latent(vae, tokens_raw, MAX_LEN, device)
#             recon_raw = decode_latent(vae, z_raw)
            
#             match_raw = (original_smi == recon_raw)
#             status_raw = "✅ Perfect" if match_raw else "❌ Lost"
#             print(f"{'RAW':<10} | {original_smi[:15]:<20} | {recon_raw[:15]:<20} | {status_raw}")
#         except Exception as e:
#             print(f"RAW Error: {e}")

#         # --- TEST 2: REGEX TOKENIZATION (Your previous method) ---
#         # Manually split and join with spaces
#         try:
#             tokens_list = regex.findall(original_smi)
#             spaced_smi = " ".join(tokens_list)
#             tokens_regex = vae.tokenizer.encode(spaced_smi, add_special_tokens=True)
#             z_regex = get_latent(vae, tokens_regex, MAX_LEN, device)
#             recon_regex = decode_latent(vae, z_regex)
            
#             match_regex = (original_smi == recon_regex)
#             status_regex = "✅ Perfect" if match_regex else "❌ Lost"
#             print(f"{'REGEX':<10} | {original_smi[:15]:<20} | {recon_regex[:15]:<20} | {status_regex}")
#         except Exception as e:
#             print(f"REGEX Error: {e}")
            
#         print("-" * 80)

# def get_latent(vae, token_ids, max_len, device):
#     # Pad to specific length to match training architecture
#     padded = torch.full((1, max_len), vae.tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
#     seq_len = min(len(token_ids), max_len)
#     padded[0, :seq_len] = torch.tensor(token_ids[:seq_len], device=device)
    
#     with torch.no_grad():
#         _, mu, _ = vae.model(padded)
#     return mu[0] # Return z

# def decode_latent(vae, z):
#     # Use the VAE's own generation logic to see what this z maps to
#     smi, _ = vae.generate_molecule(z=z.unsqueeze(0))
#     return smi

# if __name__ == "__main__":
#     diagnose_round_trip()

import pandas as pd
import re
import sys

# --- CONFIG ---
CSV_PATH = "outputs/exploration_updateMeanVar_50update.csv"

def get_true_success_rate():
    print(f"📂 Analyzing: {CSV_PATH}")
    
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    total = len(df)
    pass_count = 0
    
    # Regex to extract hERG value even from "tensor(...)" strings
    # Looks for: 'hERG_inhibition': <optional tensor stuff> <NUMBER>
    pattern = re.compile(r"'hERG_inhibition':\s*(?:tensor\s*\(\s*)?([0-9\.]+(?:e[-+]?\d+)?)")

    print(f"📊 Scanning {total} molecules...")
    
    for i, row in df.iterrows():
        # Get the caption/score string
        caption_str = str(row.get('captions', row.get('scores', '')))
        
        match = pattern.search(caption_str)
        if match:
            val = float(match.group(1))
            # Check if it passes your threshold (Low < 0.3)
            if val < 0.3:
                pass_count += 1
        else:
            # If we can't find a score, assume it failed or print error
            # print(f"Row {i}: Could not parse score.")
            pass

    success_rate = (pass_count / total) * 100
    
    print("\n" + "="*40)
    print(f"🏆 TRUE SUCCESS RATE REPORT")
    print("="*40)
    print(f"Total Molecules: {total}")
    print(f"Passing hERG (<0.3): {pass_count}")
    print("-" * 40)
    print(f"✅ REAL SUCCESS RATE: {success_rate:.2f}%")
    print("="*40)
    print("\nNOTE: This metric trusts the Agent's internal scoring at the moment of generation.")
    print("Since your VAE cannot reconstruct molecules (Encoder failure), this is the ONLY valid metric.")

if __name__ == "__main__":
    get_true_success_rate()