# # import torch
# # import torch.optim as optim
# # import sys
# # import os
# # import re

# # # Add root to path
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# # # 1. Import the Wrapper (The smart way)
# # from Generators.VAE import VAE 
# # from Models.AdmetClassifier.train_multitask import MultiHeadADMET

# # def optimize_molecule(vae_path, admet_path, vocab_path, target_task="BBBP", minimize=False):
# #     # --- 1. Setup Device & Load VAE (Using your pattern) ---
# #     print(f"🚀 Loading VAE from {vae_path}...")
    
# #     if not torch.cuda.is_available():
# #         raise RuntimeError("❌ CUDA not found! This script requires a GPU.")

# #     device = torch.device("cuda")
# #     print(f"✅ Optimization running on: {device}")
    
# #     # Use the Wrapper to handle vocab sizes automatically
# #     vae = VAE(model_path=vae_path)
# #     vae.load_model(vocab_base=vocab_path)
# #     vae.model.to(device) 
# #     vae.model.eval()
# #     print(f"✅ VAE Loaded Successfully (Latent Dim: {vae.model.latent_dim})")
    
# #     # --- 2. Load ADMET Predictor ---
# #     print(f"🧠 Loading ADMET Predictor from: {admet_path}")
# #     admet_ckpt = torch.load(admet_path, map_location=device)
# #     task_names = admet_ckpt['task_names']
    
# #     predictor = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names))
# #     predictor.load_state_dict(admet_ckpt['model_state'])
# #     predictor.to(device).eval()
    
# #     if target_task not in task_names:
# #         print(f"❌ Target {target_task} not found. Available: {task_names}")
# #         return

# #     task_idx = task_names.index(target_task)
# #     goal_str = "Minimize" if minimize else "Maximize"
# #     print(f"🎯 Goal: {goal_str} {target_task} (Task Index: {task_idx})")

# #     print("\n🔍 Searching for a weak starting molecule (< 10% probability)...")
    
# #     start_z = None
    
# #     # Try up to 100 times to find a negative sample
# #     for i in range(100):
# #         # Generate random z
# #         candidate_z = torch.randn(1, 128, device=device)
        
# #         # Check Score
# #         with torch.no_grad():
# #             logits = predictor(candidate_z)
# #             prob = torch.sigmoid(logits[:, task_idx]).item()
        
# #         # If we found a "bad" molecule (Low BBBP score), keep it
# #         if (not minimize and prob < 0.10) or (minimize and prob > 0.90):
# #             start_z = candidate_z
# #             print(f"✅ Found negative start at attempt {i+1} (Score: {prob*100:.2f}%)")
# #             break
    
# #     if start_z is None:
# #         print("⚠️ Could not find negative sample. Starting with random.")
# #         start_z = torch.randn(1, 128, device=device)

# #     # Prepare for Optimization (Leaf Tensor Logic)
# #     z = start_z.clone()
# #     z.requires_grad_(True)
    
# #     # Use Slower LR for smoother transition
# #     optimizer = optim.Adam([z], lr=0.0025) 
    
# #     print("\n--- Starting Transformation ---")
# #     start_smi = decode_z_safe(vae, z, temp=0.7)
# #     print(f"❌ START (Bad Drug): {start_smi}")

# #     best_z = z.clone()
# #     best_prob = 0.0

# #     for step in range(100):
# #         optimizer.zero_grad()
        
# #         logits = predictor(z)
# #         target_logit = logits[:, task_idx]
# #         prob = torch.sigmoid(target_logit)
        
# #         # Regularization (Leash)
# #         prior_loss = (z ** 2).mean()
# #         penalty_weight = 20.0 # Keep it tight
        
# #         if minimize:
# #             loss = target_logit + (penalty_weight * prior_loss)
# #         else:
# #             loss = -target_logit + (penalty_weight * prior_loss)
            
# #         loss.backward()
        
# #         # Clip Gradients to prevent exploding molecule structure
# #         torch.nn.utils.clip_grad_norm_([z], 0.05)
        
# #         optimizer.step()
        
# #         current_prob = prob.item()
# #         current_prior = prior_loss.item()
        
# #         if step % 1 == 0:
# #             print(f"Step {step:02d} | Score: {current_prob*100:.2f}% | Penalty: {current_prior:.4f}")

# #         if current_prob > 0.90 and current_prior < 1.0:
# #             # Decode to check if it looks sane (heuristic: no triple #'s in weird places)
# #             smi_check = decode_z_safe(vae, z, temp=0.7)
# #             if len(smi_check) > 5: # Basic length check
# #                 print(f"✅ Valid Success at Step {step}!")
# #                 best_z = z.clone()
# #                 break
            
# #         if current_prob > best_prob and current_prior < 1.2:
# #             best_z = z.clone()
# #             best_prob = current_prob

# #     # --- 4. Final Result ---
# #     final_prob = torch.sigmoid(predictor(best_z)[:, task_idx]).item()
# #     final_smi = decode_z_safe(vae, best_z, temp=0.7) 
    
# #     print("\n" + "="*50)
# #     print(f"🧪 DRUG DISCOVERY COMPLETE")
# #     print("="*50)
# #     print(f"Task:      {target_task}")
# #     print(f"Evolution: ❌ Bad -> ✅ Good ({final_prob*100:.2f}%)")
# #     print(f"SMILES:    {final_smi}")
# #     print("="*50)

# # # Update helper to accept temp
# # def decode_z_safe(vae, z, temp=1.0):
# #     with torch.no_grad():
# #         ids = vae.model.sample(
# #             max_len=100, 
# #             start_token_idx=vae.tokenizer.bos_token_id, 
# #             tokenizer=vae.tokenizer, 
# #             device=vae.device, 
# #             z=z,
# #             temp=temp # Pass the temperature
# #         )
# #     clean_ids = [i for i in ids if i not in [vae.tokenizer.bos_token_id, vae.tokenizer.eos_token_id, vae.tokenizer.pad_token_id]]
# #     return vae.tokenizer.decode(clean_ids, skip_special_tokens=True).replace(" ", "")

# # if __name__ == "__main__":
# #     # --- CONFIGURATION ---
# #     # Use the path that worked for your dataset script
# #     VAE_PATH = "trained_vae/vae_weights_bidirec.pt"
# #     ADMET_PATH = "admet_predictor_bidirec.pt"
# #     VOCAB_PATH = "vocab.json"
    
# #     # Example: Find a molecule that penetrates the Blood-Brain Barrier
# #     optimize_molecule(VAE_PATH, ADMET_PATH, VOCAB_PATH, target_task="CYP1A2_inhibition", minimize=False)
# #     # optimize_molecule(VAE_PATH, ADMET_PATH, VOCAB_PATH, target_task="BBBP", minimize=False)

# import torch
# import torch.optim as optim
# import sys
# import os
# import numpy as np

# # --- 1. RDKit Setup ---
# try:
#     from rdkit import Chem
#     RDKIT_AVAILABLE = True
# except ImportError:
#     RDKIT_AVAILABLE = False
#     print("⚠️ RDKit not found. Validation will be weak.")

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# from Generators.VAE import VAE 
# from Models.AdmetClassifier.train_multitask import MultiHeadADMET

# def is_valid_molecule(smi):
#     if len(smi) < 5: return False
#     if RDKIT_AVAILABLE:
#         try:
#             mol = Chem.MolFromSmiles(smi)
#             return mol is not None
#         except:
#             return False
#     return True

# def optimize_batch(vae_path, admet_path, vocab_path, target_task="BBBP"):
#     # CONFIG
#     BATCH_SIZE = 64        # Run 64 optimizations at once!
#     TARGET_PROB = 0.95     # Aim for 95%, not 100% (prevents exploding gradients)
#     LR = 0.02              # Moderate speed
#     STEPS = 100
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🚀 Launching SWARM Optimization ({BATCH_SIZE} agents) on {device}...")

#     # Load Models
#     vae = VAE(model_path=vae_path)
#     vae.load_model(vocab_base=vocab_path)
#     vae.model.to(device).eval()
    
#     admet_ckpt = torch.load(admet_path, map_location=device)
#     task_names = admet_ckpt['task_names']
#     predictor = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names))
#     predictor.load_state_dict(admet_ckpt['model_state'])
#     predictor.to(device).eval()
    
#     if target_task not in task_names: return
#     task_idx = task_names.index(target_task)
#     print(f"🎯 Goal: Transform {BATCH_SIZE} random molecules to match {target_task}")

#     # --- 1. Batch Initialization ---
#     # Find negative samples to start with
#     print("🔍 Initializing swarm with low-scoring molecules...")
    
#     # Generate a massive batch and pick the worst ones
#     temp_z = torch.randn(BATCH_SIZE * 10, 128, device=device)
#     with torch.no_grad():
#         logits = predictor(temp_z)
#         probs = torch.sigmoid(logits[:, task_idx])
        
#     # Sort and pick the lowest probability ones (The "Sick" Patients)
#     # We want to cure them
#     vals, indices = torch.sort(probs)
#     start_z = temp_z[indices[:BATCH_SIZE]].clone() # Top 64 worst
    
#     print(f"📉 Average Start Score: {vals[:BATCH_SIZE].mean().item()*100:.2f}%")
    
#     # --- 2. Optimization Loop ---
#     z = start_z.clone()
#     z.requires_grad_(True)
#     optimizer = optim.Adam([z], lr=LR)
    
#     for step in range(STEPS):
#         optimizer.zero_grad()
        
#         logits = predictor(z)
#         probs = torch.sigmoid(logits[:, task_idx])
        
#         # --- LOSS FUNCTION (MSE TARGET) ---
#         # Instead of maximizing to infinity, minimize distance to 0.95
#         # This keeps the z vector stable
#         target_tensor = torch.full_like(probs, TARGET_PROB)
#         score_loss = torch.nn.functional.mse_loss(probs, target_tensor)
        
#         # Regularization (Keep them close to origin)
#         prior_loss = (z ** 2).mean()
        
#         loss = score_loss + (0.5 * prior_loss) # Balance scoring vs staying valid
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_([z], 0.1) # Clip gradients
#         optimizer.step()
        
#         if step % 20 == 0:
#             avg_score = probs.mean().item()
#             print(f"Step {step:03d} | Avg Swarm Score: {avg_score*100:.2f}%")

#     # --- 3. Filtering & Selection ---
#     print("\n⚔️  Filtering Survivors...")
    
#     final_z = z.detach()
#     with torch.no_grad():
#         final_probs = torch.sigmoid(predictor(final_z)[:, task_idx])
    
#     valid_candidates = []
    
#     # Decode all 64 candidates
#     # This might take a second, but it's worth it
#     decoded_list = decode_batch(vae, final_z)
    
#     for i in range(BATCH_SIZE):
#         smi = decoded_list[i]
#         score = final_probs[i].item()
        
#         # Criteria: High Score AND Valid SMILES
#         if score > 0.85 and is_valid_molecule(smi):
#             valid_candidates.append((score, smi))

#     # Sort by score
#     valid_candidates.sort(key=lambda x: x[0], reverse=True)

#     print("\n" + "="*50)
#     print(f"🏆 TOP RESULTS ({len(valid_candidates)}/{BATCH_SIZE} Valid)")
#     print("="*50)
    
#     if not valid_candidates:
#         print("❌ No valid molecules found. Try tuning penalty.")
#     else:
#         # Show top 3
#         for rank, (score, smi) in enumerate(valid_candidates[:5]):
#             print(f"Rank {rank+1} | Score: {score*100:.2f}% | SMILES: {smi}")
#     print("="*50)

# def decode_batch(vae, z_batch):
#     # Decode a whole batch at once
#     # We loop simply because the sample method usually takes 1 z at a time
#     # unless rewritten. For safety, simple loop:
#     res = []
#     for i in range(z_batch.size(0)):
#         # Temp 0.7 for stability
#         with torch.no_grad():
#             ids = vae.model.sample(100, vae.tokenizer.bos_token_id, vae.tokenizer, vae.device, z=z_batch[i:i+1], temp=0.7)
#         clean = [x for x in ids if x not in [vae.tokenizer.bos_token_id, vae.tokenizer.eos_token_id, vae.tokenizer.pad_token_id]]
#         res.append(vae.tokenizer.decode(clean, skip_special_tokens=True).replace(" ", ""))
#     return res

# if __name__ == "__main__":
#     VAE_PATH = "trained_vae/vae_weights_bidirec.pt"
#     ADMET_PATH = "admet_predictor_bidirec.pt"
#     VOCAB_PATH = "vocab.json"
    
#     optimize_batch(VAE_PATH, ADMET_PATH, VOCAB_PATH, target_task="CYP1A2_inhibition")


import torch
import torch.optim as optim
import sys
import os
import numpy as np
import pickle  # <--- Added for saving the pkl file

# --- 1. RDKit Setup ---
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not found. Validation will be weak.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Generators.VAE import VAE 
from Models.AdmetClassifier.train_multitask import MultiHeadADMET

def is_valid_molecule(smi):
    if len(smi) < 5: return False
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smi)
            return mol is not None
        except:
            return False
    return True

def optimize_batch(vae_path, admet_path, vocab_path, target_task="BBBP"):
    # CONFIG
    BATCH_SIZE = 64        # Run 64 optimizations at once!
    TARGET_PROB = 0.95     # Aim for 95%, not 100% (prevents exploding gradients)
    LR = 0.02              # Moderate speed
    STEPS = 100
    TRAJECTORY_FILE = "optimization_trajectory.pkl" # <--- Added filename
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching SWARM Optimization ({BATCH_SIZE} agents) on {device}...")

    # Load Models
    vae = VAE(model_path=vae_path)
    vae.load_model(vocab_base=vocab_path)
    vae.model.to(device).eval()
    
    admet_ckpt = torch.load(admet_path, map_location=device)
    task_names = admet_ckpt['task_names']
    predictor = MultiHeadADMET(latent_dim=128, num_tasks=len(task_names))
    predictor.load_state_dict(admet_ckpt['model_state'])
    predictor.to(device).eval()
    
    if target_task not in task_names: return
    task_idx = task_names.index(target_task)
    print(f"🎯 Goal: Transform {BATCH_SIZE} random molecules to match {target_task}")

    # --- 1. Batch Initialization ---
    # Find negative samples to start with
    print("🔍 Initializing swarm with low-scoring molecules...")
    
    # Generate a massive batch and pick the worst ones
    temp_z = torch.randn(BATCH_SIZE * 10, 128, device=device)
    with torch.no_grad():
        logits = predictor(temp_z)
        probs = torch.sigmoid(logits[:, task_idx])
        
    # Sort and pick the lowest probability ones (The "Sick" Patients)
    # We want to cure them
    vals, indices = torch.sort(probs)
    start_z = temp_z[indices[:BATCH_SIZE]].clone() # Top 64 worst
    
    print(f"📉 Average Start Score: {vals[:BATCH_SIZE].mean().item()*100:.2f}%")
    
    # --- 2. Optimization Loop ---
    z = start_z.clone()
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=LR)
    
    history = [] # <--- Added history list

    for step in range(STEPS):
        optimizer.zero_grad()
        
        logits = predictor(z)
        probs = torch.sigmoid(logits[:, task_idx])
        
        # --- LOSS FUNCTION (MSE TARGET) ---
        # Instead of maximizing to infinity, minimize distance to 0.95
        # This keeps the z vector stable
        target_tensor = torch.full_like(probs, TARGET_PROB)
        score_loss = torch.nn.functional.mse_loss(probs, target_tensor)
        
        # Regularization (Keep them close to origin)
        prior_loss = (z ** 2).mean()
        
        loss = score_loss + (0.5 * prior_loss) # Balance scoring vs staying valid
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 0.1) # Clip gradients
        optimizer.step()
        
        # --- ADDED LOGGING BLOCK START ---
        with torch.no_grad():
            avg_score = probs.mean().item()
            best_idx = torch.argmax(probs).item()
            best_score = probs[best_idx].item()
            
            # Helper to decode just the single best molecule for the plot
            # (Using your decode logic but for 1 item to be fast)
            best_z_tensor = z[best_idx:best_idx+1]
            ids = vae.model.sample(100, vae.tokenizer.bos_token_id, vae.tokenizer, vae.device, z=best_z_tensor, temp=0.7)
            clean = [x for x in ids if x not in [vae.tokenizer.bos_token_id, vae.tokenizer.eos_token_id, vae.tokenizer.pad_token_id]]
            best_smiles = vae.tokenizer.decode(clean, skip_special_tokens=True).replace(" ", "")

            history.append({
                "step": step,
                "score": best_score,
                "avg_score": avg_score,
                "z_vector": best_z_tensor.cpu().numpy(),
                "smiles": best_smiles
            })
        # --- ADDED LOGGING BLOCK END ---

        if step % 20 == 0:
            print(f"Step {step:03d} | Avg Swarm Score: {avg_score*100:.2f}%")

    # --- 3. Filtering & Selection ---
    print("\n⚔️  Filtering Survivors...")
    
    # --- ADDED SAVE TRAJECTORY ---
    with open(TRAJECTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    print(f"✅ Visualization data saved to {TRAJECTORY_FILE}")
    # -----------------------------

    final_z = z.detach()
    with torch.no_grad():
        final_probs = torch.sigmoid(predictor(final_z)[:, task_idx])
    
    valid_candidates = []
    
    # Decode all 64 candidates
    # This might take a second, but it's worth it
    decoded_list = decode_batch(vae, final_z)
    
    for i in range(BATCH_SIZE):
        smi = decoded_list[i]
        score = final_probs[i].item()
        
        # Criteria: High Score AND Valid SMILES
        if score > 0.85 and is_valid_molecule(smi):
            valid_candidates.append((score, smi))

    # Sort by score
    valid_candidates.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "="*50)
    print(f"🏆 TOP RESULTS ({len(valid_candidates)}/{BATCH_SIZE} Valid)")
    print("="*50)
    
    if not valid_candidates:
        print("❌ No valid molecules found. Try tuning penalty.")
    else:
        # Show top 3
        for rank, (score, smi) in enumerate(valid_candidates[:5]):
            print(f"Rank {rank+1} | Score: {score*100:.2f}% | SMILES: {smi}")
    print("="*50)

def decode_batch(vae, z_batch):
    # Decode a whole batch at once
    # We loop simply because the sample method usually takes 1 z at a time
    # unless rewritten. For safety, simple loop:
    res = []
    for i in range(z_batch.size(0)):
        # Temp 0.7 for stability
        with torch.no_grad():
            ids = vae.model.sample(100, vae.tokenizer.bos_token_id, vae.tokenizer, vae.device, z=z_batch[i:i+1], temp=0.7)
        clean = [x for x in ids if x not in [vae.tokenizer.bos_token_id, vae.tokenizer.eos_token_id, vae.tokenizer.pad_token_id]]
        res.append(vae.tokenizer.decode(clean, skip_special_tokens=True).replace(" ", ""))
    return res

if __name__ == "__main__":
    VAE_PATH = "trained_vae/vae_weights_bidirec.pt"
    ADMET_PATH = "admet_predictor_bidirec.pt"
    VOCAB_PATH = "vocab.json"
    
    optimize_batch(VAE_PATH, ADMET_PATH, VOCAB_PATH, target_task="CYP1A2_inhibition")