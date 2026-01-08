import torch
import numpy as np
from Generators.VAE import VAE
import re
import sys

# Standard SMILES Regex
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(SMI_REGEX_PATTERN)

def check_vae_health(model_path, vocab_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🩺 Diagnostic Mode | Device: {device}")
    
    vae = VAE(model_path=model_path)
    vae.load_model(vocab_base=vocab_path)
    vae.model.to(device).eval()
    
    # Real drug SMILES for testing
    test_smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",     # Aspirin
        "c1ccccc1",                     # Benzene
        "O=C(NCc1ccccc1)c1ccccc1",      # Simple Amide
        "CCN(CC)CC"                     # Triethylamine
    ] * 20 

    # 1. Tokenize with Regex
    tokens_list = []
    for smi in test_smiles:
        split_smi = " ".join(regex.findall(smi))
        ids = vae.tokenizer.encode(split_smi, add_special_tokens=True)
        tokens_list.append(torch.tensor(ids))
            
    max_len = max(len(t) for t in tokens_list)
    padded_batch = torch.zeros(len(tokens_list), max_len, dtype=torch.long).to(device)
    for i, t in enumerate(tokens_list):
        padded_batch[i, :len(t)] = t
    
    # 2. Extract Latents (KL only needs Mu and LogVar)
    with torch.no_grad():
        # Calling the encoder directly to avoid the Decoder RNN crash
        if hasattr(vae.model, "encoder"):
            mu, logvar = vae.model.encoder(padded_batch)
        else:
            # If your model doesn't expose .encoder, we use the forward's internal logic
            # but we only take the latent outputs
            _, mu, logvar = vae.model(padded_batch)
            
        # 3. Calculate KL
        # Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        avg_kl = kl_div.mean().item()
        
        # Calculate Active Units (dims with meaningful variance)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
        active_units = (kl_per_dim > 0.01).sum().item()

    print("\n" + "="*40)
    print(f"📊 KL DIAGNOSTIC REPORT")
    print("="*40)
    print(f"ACTUAL KL DIVERGENCE: {avg_kl:.6f}")
    print(f"ACTIVE DIMENSIONS:    {active_units} / 128")
    print("="*40)

    if avg_kl < 0.1:
        print("🚨 RESULT: POSTERIOR COLLAPSE DETECTED")
        print("Your latent space is empty noise. ADMET will never learn.")
    elif avg_kl < 1.0:
        print("⚠️ RESULT: WEAK LATENT SIGNAL")
        print("Very hard for ADMET to distinguish molecules.")
    else:
        print("✅ RESULT: HEALTHY LATENT SPACE")

if __name__ == "__main__":
    # Point this to your CURRENT model (the one failing ADMET training)
    MODEL_PATH = "./trained_vave/vae_weights.pt" 
    VOCAB_PATH = "./vocab.json"
    
    check_vae_health(MODEL_PATH, VOCAB_PATH)