import torch
from rdkit import Chem
import sys
import os
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Generators.VAE import VAE

def evaluate_final_metrics():
    # --- CONFIG ---
    MODEL_PATH = "trained_transformer_vae/vae_weights.pt"
    VOCAB_PATH = "vocab.json"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_SAMPLES = 10

    # 1. Load VAE (Defaults will now match the 512-dim weights)
    vae = VAE(model_path=MODEL_PATH)
    vae.load_model(vocab_base=VOCAB_PATH, model_type="transformer")
    vae.model.to(DEVICE)
    vae.model.eval()

    print(f"🧪 Evaluating ULTIMATE 512-dim Transformer ({NUM_SAMPLES} samples)...")

    valid_smiles = []
    all_smiles = []

    for _ in (range(NUM_SAMPLES)):
        z = torch.randn(1, 512).to(DEVICE)
        smi, _ = vae.generate_molecule(z=z)
        print(smi)
        
        if smi:
            all_smiles.append(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_smiles.append(smi)

    validity_rate = (len(valid_smiles) / NUM_SAMPLES) * 100
    uniqueness = (len(set(valid_smiles)) / len(valid_smiles)) * 100 if valid_smiles else 0
    
    print(f"\n💎 FINAL PERFORMANCE REPORT:")
    print(f"==============================")
    print(f"✅ SMILES Validity: {validity_rate:.2f}%")
    print(f"🌈 SMILES Uniqueness: {uniqueness:.2f}%")
    print(f"==============================")
    print(f"📝 Generated Samples (First 10 Valid):")
    for i in range(min(10, len(valid_smiles))):
        print(f"   {i+1}. {valid_smiles[i]}")

if __name__ == "__main__":
    evaluate_final_metrics()
