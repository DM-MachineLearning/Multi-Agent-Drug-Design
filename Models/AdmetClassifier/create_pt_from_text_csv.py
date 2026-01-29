import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import re

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Generators.VAE import VAE

# --- CONFIG ---
BATCH_SIZE = 256
MAX_LEN = 1024
SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(SMI_REGEX_PATTERN)

def get_smiles_list(file_path):
    path = Path(file_path)
    smiles_list = []
    if path.suffix == '.csv':
        try:
            df = pd.read_csv(path)
            for col in ['Smiles', 'SMILES', 'smiles']:
                if col in df.columns: return df[col].astype(str).tolist()
        except: pass

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('Smiles,'): continue
            parts = line.split(',') if ',' in line else line.split()
            if parts and len(parts[0]) > 1: smiles_list.append(parts[0])
    return smiles_list

def encode_file(input_file, vae, device, output_dir, global_tasks):
    input_path = Path(input_file)
    output_path = output_dir / f"{input_path.stem}.pt"
    smiles_list = get_smiles_list(input_path)
    
    if not smiles_list: return

    pad_id = vae.tokenizer.pad_token_id if vae.tokenizer.pad_token_id is not None else 0
    all_data = []
    
    for i in tqdm(range(0, len(smiles_list), BATCH_SIZE), desc=f"Encoding {input_path.stem}"):
        batch_smiles = smiles_list[i : i + BATCH_SIZE]
        tokenized_batch, valid_indices = [], []

        for idx, smi in enumerate(batch_smiles):
            try:
                tokens = regex.findall(smi)
                encoded = vae.tokenizer.encode(" ".join(tokens), add_special_tokens=True)
                if 0 < len(encoded) <= MAX_LEN:
                    tokenized_batch.append(encoded)
                    valid_indices.append(idx)
            except: continue
        
        if not tokenized_batch: continue

        batch_tensor = torch.full((len(tokenized_batch), MAX_LEN), pad_id, dtype=torch.long).to(device)
        for j, seq in enumerate(tokenized_batch):
            batch_tensor[j, :len(seq)] = torch.tensor(seq)
        
        with torch.no_grad():
            model_to_use = vae.model.module if hasattr(vae.model, "module") else vae.model
            _, mu, _ = model_to_use(batch_tensor)
            z_batch = mu.cpu().numpy()

        for j, original_idx in enumerate(valid_indices):
            all_data.append({
                "z": z_batch[j],
                "y": 0.0,           # Dummy label: benchmarks don't have ground truth labels
                "task_idx": 0       # Map to the first task by default
            })

    # --- CRITICAL FIX FOR YOUR ERROR ---
    # We save the dictionary with the 'tasks' key your LatentDataset is looking for
    torch.save({
        "tasks": global_tasks,      # This resolves the KeyError: 'tasks'
        "data": all_data,           # List of dicts with 'z', 'y', and 'task_idx'
        "latent_dim": len(all_data[0]['z']) if all_data else 128
    }, output_path)
    print(f"✅ Saved to {output_path}")

def main():
    VAE_PATH = "./trained_vae/vae_weights_bidirec.pt"
    VOCAB_PATH = "./vocab.json"
    BENCHMARK_DIR = Path("./Benchmarks_new")
    ADMET_CKPT = "admet_predictor_bidirec_1000epochs.pt"
    # ADMET_CKPT = "admet_predictor_bidirec.pt"

    # Load task names from your trained ADMET predictor to keep indices consistent
    print("🔍 Extracting task names from ADMET checkpoint...")
    ckpt = torch.load(ADMET_CKPT, map_location='cpu', weights_only=False)
    global_tasks = ckpt['task_names'] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(model_path=VAE_PATH)
    vae.load_model(vocab_base=VOCAB_PATH)
    vae.model.to(device).eval()

    files = list(BENCHMARK_DIR.glob("*.csv")) + list(BENCHMARK_DIR.glob("*.txt"))
    for file_path in files:
        if file_path.suffix == '.pt': continue
    # file_path_new = "outputs/exploration_updateMeanVar_50update_CLEAN.csv"
    # file_path_new = "test.csv"
    # file_path_new = "outputs/run1.csv"
    # file_path_new = "Benchmarks/rnn_smiles.txt"
    # file_path_new = "Benchmarks/molgan_15k_smiles.csv"
    # file_path_new = "Benchmarks/bimodal.csv"
    # file_path_new = "Benchmarks/organ.csv"
    file_path_new = "Benchmarks/generated1.csv"
    encode_file(file_path_new, vae, device, BENCHMARK_DIR, global_tasks)

if __name__ == "__main__":
    main()