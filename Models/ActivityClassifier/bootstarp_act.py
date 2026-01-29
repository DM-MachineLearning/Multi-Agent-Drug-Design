import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# --- CONFIG ---
MODEL_PATH = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"
BATCH_SIZE = 512 

class LatentPredictor(nn.Module):
    def __init__(self, input_dim=128):
        super(LatentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        return self.net(z)

def safe_load_latent(data_path):
    """
    Intelligently inspects the data structure to extract latent vectors.
    Handles:
    1. List of Tensors (Standard)
    2. List of Numpy Arrays (Standard)
    3. Numpy Object Arrays (The 'numpy.object_' error)
    4. List of Dictionaries (The 'float() argument' error)
    """
    try:
        content = torch.load(data_path, map_location='cpu')
        
        # Unwrap if it's a dictionary wrapper
        raw_data = content.get('data') if isinstance(content, dict) else content
        
        if raw_data is None:
            raise ValueError(f"Could not find data container in {data_path}")

        # --- INSPECTION LOGIC ---
        
        # 1. Convert Numpy Object Arrays to List for easier processing
        if isinstance(raw_data, np.ndarray) and raw_data.dtype == np.object_:
            raw_data = raw_data.tolist()

        # 2. If it's a simple Tensor or Numpy Float Array, return immediately
        if torch.is_tensor(raw_data):
            return raw_data.float()
        if isinstance(raw_data, np.ndarray) and raw_data.dtype != np.object_:
            return torch.from_numpy(raw_data).float()

        # 3. Handle Lists (The tricky part)
        if isinstance(raw_data, list):
            if len(raw_data) == 0: return None
            
            first_item = raw_data[0]
            
            # Case A: List of Dictionaries (Fixes rl_graph error)
            if isinstance(first_item, dict):
                # Try common keys for latent vectors
                for key in ['z', 'latent', 'embedding', 'features', 'x']:
                    if key in first_item:
                        # Extract specific key from every dict
                        vecs = [torch.as_tensor(item[key]) for item in raw_data]
                        return torch.stack(vecs).float()
                raise ValueError(f"List of dicts found, but unknown key. Keys: {first_item.keys()}")

            # Case B: List of Tensors/Arrays (Fixes moflow/chemvae error)
            if torch.is_tensor(first_item) or isinstance(first_item, np.ndarray):
                # Convert item-by-item to avoid 'numpy.object_' crash
                vecs = [torch.as_tensor(x) for x in raw_data]
                return torch.stack(vecs).float()

        # 4. Fallback
        return torch.tensor(raw_data).float()

    except Exception as e:
        print(f"❌ Failed to load {Path(data_path).name}: {e}")
        return None

def analyze_activity_hit_rate(model, device, data_path, num_samples=1000, trials=10):
    # Load with new robust function
    X = safe_load_latent(data_path)
    if X is None: return None

    # Safety: Ensure shape is [N, 128]
    if X.dim() > 2:
        X = X.view(X.size(0), -1)

    total_available = X.shape[0]
    actual_samples = min(total_available, num_samples)
    
    # print(f"   ↳ Loaded {total_available} samples. Shape: {X.shape}")

    trial_percentages = []
    
    for trial in range(trials):
        # Use simple randperm for indexing tensors
        indices = torch.randperm(total_available)[:actual_samples]
        z_sample = X[indices].to(device)

        with torch.no_grad():
            probs = model(z_sample)
            preds = (probs > 0.6).float()
            
            hit_rate = (preds.sum().item() / actual_samples) * 100
            trial_percentages.append(hit_rate)

    mean_hit = np.mean(trial_percentages)
    std_hit = np.std(trial_percentages)
    
    return {
        "Benchmark": Path(data_path).name,
        "Total Mols": total_available,
        "Active %": f"{mean_hit:.2f}%",
        "Std Dev": f"± {std_hit:.2f}%"
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    benchmarks = [
        # "Benchmarks/rl_graph_pretrained_num.pt",
        # "Benchmarks/moflow_pretrained.pt",
        # "Benchmarks/chemvae_4000_pretrained.pt",
        # "Benchmarks/molobj_pretrained.pt",
        # "Benchmarks_new/rnn_smiles.pt"
        # "Benchmarks_new/molgan_15k_smiles.pt"
        # "Benchmarks_new/bimodal.pt"
        # "Benchmarks_new/organ.pt"
        "Benchmarks_new/generated1.pt"
    ]

    model = LatentPredictor().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"🚀 Model loaded on {device}.")
    else:
        print("❌ Model Checkpoint not found!")
        sys.exit()

    results = []
    print("-" * 80)

    for bf in benchmarks:
        if os.path.exists(bf):
            print(f"📂 Processing {bf}...")
            res = analyze_activity_hit_rate(model, device, bf, num_samples=100, trials=10)
            if res:
                results.append(res)
        else:
            print(f"⏩ File not found: {bf}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"{'GENERATED MOLECULE ACTIVITY REPORT':^80}")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)