import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from torch.nn.attention import sdpa_kernel, SDPBackend

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Generators.VAE import VAE

def get_kl_weight(epoch, total_epochs):
    cycle_len = 20
    pos = epoch % cycle_len
    MAX_BETA = 0.15
    ramp_len = cycle_len // 2
    ratio = pos / ramp_len
    weight = min(MAX_BETA, ratio * MAX_BETA)
    return weight

def train_transformer():
    # --- MULTI-GPU CONFIGURATION ---
    # Strictly isolate to GPUs 1 and 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    
    DATASET_PATH = "data/chembl_train.pt"
    VOCAB_PATH = "vocab.json"
    SAVE_DIR = "./trained_transformer_vae"
    EPOCHS = 100
    # Global batch size: split across 2 GPUs
    BATCH_SIZE = 2048 
    LR = 1e-4
    
    # In visibility-restricted mode, indices 1,2 become 0,1
    DEVICE = torch.device("cuda:0") 

    # 1. Initialize VAE Wrapper
    vae_wrapper = VAE(model_path=None)
    vae_wrapper.device = DEVICE
    
    # 2. Load Transformer Model
    vae_wrapper.load_model(vocab_base=VOCAB_PATH, model_type="transformer")
    
    # --- UPGRADE: 512-dim Latent Space ---
    vae_wrapper.model.latent_dim = 512
    vae_wrapper.model.fc_mu = nn.Linear(vae_wrapper.model.d_model, 512)
    vae_wrapper.model.fc_var = nn.Linear(vae_wrapper.model.d_model, 512)
    vae_wrapper.model.latent_to_decoder = nn.Linear(512, vae_wrapper.model.d_model)
    
    model = vae_wrapper.model
    
    # --- MULTI-GPU: DataParallel ---
    if torch.cuda.device_count() > 1:
        print(f"📡 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model.to(DEVICE)
    
    # Speed Optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print(f"🚀 Compiling high-capacity model (512-dim latent)...")
    # torch.compile might have issues with DataParallel, but let's try. 
    # If it fails, I'll remove compile in next turn.
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"⚠️ Compilation failed, proceeding without: {e}")
    
    # 3. Load Data to GPU
    # Note: We load to DEVICE (cuda:0, which is physically GPU 1)
    full_tensor = torch.load(DATASET_PATH, map_location=DEVICE, weights_only=False)
    dataset = TensorDataset(full_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    tokenizer = vae_wrapper.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_kl, total_recon = 0.0, 0.0, 0.0
        steps = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for (batch,) in pbar:
                input_ids = batch.long().to(DEVICE)
                labels = input_ids.clone()
                labels[labels == pad_id] = -100

                # Efficient Word Dropout
                prob_mask = torch.rand(input_ids.shape, device=DEVICE) < 0.15
                masked_input = input_ids.masked_fill(prob_mask, unk_id)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    recon_logits, mu, logvar = model(masked_input)
                    
                    logits = recon_logits.reshape(-1, recon_logits.size(-1))
                    targets = labels[:, 1:].reshape(-1)
                    
                    recon_loss = F.cross_entropy(logits, targets, ignore_index=-100)
                    
                    # --- IMPROVEMENT: Free Bits (lambda_fb = 2.0) ---
                    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
                    kl_div_raw = -0.5 * torch.sum(kl_element, dim=1)
                    kl_loss = torch.clamp(kl_div_raw - 2.0, min=0).mean()
                    
                    kl_weight = get_kl_weight(epoch, EPOCHS)
                    loss = recon_loss + kl_weight * kl_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue 

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_kl += kl_div_raw.mean().item()
                total_recon += recon_loss.item()
                steps += 1
                
                if steps % 20 == 0:
                    pbar.set_postfix({'L': f"{loss.item():.2f}", 'R': f"{recon_loss.item():.2f}"})

        avg_recon = total_recon / steps
        scheduler.step(avg_recon)
        print(f"Epoch {epoch+1} | Loss: {total_loss/steps:.4f} | Recon: {avg_recon:.4f} | KL: {total_kl/steps:.4f}")
        
        # Save raw model state (extracting from DataParallel/Compiled)
        raw_model = model
        if hasattr(raw_model, "_orig_mod"): raw_model = raw_model._orig_mod
        if isinstance(raw_model, nn.DataParallel): raw_model = raw_model.module
            
        torch.save(raw_model.state_dict(), save_path / "vae_weights.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(raw_model.state_dict(), save_path / f"vae_weights_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train_transformer()
