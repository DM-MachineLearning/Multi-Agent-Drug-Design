import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp

# Ensure spawn for CUDA compatibility with num_workers
if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

from Generators.VAE import VAE

def setup_logger(log_file):
    logger = logging.getLogger("VAETraining")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
        c_format = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    return logger

def get_kl_weight_monotonic(epoch, warmup_epochs=50, max_beta=0.02):
    """Slower, gentler warmup for high-capacity models."""
    return min(max_beta, (epoch / warmup_epochs) * max_beta)

def train_transformer():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    DEVICE = torch.device("cuda:0") 
    
    LATENT_DIM = 128
    RUN_ID = f"transformer_stable_z128_{int(time.time())}"
    SAVE_DIR = Path(f"./trained_models/{RUN_ID}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(SAVE_DIR / "training.log")
    logger.info(f"🛡️ Starting STABLE Run: {RUN_ID}")
    
    # HYPERPARAMETERS
    EPOCHS = 100
    BATCH_SIZE = 4096  # Reduced for gradient stability
    LR = 1e-4          # Reduced for safer convergence
    
    vae_wrapper = VAE(model_path=None)
    vae_wrapper.device = DEVICE
    vae_wrapper.load_model(vocab_base="vocab.json", model_type="transformer")
    
    # Structure setup
    vae_wrapper.model.latent_dim = LATENT_DIM
    vae_wrapper.model.fc_mu = nn.Linear(vae_wrapper.model.d_model, LATENT_DIM)
    vae_wrapper.model.fc_var = nn.Linear(vae_wrapper.model.d_model, LATENT_DIM)
    vae_wrapper.model.latent_to_decoder = nn.Linear(LATENT_DIM, vae_wrapper.model.d_model)
    
    model = nn.DataParallel(vae_wrapper.model).to(DEVICE)
    
    # Data Loading to CPU
    full_tensor = torch.load("data/chembl_train.pt", map_location='cpu', weights_only=False)
    loader = DataLoader(TensorDataset(full_tensor), batch_size=BATCH_SIZE, 
                        shuffle=True, num_workers=32, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2) # Increased weight decay for regularization
    scaler = torch.amp.GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # More aggressive schedule

    tokenizer = vae_wrapper.tokenizer
    pad_id = tokenizer.pad_token_id or 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_kl, total_recon = 0.0, 0.0, 0.0
        steps = 0
        kl_weight = get_kl_weight_monotonic(epoch)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for (batch,) in pbar:
            input_ids = batch.long().to(DEVICE)
            labels = input_ids.clone()
            labels[labels == pad_id] = -100

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                recon_logits, mu, logvar = model(input_ids)
                logits = recon_logits.reshape(-1, recon_logits.size(-1))
                targets = labels[:, 1:].reshape(-1)
                
                recon_loss = F.cross_entropy(logits, targets, ignore_index=-100)
                kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
                kl_div = -0.5 * torch.sum(kl_element, dim=1).mean()
                loss = recon_loss + kl_weight * kl_div
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue 

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            # More aggressive clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_kl += kl_div.item()
            total_recon += recon_loss.item()
            steps += 1
            
        avg_recon = total_recon / steps
        scheduler.step(avg_recon)
        logger.info(f"Epoch {epoch+1:03d} | LR: {optimizer.param_groups[0]['lr']:.2e} | Beta: {kl_weight:.4f} | Recon: {avg_recon:.4f} | KL: {total_kl/steps:.4f}")
        
        # Save latest
        torch.save(model.module.state_dict(), SAVE_DIR / "vae_weights_latest.pt")

if __name__ == "__main__":
    train_transformer()