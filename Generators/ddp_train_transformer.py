import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from pathlib import Path
from tqdm import tqdm

from Generators.VAE import VAE

def setup_logger(log_file, local_rank):
    """Ensure only the master process (rank 0) writes to the logs."""
    logger = logging.getLogger("VAETraining")
    logger.setLevel(logging.INFO)
    if local_rank == 0 and not logger.handlers:
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
    """Gradually introduce the KL penalty over 50 epochs."""
    return min(max_beta, (epoch / warmup_epochs) * max_beta)

def init_weights(m):
    """Xavier Initialization to break the random guessing plateau."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_transformer_ddp():
    # 1. Initialize Distributed Process Group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")
    
    # H200 TF32 Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    
    LATENT_DIM = 128
    RUN_ID = f"transformer_ddp_z{LATENT_DIM}_{int(time.time())}"
    SAVE_DIR = Path(f"./trained_models/{RUN_ID}")
    
    if local_rank == 0:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    
    logger = setup_logger(SAVE_DIR / "training.log", local_rank)
    if local_rank == 0:
        logger.info(f"🚀 Starting DDP Run: {RUN_ID} on {dist.get_world_size()} GPUs")
    
    # 2. Hyperparameters
    EPOCHS = 100
    PER_GPU_BATCH_SIZE = 1024 
    START_LR = 5e-4 
    
    # 3. Model Architecture Setup
    vae_wrapper = VAE(model_path=None)
    vae_wrapper.device = DEVICE
    vae_wrapper.load_model(vocab_base="vocab.json", model_type="transformer")
    
    vae_wrapper.model.latent_dim = LATENT_DIM
    vae_wrapper.model.fc_mu = nn.Linear(vae_wrapper.model.d_model, LATENT_DIM)
    vae_wrapper.model.fc_var = nn.Linear(vae_wrapper.model.d_model, LATENT_DIM)
    vae_wrapper.model.latent_to_decoder = nn.Linear(LATENT_DIM, vae_wrapper.model.d_model)
    
    # Apply the weight initialization
    vae_wrapper.model.apply(init_weights)
    
    model = vae_wrapper.model.to(DEVICE)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 4. Data Loading
    full_tensor = torch.load("data/chembl_train.pt", map_location='cpu', weights_only=False)
    dataset = TensorDataset(full_tensor)
    
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=PER_GPU_BATCH_SIZE, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 5. Optimizers and Schedulers
    optimizer = AdamW(model.parameters(), lr=START_LR, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    
    # LR Warmup for the first 5 epochs
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 1.0
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    pad_id = vae_wrapper.tokenizer.pad_token_id or 0

    # 6. Training Loop
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        
        total_loss, total_kl, total_recon = 0.0, 0.0, 0.0
        steps = 0
        kl_weight = get_kl_weight_monotonic(epoch)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}") if local_rank == 0 else loader
        
        for (batch,) in pbar:
            input_ids = batch.long().to(DEVICE)
            labels = input_ids.clone()
            labels[labels == pad_id] = -100

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            total_kl += kl_div.detach()
            total_recon += recon_loss.detach()
            steps += 1
            
        # Synchronize metrics across all GPUs
        dist.all_reduce(total_recon, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_kl, op=dist.ReduceOp.SUM)
        
        global_steps = steps * dist.get_world_size()
        avg_recon = total_recon.item() / global_steps
        avg_kl = total_kl.item() / global_steps
        
        # Step Schedulers
        lr_scheduler.step()
        if epoch >= 5:  # Let the warmup finish before allowing the plateau scheduler to intervene
            plateau_scheduler.step(avg_recon)
        
        # Logging & Checkpointing (Master Process Only)
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | Beta: {kl_weight:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
            torch.save(model.module.state_dict(), SAVE_DIR / "vae_weights_latest.pt")
            
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    train_transformer_ddp()