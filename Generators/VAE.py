import os
import yaml
import torch
from pathlib import Path
from typing import Optional
from rdkit import RDLogger

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerFast

from Datasets.SMILESDataset import SMILESDataset
from Datasets.BinarySMILESDataset import BinarySMILESDataset
from Generators.metrics import token_reconstruction_accuracy
from .MolGRUVAE import MolGRUVAE


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# --- Configuration Loading ---
_config_path = Path(__file__).resolve().parent / "config.yaml"
VAE_Model_Path: Optional[str] = None

if _config_path.is_file():
    try:
        with _config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            VAE_Model_Path = cfg.get("VAE_Model_Path") 
    except Exception:
        VAE_Model_Path = None

class VAE:
    """
    Wrapper class for the MolGRUVAE model, providing high-level functionality for loading, training, and generating molecular SMILES strings.

    This class manages the tokenizer, model initialization, distributed training support, and inference for molecular generation using a Variational Autoencoder.

    Attributes:
        model_path (Optional[str]): Path to the pre-trained model weights.
        model (Optional[MolGRUVAE]): The underlying VAE model instance.
        tokenizer (Optional[PreTrainedTokenizerFast]): The tokenizer for encoding/decoding SMILES strings.
        local_rank (int): The local rank for distributed training (0 if not distributed).
        device (torch.device): The device (CPU or GPU) on which the model runs.

    Args:
        model_path (Optional[str]): Path to the model weights file. If None, uses the default path from config.
    """
    def __init__(self, model_path: Optional[str] = VAE_Model_Path):
        """
        Initializes the VAE wrapper class.

        Sets up the model path, initializes model and tokenizer to None, and determines the device and local rank
        for distributed training support.

        Args:
            model_path (Optional[str]): Path to the pre-trained model weights. If None, uses the default from config.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, vocab_base=None):
        """
        Loads the tokenizer and initializes the VAE model.

        Loads a tokenizer from a local JSON file or a pre-trained model, initializes the MolGRUVAE model
        with the appropriate vocabulary size, and optionally loads pre-trained weights if available.

        Args:
            vocab_base (str, optional): Path to a local tokenizer JSON file or name of a pre-trained tokenizer.
                                       If None, defaults to 'gpt2'.
        """
        if vocab_base and vocab_base.endswith('.json'):
            if self.local_rank == 0:
                print(f"✅ Loading LOCAL tokenizer from {vocab_base}")
            
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=vocab_base,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                vocab_base, 
                local_files_only=True if vocab_base != "gpt2" else False
            )
        
        vocab_size = len(self.tokenizer)
        self.model = MolGRUVAE(vocab_size=vocab_size).to(self.device)

        if self.model_path and Path(self.model_path).exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # --- SIZE CHECK ---
            weight_vocab_size = state_dict['embedding.weight'].shape[0]
            if weight_vocab_size != vocab_size:
                if self.local_rank == 0:
                    print(f"⚠️ Skipping weight load: Vocab size mismatch! "
                        f"(Weights: {weight_vocab_size}, Model: {vocab_size})")
            else:
                self.model.load_state_dict(state_dict)
                if self.local_rank == 0: print("✅ Weights loaded successfully.")

    def _fix_tokenizer(self):
        """
        Fixes missing special tokens in the tokenizer.

        Ensures that pad_token and bos_token are set, adding them if necessary.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            if self.tokenizer.pad_token == "[PAD]":
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.bos_token is None:
             self.tokenizer.add_special_tokens({"bos_token": "[BOS]"})

    def generate_molecule(self, num_samples=100, max_length: int=100, temperature: float=0.8) -> list:
        """
        Generates molecular SMILES strings using the trained VAE model.

        Samples from the latent space and decodes to generate new SMILES strings.

        Args:
            num_samples (int, optional): Number of SMILES strings to generate. Default is 100.
            max_length (int, optional): Maximum length of generated sequences. Default is 100.
            temperature (float, optional): Sampling temperature for generation. Default is 0.8.

        Returns:
            list: List of generated SMILES strings.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inference_model = self.model.module if hasattr(self.model, "module") else self.model
        inference_model.eval()
        
        start_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        generated = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                token_ids = inference_model.sample(
                    max_len=max_length,
                    start_token_idx=start_id,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    temp=temperature
                )
                smi = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                smi = smi.replace(" ", "")
                generated.append(smi)
            
        return generated

    def fine_tune(self, dataset_path: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, start_epoch: int = 0, save_dir: str = "./trained_vae"):
        """
        Fine-tunes the VAE model on a SMILES dataset with support for distributed training.

        Trains the model using reconstruction loss and KL divergence, with cyclic annealing of the KL weight.
        Supports both text (.txt, .smi) and binary (.npy) datasets. Automatically saves checkpoints and the latest model.

        Args:
            dataset_path (str): Path to the dataset file or directory containing SMILES data.
            epochs (int, optional): Number of training epochs. Default is 10.
            batch_size (int, optional): Batch size for training. Default is 32.
            lr (float, optional): Initial learning rate. Default is 1e-3.
            start_epoch (int, optional): Epoch to start training from (for resuming). Default is 0.
            save_dir (str, optional): Directory to save model checkpoints. Default is "./trained_vae".

        Returns:
            str: Path to the save directory.

        Raises:
            FileNotFoundError: If the dataset path does not exist or no valid files are found.
        """
        is_distributed = False
        if "WORLD_SIZE" in os.environ:
            is_distributed = int(os.environ["WORLD_SIZE"]) > 1
        
        if is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Path not found: {ds_path}")
        
        if ds_path.is_dir():
            files = list(ds_path.glob("**/*.npy")) + list(ds_path.glob("**/*.smi")) + list(ds_path.glob("**/*.txt"))
            target_file = files[0] if files else None
            if not target_file: raise FileNotFoundError("No files found.")
        else:
            target_file = ds_path

        self.model.to(self.device)
        if is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        if str(target_file).endswith('.npy'):
            if self.local_rank == 0:
                print(f"⚡ Loading BINARY dataset from {target_file}")
            
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            dataset = BinarySMILESDataset(target_file, pad_token_id=pad_id)
            
            num_workers = 4 
            prefetch_factor = 2
        else:
            if self.local_rank == 0:
                print(f"📖 Loading TEXT dataset from {target_file}")
            dataset = SMILESDataset(target_file, self.tokenizer)
            
            num_workers = 16
            prefetch_factor = 4
        
        sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None), 
            sampler=sampler,
            num_workers=num_workers,       
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor 
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler('cuda')
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-5:
            target_lr = 1e-4
            if self.local_rank == 0:
                print(f"⚠️ DETECTED LOW LR ({current_lr:.2e})... BOOSTING TO {target_lr}!")
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=25,    
            verbose=True
        )

        if self.local_rank == 0:
            print(f"Starting VAE training on {dist.get_world_size() if is_distributed else 1} GPUs...")

        def get_kl_weight(epoch, total_epochs):
            # Cycle every 20 epochs (Keep this, it's working well)
            cycle_len = 20
            pos = epoch % cycle_len

            MAX_BETA = 0.15
            
            # Linear ramp from 0.0 to MAX_BETA over the first half of the cycle
            ramp_len = cycle_len // 2
            ratio = pos / ramp_len
            
            # Calculate weight
            weight = min(MAX_BETA, ratio * MAX_BETA)
            
            return weight

        for epoch in range(start_epoch, epochs):
            if is_distributed:
                sampler.set_epoch(epoch)
            
            self.model.train()
            total_loss = torch.zeros(1, device=self.device)
            total_recon_acc = torch.zeros(1, device=self.device)
            total_kl = torch.zeros(1, device=self.device)
            steps = 0
            
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) 

                # --- WORD DROPOUT (Token Masking) ---
                if self.model.training:
                    # 25% Masking (Safer than 50% for initial stability)
                    prob_mask = torch.rand(input_ids.shape, device=self.device) < 0.25
                    masked_input = input_ids.clone()
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    masked_input[prob_mask] = unk_id
                else:
                    masked_input = input_ids

                # Safety Check for illegal IDs
                if is_distributed:
                     # Access .module for DDP wrapped models
                    out_features = self.model.module.fc_out.out_features
                else:
                    out_features = self.model.fc_out.out_features

                # Only check max ID occasionally to save time, or if paranoid check every batch
                if input_ids.max().item() >= out_features:
                     print(f"❌ ILLEGAL ID DETECTED. Max: {input_ids.max().item()}")
                     continue

                optimizer.zero_grad(set_to_none=True)
                
                with autocast("cuda", dtype=torch.float16):
                    recon_logits, mu, logvar = self.model(masked_input)
                    
                    # Clamp logits to prevent NaN
                    recon_logits = torch.clamp(recon_logits, min=-20.0, max=20.0)
                    
                    mu, logvar = mu.float(), logvar.float()
                    
                    logits = recon_logits.reshape(-1, recon_logits.size(-1))
                    targets = labels[:, 1:].reshape(-1)
                    
                    recon_loss = F.cross_entropy(logits, targets, ignore_index=-100)
                    
                    # Stable KL
                    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
                    kl_div = -0.5 * torch.sum(kl_element, dim=1).mean()

                    # Free Bits (Min 2.0 nats)
                    kl_loss_clamped = torch.max(kl_div, torch.tensor(2.0, device=self.device))
                    
                    kl_weight = get_kl_weight(epoch, epochs)
                    loss = recon_loss + kl_weight * kl_loss_clamped
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.local_rank == 0:
                        print(f"⚠️ NaN detected! Loss: {loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue 

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.detach()
                total_kl += kl_div.detach()
                
                with torch.no_grad():
                    pred_ids = torch.argmax(recon_logits, dim=-1)
                    acc = token_reconstruction_accuracy(pred_ids, labels[:, 1:], pad_token_id=-100)
                    total_recon_acc += acc
                steps += 1

            # --- Aggregation ---
            if is_distributed:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_kl, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_recon_acc, op=dist.ReduceOp.SUM)
                
            world_size = dist.get_world_size() if is_distributed else 1
            global_steps = steps * world_size
            
            if global_steps > 0:
                avg_loss = total_loss.item() / global_steps
                avg_acc = total_recon_acc.item() / global_steps
                avg_kl = total_kl.item() / global_steps
                
                if self.local_rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    # print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | KL: {avg_kl:.4f} | LR: {current_lr:.6f}")
                
                scheduler.step(avg_acc)
                
            if self.local_rank == 0:
                print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | Recon Acc: {avg_acc:.3f}")
                
                # Unwrap DDP model for saving
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                
                # --- AUTO-SAVE LOGIC ---
                # Save 'latest' every epoch
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model_to_save.state_dict(), save_path / "vae_weights.pt")
                self.tokenizer.save_pretrained(save_path)
                
                # Save checkpoints every 10 epochs
                if (epoch + 1) % 50 == 0:
                     ckpt_path = save_path / f"checkpoint_epoch_{epoch+1}"
                     ckpt_path.mkdir(parents=True, exist_ok=True)
                     torch.save(model_to_save.state_dict(), ckpt_path / "vae_weights.pt")
                     self.tokenizer.save_pretrained(ckpt_path)
                     print(f"💾 Checkpoint saved to {ckpt_path}")
                else:
                     print(f"💾 Model updated in {save_path}")

            if is_distributed:
                dist.barrier()
        
        if is_distributed:
            dist.destroy_process_group()
                
        return str(Path.cwd() / "vae_ckpts")