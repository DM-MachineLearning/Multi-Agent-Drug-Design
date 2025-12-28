import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW

# --- Import from your existing file ---
from Datasets.SMILES import SMILESDataset

# --- Configuration Loading ---
_config_path = Path(__file__).resolve().parent / "config.yaml"
VAE_Model_Path: Optional[str] = None

if _config_path.is_file():
    try:
        with _config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            VAE_Model_Path = cfg.get("VAE_Model_Path") # e.g., a path to a .pt file
    except Exception:
        VAE_Model_Path = None

# ==========================================
# 1. The Neural Network Architecture
# ==========================================
class MolGRUVAE(nn.Module):
    """
    Standard GRU-based Variational Autoencoder for SMILES.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        """The 'Reparameterization Trick': z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask=None):
        # 1. Encode
        embedded = self.embedding(input_ids)
        # Pack padded sequence handling could go here, but keeping simple for now
        _, h_n = self.encoder_gru(embedded) # h_n: [1, batch, hidden]
        h_n = h_n.squeeze(0)
        
        # 2. Latent Space
        mu = self.fc_mu(h_n)
        logvar = self.fc_var(h_n)
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode (Teacher Forcing Mode)
        # Initialize decoder hidden state with latent vector z
        h_dec = self.decoder_input(z).unsqueeze(0) # [1, batch, hidden]
        
        # In training, we feed the *actual* input as the next step (Teacher Forcing)
        decoder_out, _ = self.decoder_gru(embedded, h_dec)
        logits = self.fc_out(decoder_out)
        
        return logits, mu, logvar

    def sample(self, max_len, start_token_idx, tokenizer, device, temp=1.0):
        """Inference Mode: Autoregressive generation from latent space"""
        batch_size = 1
        # Sample random z
        z = torch.randn(batch_size, self.latent_dim).to(device)
        h_dec = self.decoder_input(z).unsqueeze(0)
        
        # Start token
        curr_token = torch.tensor([[start_token_idx]], device=device)
        
        generated_ids = []
        
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            output, h_dec = self.decoder_gru(embed, h_dec)
            logits = self.fc_out(output.squeeze(1))
            
            # Sampling
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id:
                break
                
            generated_ids.append(token_id)
            curr_token = next_token
            
        return generated_ids, z

# ==========================================
# 2. The Management Class (Interface)
# ==========================================
class VAE:
    """
    A class representing a VAE model for drug generation.
    Matches the API of the GPT class.
    """

    def __init__(self, model_path: Optional[str] = VAE_Model_Path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, vocab_base="gpt2"):
        """
        Initializes the VAE architecture and loads weights if model_path exists.
        """
        # 1. Load a standard tokenizer (using GPT2's BPE is fine for SMILES, or a custom one)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_base, use_fast=True)
            self._fix_tokenizer()
        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer: {e}")

        # 2. Initialize Architecture
        vocab_size = len(self.tokenizer)
        self.model = MolGRUVAE(vocab_size=vocab_size).to(self.device)

        # 3. Load Weights (if they exist)
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading VAE weights from {self.model_path}...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("No existing weights found. Initialized fresh VAE model.")

        self.model.eval()
        return self.model

    def _fix_tokenizer(self):
        """Ensures pad/bos/eos tokens exist."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            if self.tokenizer.pad_token == "[PAD]":
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # Ensure we have a start token concept
        if self.tokenizer.bos_token is None:
             self.tokenizer.add_special_tokens({"bos_token": "[BOS]"})

    def generate_molecule(self, max_length: int = 100, temperature: float = 0.8) -> str:
        """
        Generates a SMILES string by sampling from the latent space.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        # Determine start token (BOS preferred, else EOS/PAD)
        start_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        
        with torch.no_grad():
            token_ids, z = self.model.sample(
                max_len=max_length,
                start_token_idx=start_id,
                tokenizer=self.tokenizer,
                device=self.device,
                temp=temperature
            )

        return token_ids, z

    def fine_tune(self, dataset_path: str, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
        """
        Trains the VAE on a SMILES dataset.
        """
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Path not found: {ds_path}")
        
        # Handle directory vs file
        target_file = ds_path
        if ds_path.is_dir():
            files = list(ds_path.glob("**/*.smi")) + list(ds_path.glob("**/*.txt"))
            target_file = files[0] if files else None
            if not target_file: raise FileNotFoundError("No data files found.")

        # Ensure model is ready
        if self.model is None:
            self.load_model()
        
        # Resize embeddings if tokenizer changed (e.g. added BOS/PAD)
        self.model.embedding = nn.Embedding(len(self.tokenizer), self.model.embedding.embedding_dim).to(self.device)
        self.model.fc_out = nn.Linear(self.model.hidden_dim, len(self.tokenizer)).to(self.device)

        # Data Loading
        print(f"Loading data from {target_file}...")
        dataset = SMILESDataset(target_file, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        print(f"Starting VAE training for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0
            # KL Annealing: Slowly increase KL weight to prevent posterior collapse
            kl_weight = min(1.0, (epoch + 1) / (epochs // 2)) 

            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                # Labels for VAE reconstruction (usually same as input)
                labels = batch["labels"].to(self.device) 

                optimizer.zero_grad()
                
                # Forward Pass
                recon_logits, mu, logvar = self.model(input_ids)
                
                # --- VAE LOSS CALCULATION ---
                # 1. Reconstruction Loss (Cross Entropy)
                # Flatten logits: [batch * seq_len, vocab_size]
                # Flatten labels: [batch * seq_len]
                recon_loss = F.cross_entropy(
                    recon_logits.view(-1, len(self.tokenizer)), 
                    labels.view(-1), 
                    ignore_index=-100 # Ignore padding
                )
                
                # 2. KL Divergence
                # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_div = kl_div / input_ids.size(0) # Normalize by batch size
                
                loss = recon_loss + (kl_weight * kl_div)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | KL Weight: {kl_weight:.2f}")

        # Save Model
        timestamp = int(time.time())
        save_dir = Path.cwd() / f"trained_vae_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch weights (since it's not a HF AutoModel)
        torch.save(self.model.state_dict(), save_dir / "vae_weights.pt")
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")
        return str(save_dir)