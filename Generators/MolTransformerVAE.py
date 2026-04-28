import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MolTransformerVAE(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, latent_dim=128, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # FIX 1: norm_first=True (Pre-LN) prevents FP16 gradient explosions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            batch_first=True, activation='gelu', dropout=0.1, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)

        self.latent_to_decoder = nn.Linear(latent_dim, d_model)
        
        # FIX 1: norm_first=True (Pre-LN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, activation='gelu', dropout=0.1, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, word_dropout_rate=0.0):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        src_key_padding_mask = (input_ids == 0)
        embed = self.embedding(input_ids) * math.sqrt(self.d_model)
        embed = self.pos_encoder(embed)
        
        enc_output = self.transformer_encoder(embed, src_key_padding_mask=src_key_padding_mask)
        
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        pooled = (enc_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        mu = self.fc_mu(pooled)
        logvar = torch.clamp(self.fc_var(pooled), max=10)
        z = self.reparameterize(mu, logvar)
        
        tgt_input = input_ids[:, :-1]
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        tgt_key_padding_mask = (tgt_input == 0)
        
        tgt_embed = self.embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # FIX 2: Do NOT repeat the memory. Pass the single projected vector.
        memory = self.latent_to_decoder(z).unsqueeze(1) # Shape: (batch, 1, d_model)
        
        dec_output = self.transformer_decoder(
            tgt_embed, memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        logits = self.fc_out(dec_output)
        return logits, mu, logvar

    def sample(self, max_len, start_token_idx, tokenizer, device, temp=1.0, z=None):
        self.eval()
        if z is None:
            z = torch.randn(1, self.latent_dim).to(device)
        else:
            z = z.to(device)
        
        z_projected = self.latent_to_decoder(z)
        generated = torch.tensor([[start_token_idx]], device=device)
        
        # FIX 2: Apply the same memory fix to the sampling logic
        memory = z_projected.unsqueeze(1) # Shape: (1, 1, d_model)
        
        for _ in range(max_len):
            curr_seq_len = generated.size(1)
            tgt_mask = self.generate_square_subsequent_mask(curr_seq_len, device)
            tgt_embed = self.embedding(generated) * math.sqrt(self.d_model)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output[:, -1, :]) / temp
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        return generated[0].tolist()