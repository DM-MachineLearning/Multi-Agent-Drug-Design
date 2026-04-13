import torch
import torch.nn as nn
import torch.nn.functional as F

class MolGRUVAE(nn.Module):
    """
    Standard GRU-based Variational Autoencoder for SMILES.
    Updated: Bidirectional Encoder + Word Dropout support.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # --- 1. Bidirectional Encoder (Match your checkpoint) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # bidirectional=True doubles the hidden state output size
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # FC layers must accept (hidden_dim * 2) because of bidirectionality
        # Checkpoint expects 512 input features (256 * 2)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # --- 2. Decoder (Unidirectional) ---
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, word_dropout_rate=0.0):
        # --- ENCODING ---
        embedded = self.embedding(input_ids)
        
        # Run GRU. h_n shape: [2, batch, hidden] (Forward + Backward layers)
        _, h_n = self.encoder_gru(embedded)
        
        # Concatenate the two directions: [batch, hidden * 2]
        # h_n[0] is forward, h_n[1] is backward
        h_n_concat = torch.cat((h_n[0], h_n[1]), dim=1)
        
        # Latent Space
        mu = self.fc_mu(h_n_concat)
        logvar = self.fc_var(h_n_concat)
        
        # Clamp for numerical stability
        logvar = torch.clamp(logvar, max=10) 
        z = self.reparameterize(mu, logvar)
        
        # --- DECODING ---
        h_dec = self.decoder_input(z).unsqueeze(0) # [1, batch, hidden]
        decoder_input_emb = embedded[:, :-1, :] # Teacher Forcing
        
        # Word Dropout Logic
        if self.training and word_dropout_rate > 0:
            # Drop tokens with probability 'rate'
            mask = torch.rand(decoder_input_emb.shape[:2], device=input_ids.device) > word_dropout_rate
            mask = mask.unsqueeze(2) # Expand to cover embedding dim
            decoder_input_emb = decoder_input_emb * mask
            
        decoder_out, _ = self.decoder_gru(decoder_input_emb, h_dec)
        logits = self.fc_out(decoder_out)
        
        return logits, mu, logvar

    # Needed for Inference / Gradient Ascent
    def sample(self, max_len, start_token_idx, tokenizer, device, temp=1.0, z=None):
        batch_size = 1
        # Accept external z for Gradient Ascent
        if z is None:
            z = torch.randn(batch_size, self.latent_dim).to(device)
        
        h_dec = self.decoder_input(z).unsqueeze(0)
        curr_token = torch.tensor([[start_token_idx]], device=device)
        generated_ids = []
        
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            output, h_dec = self.decoder_gru(embed, h_dec)
            logits = self.fc_out(output.squeeze(1))
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id: break
            generated_ids.append(token_id)
            curr_token = next_token
        return generated_ids