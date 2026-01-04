import torch
import torch.nn as nn
import torch.nn.functional as F

class MolGRUVAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for generating molecular SMILES strings using GRU networks.

    This model consists of an encoder that maps input sequences to a latent space and a decoder that reconstructs the sequences from the latent representation. It uses GRU layers for sequence processing and implements the reparameterization trick for sampling from the latent distribution.

    Attributes:
        hidden_dim (int): The dimension of the hidden state in the GRU layers.
        latent_dim (int): The dimension of the latent space.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        embed_dim (int, optional): The dimension of the embedding layer. Default is 128.
        hidden_dim (int, optional): The dimension of the GRU hidden state. Default is 256.
        latent_dim (int, optional): The dimension of the latent space. Default is 128.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=0.1 if hidden_dim > 1 else 0.0)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=0.1 if hidden_dim > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.input_dropout = nn.Dropout(p=0.4) # Added for masking later :)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights and biases of the model parameters.

        This method applies Xavier uniform initialization to weight matrices (parameters with 2 or more dimensions),
        normal initialization with standard deviation 0.01 to 1D weight parameters, and constant initialization
        to 0.0 for bias parameters.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick for the VAE.

        This method samples from the latent distribution by transforming the mean and log-variance
        into a sample using the reparameterization trick, which allows backpropagation through the sampling process.

        Args:
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log-variance of the latent distribution.

        Returns:
            torch.Tensor: A sample from the latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the VAE model.

        Encodes the input sequence into a latent representation, samples from the latent space,
        and decodes back to logits for reconstruction.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask (not used in this implementation).

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Output logits of shape (batch_size, seq_len-1, vocab_size).
                - mu (torch.Tensor): Mean of the latent distribution.
                - logvar (torch.Tensor): Log-variance of the latent distribution.
        """
        embedded = self.embedding(input_ids)
        _, h_n = self.encoder_gru(embedded)
        h_n = h_n.squeeze(0)
        
        mu = torch.clamp(self.fc_mu(h_n), min=-10.0, max=10.0)
        logvar = torch.clamp(self.fc_var(h_n), min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        
        h_dec = self.decoder_input(z).unsqueeze(0)
        decoder_input_emb = embedded[:, :-1, :]

        decoder_input_emb = self.input_dropout(decoder_input_emb)
        
        decoder_out, _ = self.decoder_gru(decoder_input_emb, h_dec)
        logits = self.fc_out(decoder_out)
        
        return logits, mu, logvar

    def sample(self, max_len, start_token_idx, tokenizer, device, temp=1.0):
        """
        Performs autoregressive generation from the latent space.

        Samples a latent vector and generates a sequence of token IDs by autoregressively predicting the next token
        until the maximum length is reached or an end-of-sequence token is generated.

        Args:
            max_len (int): Maximum length of the generated sequence.
            start_token_idx (int): Index of the starting token (e.g., BOS token).
            tokenizer: The tokenizer object with eos_token_id.
            device (torch.device): Device to run the generation on.
            temp (float, optional): Temperature for sampling. Lower values make output more deterministic. Default is 1.0.

        Returns:
            list: List of generated token IDs.
        """
        batch_size = 1
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
            if token_id == tokenizer.eos_token_id:
                break
                
            generated_ids.append(token_id)
            curr_token = next_token
            
        return generated_ids
