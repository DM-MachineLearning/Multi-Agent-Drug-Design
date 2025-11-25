import torch.nn as nn

class BaseGenerator(nn.Module):
    """
    Abstract generator interface.

    Subclasses should implement:
    - sample(batch_size, condition=None, return_latent=False)
      returning (smiles_list, log_probs, latent_embeddings)
    """

    def sample(self, batch_size: int, condition=None, return_latent: bool = False):
        raise NotImplementedError

    def reinforce_update(self, log_probs, rewards):
        """
        Default REINFORCE-style loss.

        log_probs: tensor [batch]
        rewards:  tensor [batch]
        """
        import torch
        loss = -(log_probs * rewards).mean()
        loss.backward()
        return float(loss.item())
